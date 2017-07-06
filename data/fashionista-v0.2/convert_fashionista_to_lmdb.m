function convert_fashionista_to_lmdb
%CONVERT_FASHIONISTA_TO_LMDB
  %startup;
  load fashionista-v0.2.1/fashionista_v0.2.1.mat truths test_index;
  [train_index, val_index] = generateIndex(truths, test_index);
  createDatasets(truths(train_index), 'train.lmdb', 'train-gt.lmdb');
  createDatasets(truths(val_index), 'val.lmdb', 'val-gt.lmdb');
  createDatasets(truths(test_index), 'test.lmdb', 'test-gt.lmdb');
end

% function startup()
% %STARTUP Initialize the environment.
%   root = fileparts(mfilename('fullpath'));
%   cd(root);
%   run(fullfile(root, 'fashionista-v0.2.1', 'startup'));
%   lmdb_dir = fullfile(root, 'matlab-lmdb');
%   % Build LMDB package if not done.
%   if ~exist(lmdb_dir, 'dir')
%     lmdb_url = 'https://github.com/kyamagu/matlab-lmdb/archive/master.zip';
%     disp('Downloading matlab-lmdb');
%     urlwrite(lmdb_url, 'master.zip');
%     unzip('master.zip');
%     movefile('matlab-lmdb-master', lmdb_dir);
%     delete('master.zip');
%     addpath(lmdb_dir);
%     disp('Building matlab-lmdb');
%     lmdb.make();
%     mex -I/usr/local/include ...
%         -Imatlab-lmdb/include ...
%         -I../../build/src/caffe/proto/ ...
%         -L/usr/local/lib ...
%         CXXFLAGS="\$CXXFLAGS -std=c++11" ...
%         caffe_proto_.cc ...
%         ../../build/src/caffe/proto/caffe.pb.o ...
%         -lprotobuf;
%   else
%     addpath(lmdb_dir);
%   end
% end

function [train_index, val_index] = generateIndex(truths, test_index)
%GENERATEINDEX
  train_val_index = 1:numel(truths);
  train_val_index(test_index) = [];
  val_index = mod(1:numel(train_val_index), 10) == 0;
  train_index = train_val_index(~val_index);
  val_index = train_val_index(val_index);
  fprintf('Data size: (train, val, test) = (%g, %g, %g)\n', ...
          numel(train_index), numel(val_index), numel(test_index));
end

function createDatasets(truths, image_db_file, label_db_file)
%CREATEDATASETS Create LMDB of fashionista data
  if exist(image_db_file, 'dir') && exist(label_db_file, 'dir')
    return;
  end
  fprintf('Creating %s and %s\n', image_db_file, label_db_file);
  image_db = lmdb.DB(image_db_file, 'MAPSIZE', 1024^3);
  label_db = lmdb.DB(label_db_file, 'MAPSIZE', 1024^3);
  image_size = [600, 400, 3];
  image_transaction = image_db.begin();
  label_transaction = label_db.begin();
  try
    for i = 1:numel(truths)
      % Insert image JPG.
      image = imdecode(truths(i).image, 'jpg');
      if size(image, 3) == 1
        image = repmat(image, [1, 1, 3]);
        assert(all(size(image) == image_size));
        image = imencode(image, 'jpg');
      else
        image = truths(i).image;
      end
      image_transaction.put(num2str(truths(i).index), ...
                        caffe_proto_('toEncodedDatum', image, 0));
      % Insert mask PNG.
      superpixel_map = imdecode(truths(i).annotation.superpixel_map, 'png');
      superpixel_labels = uint8(truths(i).annotation.superpixel_labels) - 1;
      segmentation = superpixel_labels(superpixel_map);
      segmentation = imencode(segmentation, 'png');
      label_transaction.put(num2str(truths(i).index), ...
                            caffe_proto_('toEncodedDatum', segmentation, 0));
      if mod(i, 100) == 0
        fprintf('Imported %g of %g\n', i, numel(truths));
      end
    end
    image_transaction.commit();
    label_transaction.commit();
  catch exception
    disp(exception.getReport);
    image_transaction.abort();
    label_transaction.abort();
  end
end
