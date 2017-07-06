function convert_fashionista_to_lmdb(varargin)
%CONVERT_FASHIONISTA_TO_LMDB
%
% mex -Imatlab/matlab-lmdb/include -Ibuild/src/caffe/proto/ \
%     matlab/caffe_proto_.cc -outdir matlab/ \
%     build/src/caffe/proto/caffe.pb.o -lprotobuf
%
  options.image_db_file = '%s.lmdb';
  options.label_db_file = '%s-gt.lmdb';
  options.pose_db_file = '%s-pose.lmdb';
  options.attr_db_file = '%s-attr.lmdb';
  options.posemap_radius = 5;
  options.posemap_sigma = 1.5;
  options = getOptions(options, varargin{:});

  load fashionista-v1.0.mat truths test_index labels;

  % % Debug posemap here...
  % posemap = makePosemap(truths(1).pose, [600, 400], 7, 4);
  % posemap = reshape(posemap, [600, 400, 1, 14]);
  % montage(im2uint8(posemap), jet(255));
  % keyboard;

  splits = struct('name', {'train', 'val', 'test'}, ...
                  'index', {[], [], test_index});
  [splits(1).index, splits(2).index] = generateIndex(truths, test_index);

  exportLabelFile('labels.txt', labels);
  for i = 1:numel(splits)
    createDatasets(truths(splits(i).index), ...
                   labels, ...
                   sprintf(options.image_db_file, splits(i).name), ...
                   sprintf(options.label_db_file, splits(i).name), ...
                   sprintf(options.pose_db_file, splits(i).name), ...
                   sprintf(options.attr_db_file, splits(i).name), ...
                   options);
  end
end

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

function createDatasets(truths, labels, image_db_file, label_db_file, ...
                        pose_db_file, attr_db_file, options)
%CREATEDATASETS Create LMDB of fashionista data
  if exist(image_db_file, 'dir') && exist(label_db_file, 'dir') && ...
      exist(pose_db_file, 'dir') && exist(attr_db_file, 'dir')
    return;
  end
  fprintf('Creating %s, %s, %s, %s\n', image_db_file, label_db_file, ...
          pose_db_file, attr_db_file);
  image_db = lmdb.DB(image_db_file, 'MAPSIZE', 1024^3);
  label_db = lmdb.DB(label_db_file, 'MAPSIZE', 1024^3);
  pose_db = lmdb.DB(pose_db_file, 'MAPSIZE', 1024^4);
  attr_db = lmdb.DB(attr_db_file, 'MAPSIZE', 1024^3);
  image_size = [600, 400, 3];
  image_transaction = image_db.begin();
  label_transaction = label_db.begin();
  pose_transaction = pose_db.begin();
  attr_transaction = attr_db.begin();
  try
    batches = sliceArray(1:numel(truths), 50);
    for j = 1:numel(batches)
      for i = batches{j}
        id = num2str(truths(i).index);
        % Insert image JPG.
        image = imdecode(truths(i).image, 'jpg');
        if size(image, 3) == 1
          warning('Grayscale image found: %g / %g', i, numel(truths));
          image = repmat(image, [1, 1, 3]);
          assert(all(size(image) == image_size));
          image = imencode(image, 'jpg');
        else
          image = truths(i).image;
        end
        image_transaction.put(id, caffe_proto_('toEncodedDatum', image, 0));
        % Insert mask PNG.
        annotation = imdecode(truths(i).annotation, 'png');
        annotation = annotation(:, :, 1);
        annotation_png = imencode(annotation, 'png');
        label_transaction.put(id, ...
                              caffe_proto_('toEncodedDatum', annotation_png, 0));
        % Insert pose map.
        posemap = makePosemap(truths(i).pose, image_size, ...
                              options.posemap_radius, ...
                              options.posemap_sigma);
        pose_transaction.put(id, caffe_proto_('toDatum', posemap, 0));
        % Insert attribute vector.
        attributes = zeros(size(labels), 'single');
        attributes(unique(annotation(:)) + 1) = 1.0;
        attr_transaction.put(id, caffe_proto_('toDatum', attributes, 0));
      end
      image_transaction.commit();
      label_transaction.commit();
      pose_transaction.commit();
      attr_transaction.commit();
      fprintf('Imported %g of %g\n', batches{j}(end), numel(truths));
      if j ~= numel(batches)
        image_transaction = image_db.begin();
        label_transaction = label_db.begin();
        pose_transaction = pose_db.begin();
        attr_transaction = attr_db.begin();
      end
    end
  catch exception
    disp(exception.getReport);
    image_transaction.abort();
    label_transaction.abort();
    pose_transaction.abort();
    attr_transaction.abort();
  end
end

function result = decodeAnnotation(annotation)
%decodeAnnotation Convert from RGB to index image.
  result = annotation;
  if size(annotation, 3) == 3
    result = uint32(annotation(:, 1));
    result = bitor(decoded_annotation, bitshift(uint32(annotation(:, 2)), 8));
    result = bitor(decoded_annotation, bitshift(uint32(annotation(:, 3)), 16));
  end
  result = double(result);
end

function posemap = makePosemap(pose, image_size, radius, sigma)
%MAKEPOSEMAP Create a heatmap of joint location.
  num_joints = size(pose.point, 1);
  probability = fspecial('gaussian', 2 * radius + 1, sigma);
  probability = probability ./ max(probability(:));
  posemap = zeros([image_size(1:2), num_joints], 'single');
  for i = 1:num_joints
    point = pose.point(i, :);
    width_1 = min(radius, point(1) - 1);
    height_1 = min(radius, point(2) - 1);
    width_2 = min(radius, image_size(2) - point(1));
    height_2 = min(radius, image_size(1) - point(2));
    posemap((point(2) - height_1):(point(2) + height_2), ...
            (point(1) - width_1):(point(1) + width_2), ...
            i) = ...
      probability((radius + 1 - height_1):(radius + 1 + height_2), ...
                  (radius + 1 - width_1):(radius + 1 + width_2));
  end
end

function exportLabelFile(filename, labels)
%exportLabelFile
  fprintf('Exporting %s\n', filename);
  fid = fopen(filename, 'w');
  try
    fprintf(fid, '%s\n', labels{:});
    fclose(fid);
  catch e
    if fid > 0
      fclose(fid);
    end
    rethrow(e);
  end
end

function slices = sliceArray(array, slice_size)
%SLICEARRAY Split array into slices.
  assert(isvector(array));
  n = numel(array);
  mod_n = mod(n, slice_size);
  sizes = [repmat(slice_size, 1, floor(n / slice_size)), ...
           repmat(mod_n, 1, mod_n ~= 0)];
  if isrow(array)
    slices = mat2cell(array, 1, sizes);
  else
    slices = mat2cell(array, sizes);
  end
end
