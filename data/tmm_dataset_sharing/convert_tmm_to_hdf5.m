function convert_tmm_to_hdf5(matfile)
    %matfile = 'fashion_parsing_data.mat'
    % This function convert TMM dataset to a hdf5 file.
    % The target hdf5 file will contain 4 sub-dataset: 'attributes', 'id', 'image', 'segmentation'
    % TMM dataset consists of 2682 images with its id (as filename), and segmentation.
    % This function load needed info and image files into variables, then export them to a hdf5 file,
    % and create a text file containing path of that hdf5 file.

    image_mean = [104.00699, 116.66877, 122.67892]; % from FCN, which trained on PASCAL VOC 2011
    disp (['loading ',matfile]);
    mat = load(matfile)
    
    fashion_dataset = mat.fashion_dataset;
    labels = mat.all_category_name;
    
    % train val and test file
    numFile = numel(fashion_dataset)
    numVal = round(numFile*0.02)
    numTest = round(numFile*0.2)
    numTrain = numFile-numVal-numTest
    
    trainPart = 5;
    nTrain = zeros(5,1);
    nTrain(1:4) = round(numTrain/5);
    nTrain(5) = numTrain-sum(nTrain(1:4));
    prev = 419;
    fileid = fopen('TMM_train.h5.txt','w');
    for i=2:5
        H5_TRAIN_FILE = strcat('TMM_train-',int2str(i),'.h5');
        createH5File(H5_TRAIN_FILE,prev,prev+nTrain(i)-1,mat,image_mean);
        prev = prev+nTrain(i);
        fwrite(fileid,strcat(pwd,'/',H5_TRAIN_FILE));
    end
    fclose(fileid);

    H5_TEST_FILE = 'TMM_test.h5'
    H5_VAL_FILE = 'TMM_val.h5'
    
    
    createH5File(H5_TEST_FILE,numTrain+1,numTrain+numTest,mat,image_mean)
    createH5File(H5_VAL_FILE,numTrain+numTest+1,numTrain+numTest+numVal,mat,image_mean)   

    fileid = fopen('TMM_test.h5.txt','w');
    fwrite(fileid,strcat(pwd,'/',H5_TEST_FILE));
    fclose(fileid);

    fileid = fopen('TMM_val.h5.txt','w');
    fwrite(fileid,strcat(pwd,'/',H5_VAL_FILE));
    fclose(fileid);

    LABEL_FILE = 'labels.txt'
    
    exportLabelFile(LABEL_FILE, labels)
end

function createH5File(filename,startNum,endNum,mat,image_mean)
    
    fashion_dataset = mat.fashion_dataset;
    
    labels = mat.all_category_name;% all possible attributes name
    
    image = imread(strcat('image/',fashion_dataset{1}.img_name));
    image_size = size(image);
    blob_size = image_size([2,1,3]);

    % create datasets
    % h5create(filename,datasetname,data)
    numData = endNum-startNum+1
    h5create(filename,'/id',[1,numData],'Datatype','int32')
    h5create(filename,'/image', ...
        [blob_size, numData], ...
        'Datatype', 'single', ...
        'ChunkSize', [blob_size, 1], ...
        'Deflate', 9);
    h5create(filename, ...
        '/segmentation', ...
        [blob_size(1:2), 1, numData], ...
        'Datatype', 'single', ...
        'ChunkSize', [blob_size(1:2), 1, 1], ...
        'Deflate', 9);
    % 1 dimension binary vector. Each cell i is 1 if attributes[i] is presented, 0 otherwise
    h5create(filename, ...
        '/attributes', ...
        [numel(labels), 1, 1, numData], ...
        'Datatype', 'single');
    h5disp(filename);

    % write datasets
    % for each image, 
    for i = startNum:endNum
        status = ['start: ',int2str(startNum),'current: ',int2str(i),' end: ',int2str(endNum)];
        disp(status)
        % segmentation = segmentation map (label map)
        segmentation = fashion_dataset{i}.category_label(double(fashion_dataset{i}.segmentation+1));
        segmentation = segmentation-1;
        % attributes number
        attributes = unique(segmentation)+1;
        attr_vec = zeros(numel(labels),1);
        attr_vec(attributes) = 1;
        % image file name
        imname = fashion_dataset{i}.img_name;
        % image ID extract from the file name.
        id = sscanf(fashion_dataset{i}.img_name,'%d');
        image = imread(strcat('image/',fashion_dataset{i}.img_name));
        
        h5write(filename,'/id',id,[1,i-startNum+1],[1,1]);
        
        % matlab load image in 600row 400col 3depth
        % convert to 400x600x3
        image = bsxfun(@minus, ...
                   single(image(:, :, [3, 2, 1])), ...
                   reshape(image_mean, [1, 1, 3]));
        image = permute(image, [2, 1, 3]);
        % change from RGB to BGR and subtract mean
        h5write(filename, '/image', image, [1, 1, 1, i-startNum+1], [size(image), 1]);
        
        segmentation = permute(segmentation(:, :, 1), [2, 1]);
        h5write(filename,'/segmentation',single(segmentation), ...
            [1, 1, 1, i-startNum+1], [size(segmentation), 1, 1]);
       
       h5write(filename, '/attributes', attr_vec, ...
            [1, 1, 1, i-startNum+1], [size(labels), 1, 1]);
       
        status = [imname,' goes to ',filename];
        disp(status)
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
    