function convert_fashionista_to_hdf5(varargin)
%CONVERT_FASHIONISTA_TO_HDF5
%
% mex -Imatlab/matlab-lmdb/include -Ibuild/src/caffe/proto/ \
%     matlab/caffe_proto_.cc -outdir matlab/ \
%     build/src/caffe/proto/caffe.pb.o -lprotobuf
%
  root = fileparts(mfilename('fullpath'));
  options.input = fullfile(root, 'fashionista-v1.0.mat');
  options.output = fullfile(root, '%s-%g.h5');
  options.output_list = fullfile(root, '%s-%g.h5.txt');
  options.posemap_radius = 5;
  options.posemap_sigma = 1.5;
  options.image_mean = [104.00699, 116.66877, 122.67892];
  options.scale = 0.5;
  options = getOptions(options, varargin{:});

  load(options.input, 'truths', 'test_index', 'labels');

  % % Debug posemap here...
  % posemap = makePosemap(truths(1).pose, [600, 400], 7, 4);
  % posemap = reshape(posemap, [600, 400, 1, 14]);
  % montage(im2uint8(posemap), jet(255));
  % keyboard;

  % Make data splits.
  splits = struct('name', {'train', 'val', 'test'}, ...
                  'index', {[], [], test_index});
  [splits(1).index, splits(2).index] = generateIndex(truths, test_index);

  exportLabelFile('labels.txt', labels);
  for i = 1:numel(splits)
    output_file = sprintf(options.output, splits(i).name, options.scale);
    output_list_file = sprintf(options.output_list, ...
                               splits(i).name, ...
                               options.scale);
    createDatasets(output_file, truths(splits(i).index), labels, options);
    writeTextToFile(output_file, output_list_file);
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

function createDatasets(output, truths, labels, options)
%CREATEDATASETS Create HDF5 format.
  if exist(output, 'file')
    fprintf('Replacing the existing file: %s\n', output);
    delete(output);
  else
    fprintf('Creating %s\n', output);
  end

  % Get size information.
  image = imdecode(truths(1).image, 'jpg');
  image = imresize(image, options.scale, 'bilinear');
  image_size = size(image);
  blob_size = image_size([2, 1, 3]);
  num_joints = size(truths(1).pose.point, 1);
  fprintf('Target image size = [%s]\n', sprintf(' %g', image_size));

  % Create datasets.
  h5create(output, ...
           '/id', ...
           [1, numel(truths)], ...
           'Datatype', 'int32')
  h5create(output, ...
           '/image', ...
           [blob_size, numel(truths)], ...
           'Datatype', 'single', ...
           'ChunkSize', [blob_size, 1], ...
           'Deflate', 9);
  h5create(output, ...
           '/segmentation', ...
           [blob_size(1:2), 1, numel(truths)], ...
           'Datatype', 'single', ...
           'ChunkSize', [blob_size(1:2), 1, 1], ...
           'Deflate', 9);
  h5create(output, ...
           '/posemap', ...
           [blob_size(1:2), num_joints, numel(truths)], ...
           'Datatype', 'single', ...
           'ChunkSize', [blob_size(1:2), num_joints, 1], ...
           'Deflate', 9);
  h5create(output, ...
           '/attributes', ...
           [numel(labels), 1, 1, numel(truths)], ...
           'Datatype', 'single');
  h5disp(output);

  % Write datasets.
  for i = 1:numel(truths)
    id = int32(truths(i).index);
    h5write(output, '/id', id, [1, i], [1, 1]);
    % Preprocess images.
    image = imdecode(truths(i).image, 'jpg');
    assert(size(image, 3) == 3);
    image = imresize(image, options.scale, 'bilinear');
    % Permute RGB to BGR and subtract mean.
    image = bsxfun(@minus, ...
                   single(image(:, :, [3, 2, 1])), ...
                   reshape(options.image_mean, [1, 1, 3]));
    image = permute(image, [2, 1, 3]);
    h5write(output, '/image', image, [1, 1, 1, i], [size(image), 1]);
    % Insert mask PNG.
    annotation = imdecode(truths(i).annotation, 'png');
    annotation = imresize(annotation, options.scale, 'nearest');
    annotation = permute(annotation(:, :, 1), [2, 1]);
    h5write(output, '/segmentation', single(annotation), ...
            [1, 1, 1, i], [size(annotation), 1, 1]);
    % Insert pose map.
    posemap = makePosemap(truths(i).pose.point * options.scale, ...
                          image_size, ...
                          options.posemap_radius, ...
                          options.posemap_sigma);
    posemap = permute(posemap, [2, 1, 3]);
    h5write(output, '/posemap', posemap, ...
            [1, 1, 1, i], [size(posemap), 1]);
    % Insert attribute vector.
    attributes = zeros([numel(labels), 1], 'single');
    attributes(unique(annotation(:)) + 1) = 1.0;
    h5write(output, '/attributes', attributes, ...
            [1, 1, 1, i], [size(attributes), 1, 1]);
    if mod(i, 100) == 0
      fprintf('Imported %g of %g\n', i, numel(truths));
    end
  end
  fprintf('Imported %g of %g\n', i, numel(truths));
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

function posemap = makePosemap(pose_point, image_size, radius, sigma)
%MAKEPOSEMAP Create a heatmap of joint location.
  num_joints = size(pose_point, 1);
  probability = fspecial('gaussian', 2 * radius + 1, sigma);
  probability = probability ./ max(probability(:));
  posemap = zeros([image_size(1:2), num_joints], 'single');
  for i = 1:num_joints
    point = round(pose_point(i, :));
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
