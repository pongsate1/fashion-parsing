function slices = sliceArray(array, slice_size)
%SLICEARRAY Split array into slices.
%
%  slices = sliceArray(array, slice_size)
%
% The function takes an array and split into a cell array of sliced elements.
%
% Example:
%
% >> sliceArray(1:10, 3)
%
% ans =
%
%     [1x3 double]    [1x3 double]    [1x3 double]    [10]
%
% See also num2cell mat2cell
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
