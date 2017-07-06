function startup
  root_dir = fileparts(mfilename('fullpath'));
  addpath(fullfile(root_dir, 'matlab'));
  addpath(fullfile(root_dir, 'matlab/matlab-lmdb'));
  addpath(fullfile(root_dir, 'matlab/matlab-leveldb'));
  addpath(fullfile(root_dir, 'matlab/encoder'));
  addpath(fullfile(root_dir, 'matlab/util'));

  % mex -Imatlab/matlab-lmdb/include -Ibuild/src/caffe/proto/ ...
  %    matlab/caffe_proto_.cc -outdir matlab/ ...
  %    build/src/caffe/proto/caffe.pb.o -lprotobuf ...
  %    CXXFLAGS="$CXXFLAGS -std=c++11"
end
