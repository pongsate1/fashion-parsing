// Caffe proto converter.
//
// Build.
//
// mex -I/path/to/matlab-lmdb/include/mexplus ...
//     -I/path/to/caffe/build/src/caffe/proto/ ...
//     caffe_proto_.cc ...
//     /path/to/caffe/build/src/caffe/proto/caffe.pb.o ...
//     -lprotobuf
//
// Usage.
//
// fid = fopen('cat.jpg', 'r');
// jpg_image = fread(fid, inf, 'uint8=>uint8');
// fclose(fid);
// datum = caffe_proto_('toEncodedDatum', jpg_image, label);
// [jpg_image, label] = caffe_proto_('fromDatum', datum);
//
// image = imread('cat.jpg');
// label = 1;
// datum = caffe_proto_('toDatum', image, label);
// [image, label] = caffe_proto_('fromDatum', datum);
//
// Importing into LMDB database.
//
// addpath('/path/to/matlab-lmdb');
// image_files = {
//   '/path/to/image-1.jpg', ...
//   '/path/to/image-2.jpg', ...
//   ...
// };
// database = lmdb.DB('/path/to/lmdb');
// for i = 1:numel(image_files)
//     label = 0;
//     fid = fopen(image_files{i}, 'r');
//     jpg_image = fread(fid, inf, 'uint8=>uint8');
//     fclose(fid);
//     datum = caffe_proto_('toEncodedDatum', jpg_image, label);
//     database.put(image_files{i}, datum);
// end
// clear database;
//

#include "caffe.pb.h"
#include "mexplus.h"

using namespace std;
using namespace mexplus;

#define ASSERT(condition, ...) \
    if (!(condition)) mexErrMsgIdAndTxt("caffe_proto:error", __VA_ARGS__)

MEX_DEFINE(toEncodedDatum) (int nlhs, mxArray* plhs[],
                            int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  caffe::Datum datum;
  MxArray array(input.get(0));
  datum.set_data(array.getData<uint8_t>(), array.size());
  datum.set_label(input.get<int>(1));
  datum.set_encoded(true);
  output.set(0, datum.SerializeAsString());
}

MEX_DEFINE(toDatum) (int nlhs, mxArray* plhs[],
                     int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  caffe::Datum datum;
  MxArray array(input.get(0));
  datum.set_label(input.get<int>(1));
  vector<mwSize> dimensions = array.dimensions();
  int width = dimensions[1];
  int height = dimensions[0];
  int channels = 1;
  for (int i = 2; i < dimensions.size(); ++i)
    channels *= dimensions[i];
  datum.set_channels(channels);
  datum.set_width(width);
  datum.set_height(height);
  vector<mwIndex> subscripts(3);
  if (array.isUint8()) {
    datum.mutable_data()->reserve(array.size());
    for (int k = channels - 1; k >= 0; --k) {  // RGB to BGR order.
      subscripts[2] = k;
      for (int i = 0; i < height; ++i) {
        subscripts[0] = i;
        for (int j = 0; j < width; ++j) {
          subscripts[1] = j;
          datum.mutable_data()->push_back(array.at<uint8_t>(subscripts));
        }
      }
    }
  }
  else {
    datum.mutable_float_data()->Reserve(array.size());
    for (int k = 0; k < channels; ++k) {  // No reverse order.
      subscripts[2] = k;
      for (int i = 0; i < height; ++i) {
        subscripts[0] = i;
        for (int j = 0; j < width; ++j) {
          subscripts[1] = j;
          datum.add_float_data(array.at<float>(subscripts));
        }
      }
    }
  }
  output.set(0, datum.SerializeAsString());
}

MEX_DEFINE(fromDatum) (int nlhs, mxArray* plhs[],
                       int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 2);
  caffe::Datum datum;
  ASSERT(datum.ParseFromString(input.get<string>(0)),
         "Failed to parse datum.");
  if (datum.has_encoded() && datum.encoded()) {
    output.set(0, datum.data());
  }
  else {
    vector<mwIndex> dimensions(3);
    dimensions[0] = (datum.has_height()) ? datum.height() : 0;
    dimensions[1] = (datum.has_width()) ? datum.width() : 0;
    dimensions[2] = (datum.has_channels()) ? datum.channels() : 0;
    MxArray array;
    vector<mwIndex> subscripts(3);
    int index = 0;
    if (datum.has_data()) {
      array.reset(mxCreateNumericArray(dimensions.size(),
                                       &dimensions[0],
                                       mxUINT8_CLASS,
                                       mxREAL));
      const string& data = datum.data();
      for (int k = dimensions[2] - 1; k >= 0; --k) {  // BGR to RGB order.
        subscripts[2] = k;
        for (int i = 0; i < dimensions[0]; ++i) {
          subscripts[0] = i;
          for (int j = 0; j < dimensions[1]; ++j) {
            subscripts[1] = j;
            array.set(subscripts, data[index++]);
          }
        }
      }
    }
    else if (datum.float_data_size() > 0) {
      array.reset(mxCreateNumericArray(dimensions.size(),
                                       &dimensions[0],
                                       mxSINGLE_CLASS,
                                       mxREAL));
      for (int k =  0; k < dimensions[2]; ++k) {  // No reverse order.
        subscripts[2] = k;
        for (int i = 0; i < dimensions[0]; ++i) {
          subscripts[0] = i;
          for (int j = 0; j < dimensions[1]; ++j) {
            subscripts[1] = j;
            array.set(subscripts, datum.float_data(index++));
          }
        }
      }
    }
    output.set(0, array.release());
  }
  output.set(1, (datum.has_label()) ? datum.label() : 0);
}

MEX_DISPATCH
