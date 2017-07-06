#include "color_map.h"
#include "Metric.h"
#include "MaskRefine.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

struct stat info;
using namespace std;

void loadImgList(string path, string list_name, vector<string>& image_names);

bool checkDir(const char* pathname){
  if( stat( pathname, &info ) != 0 ){
      // printf( "cannot access %s\n", pathname );
      return false;
  }
  else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows 
  {
      // printf( "%s is a directory\n", pathname );
      return true;
  } 
  else
  {
      // printf( "%s is no directory\n", pathname );
      return false;
  }
}

int main (int argc, char **argv) {
  
  string model_name = argv[1];
  string DATASET = argv[2];
  
  // const string model_name = "fcn-8s";
  // const string model_name = "sege-8s";
  // const string model_name = "segf-8s";
  // const string model_name = "attrlog";

  


  // const string DATASET = "fashionista-v1.0";
  // const string DATASET = "tmm_dataset_sharing";
  // const string DATASET = "fashionista-v0.2";

  string model_suffix;
  int label_num;
  if (DATASET == "fashionista-v1.0")
  {
    label_num = 25;
    model_suffix = DATASET;
  }
  else if (DATASET == "tmm_dataset_sharing")
  {
    label_num = 23;
    model_suffix = "tmm";
  }
  else if (DATASET == "fashionista-v0.2")
  {
    label_num = 56;
    model_suffix = DATASET;
  }

  string img_path = "data/"+DATASET+"/testimages/";
  string gt_path = "data/"+DATASET+"/testimages/";

  // output from FCN, used as input to this CRF
  string mask_path = "models/"+model_name+"-"+model_suffix+"/mask/";

  //========================================================================================
  // output folder of this CRF.
  string out_folder = "models/"+model_name+"-"+model_suffix+"/refine/";
  if (!checkDir(out_folder.c_str()))
    mkdir(out_folder.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  // May be intermediate output
  string out_bgr_folder = "models/"+model_name+"-"+model_suffix+"/bgr/";
  if (!checkDir(out_bgr_folder.c_str()))
    mkdir(out_bgr_folder.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  //========================================================================================

  vector<string> img_names;
  vector<string> gt_names;
  vector<string> mask_names;

  // see if the filename in img_list.txt really exist, then get the path to existing files in the corresponding variable
  // loadImgList(img_path, "img_list.txt", img_names);
  loadImgList(img_path, "img_list.txt", img_names);
  loadImgList(gt_path, "gt_list.txt", gt_names);
  loadImgList(mask_path, "mask_list.txt", mask_names);

  CRF_TAG param;
  param.fg_prob_ = 0.5f;
  param.iter_num = 10;
  param.sigma_pos_ = 3;
  param.weight_pos_ = 10;
  param.sigma_color_bi_ = 10;
  param.sigma_pos_bi_ = 30;
  param.weight_bi_ = 10;
  param.sigma_color_ = 30;
  param.weight_color_ = 2;


  float avg_ori_acc = 0;
  float avg_new_acc = 0;
  float avg_ori_fg_acc = 0;
  float avg_new_fg_acc = 0;

  for (int i = 0; i < img_names.size(); ++i) {
    cout << i << endl;

    cv::Mat img = cv::imread(img_path + img_names[i], 1);
    cv::Mat gt = cv::imread(gt_path + gt_names[i], 0);
    cv::Mat mask = cv::imread(mask_path + mask_names[i], 0);

    int width = img.cols;
    int height = img.rows;

    mask.convertTo(mask, CV_32SC1);
    gt.convertTo(gt, CV_32SC1);
    cv::Mat mask_refine(mask.size(), CV_32SC1);

    MaskRefine((int*)mask.data, img.data, width, height, label_num, &param, (int*)mask_refine.data);

    cv::Mat gt_bgr(img.size(), CV_8UC3);
    cv::Mat mask_bgr(img.size(), CV_8UC3);
    cv::Mat mask_refine_bgr(img.size(), CV_8UC3);
    Visualize((int*)mask.data, width, height, mask_bgr.data);
    Visualize((int*)mask_refine.data, width, height, mask_refine_bgr.data);
    Visualize((int*)gt.data, width, height, gt_bgr.data);

    float ori_acc = AllPixelAcc((int*)mask.data, (int*)gt.data, width, height);
    float new_acc = AllPixelAcc((int*)mask_refine.data, (int*)gt.data, width, height);
    float ori_fg_acc = FgdPixelAcc((int*)mask.data, (int*)gt.data, width, height);
    float new_fg_acc = FgdPixelAcc((int*)mask_refine.data, (int*)gt.data, width, height);

    avg_ori_acc += ori_acc;
    avg_new_acc += new_acc;
    avg_ori_fg_acc += ori_fg_acc;
    avg_new_fg_acc += new_fg_acc;

    cout << "original Acc. : " << ori_acc << endl;
    cout << "Refined  Acc. : " << new_acc << endl;
    cout << "original Fg. Acc. : " << ori_fg_acc << endl;
    cout << "Refined  Fg. Acc. : " << new_fg_acc << endl;
    cout << "**********************************" << endl << endl;

    // cv::imshow("image", img);
    // cv::imshow("gt", gt_bgr);
    // cv::imshow("before", mask_bgr);
    // cv::imshow("after", mask_refine_bgr);
    // cv::waitKey();

    // change img_names to gt_names because gt_names are jpg but gt_names are png and jpg is low quality and produce noise
    string save_path = out_folder + gt_names[i];
    cv::imwrite(save_path, mask_refine);
    string save_bgr_path = out_bgr_folder + gt_names[i];
    cv::imwrite(save_bgr_path, mask_refine_bgr);

  }

  avg_ori_acc /= img_names.size();
  avg_new_acc /= img_names.size();
  avg_ori_fg_acc /= img_names.size();
  avg_new_fg_acc /= img_names.size();

  cout << "original Acc. : " << avg_ori_acc << endl;
  cout << "Refined  Acc. : " << avg_new_acc << endl;
  cout << "original Fg. Acc. : " << avg_ori_fg_acc << endl;
  cout << "Refined  Fg. Acc. : " << avg_new_fg_acc << endl;

  system("pause");
  return 0;
}


//***************************************************************************************************************/

void loadImgList(string path, string list_name, vector<string>& image_names) {
  ifstream list_file((path + list_name).c_str(), ios::in);
  string name;
  while(getline(list_file, name)) {
    image_names.push_back(name);
  }
  list_file.close();
}