#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/core.hpp>
#include "opencv2/ximgproc/segmentation.hpp"


#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;
using namespace dnn;
using namespace cv::ximgproc::segmentation;

//Computing the IOU given two boundary box
double intersection_over_union(vector<int> boxA, vector<int> boxB);