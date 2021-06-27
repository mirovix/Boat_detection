#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

//Computing the IOU given two boundary box
double intersection_over_union(vector<int> boxA, vector<int> boxB);