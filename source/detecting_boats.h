#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <thread>
#include <assert.h> 

#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;
using namespace dnn;
using namespace cv::ximgproc::segmentation;

typedef struct {
	double start;
	double stop;
} Timer;

// Start the timer
void start(Timer timer);

// Stop the timer, free the struct and return the time spent
double stop(Timer timer);

void parsing(string testIoU, string name_image, vector<Rect> &bb_in);

void printError(string error);

//Computing the IOU given two boundary box
double IoU(Rect boxA, Rect boxB);

void checkDisplayIoU(Mat &imOut, vector<Rect> bounday_box, vector<int> indices, vector<Rect> bb_in);

void wt(Mat src, Mat &fres, Mat &bin);

void preprocess_image(Mat input, Mat &result, double sigma, Range hue_range, Range value_range, int delta_brightness);

//void NN(double th, int a, Mat imOut, Net model, vector<Rect> rects, vector<Rect> &bounday_box, vector<float> &scores);

void predict(Net model, Mat &image, Rect rect, double th, vector<Rect> &bounday_box, vector<float> &scores);
