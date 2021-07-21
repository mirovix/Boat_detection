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

//verify if the input arguments are given in the right way
void checkInput(int argc, char** argv, vector<String> &paths, string &testIoU);

//detecting phase
void detect(Net model, string path, string testIoU, double threshold, int size_processed_image);

//parse the input file IOU given in input
void parsingInputIOU(string testIoU, string name_image, vector<Rect> &bounding_box_input);

//function used for print the errors
void printError(string error);

//computing the IOU given two boundary box
double iou(Rect boxA, Rect boxB);

//compute the non-maxima suppresion and drawing the values
void NMSandDrawing(Mat &output_image, vector<int> &indices, vector<Rect> bounding_box, vector<float> scores);

//given the regions found predict them
void predictRegions(Mat &image, Net model, vector<Rect> &bounding_box, vector<float> &scores, vector<Rect> rects, int size_processed_image, double threshold);

//compute the selective search approch
void selectiveSearch(Mat image, vector<Rect> &rects, char method);

//check the IoU and then display the value
void checkDisplayIoU(Mat &output_image, vector<Rect> bounday_box, vector<int> indices, vector<Rect> bounding_box_input);

//sequence of functions for upgrade the input image
void preprocessig(Mat input_image, Mat &processed_image, int size_processed_image);

//single prediction of a region
void regionPrediction(Net model, Mat &image, Rect rect, double threshold, vector<Rect> &bounding_box, vector<float> &scores);
