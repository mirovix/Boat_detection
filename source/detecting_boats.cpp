#include "detecting_boats.h"

int main(int argc, char** argv)
{

	const string pb_model = "C:/data/model2/model.pb";
	const string pbtxt_model = "C:/data/model2/model.pbtxt";
	//const string img_path = "C:/data/train/20130412_153327_37259.jpg";
	const string img_path = "C:/data/venice_dataset/venice_dataset/05.png";
	
	Mat image = imread(img_path);
	Mat imOut = image.clone();
	Net model = readNetFromTensorflow(pb_model, pbtxt_model);

	//Mat result;
	//preprocess_image(image, result, 94, Range(25, 50), Range(25, 50), 10);
	/*Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(20);

	Mat fin;
	cvtColor(image, fin, COLOR_BGR2GRAY);
	clahe->apply(fin, fin);
	imshow("Output", fin);
	waitKey(0);
	cvtColor(fin, image, COLOR_GRAY2BGR);*/
	Mat fin;
	cvtColor(image, fin, COLOR_BGR2GRAY);
	equalizeHist(fin, fin);
	imshow("Output", fin);
	waitKey(0);
	cvtColor(fin, image, COLOR_GRAY2BGR);
	GaussianBlur(image, image, Size( 3,3 ), 11, 11, BORDER_DEFAULT);

	Mat fres;
	Mat bin;
	wt(image, fres, bin);

	cvtColor(bin, bin, COLOR_GRAY2BGR);

	GaussianBlur(bin, bin, Size(25,25), 33, 33, BORDER_DEFAULT);
	imshow("Output", bin);
	waitKey(0);

	//setUseOptimized(true);
	//setNumThreads(8);

	Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
	ss->setBaseImage(bin);
	//ss->switchToSelectiveSearchFast();
	ss->switchToSingleStrategy();

	vector<Rect> rects;
	ss->process(rects);

	cout << "Total Number of Region Proposals: " << rects.size() << endl;
	//cout << "Total Number of Region Proposals with no mod: " << rects2.size() << endl;

	vector<Rect> bounday_box;
	vector<float> scores;
	

	for (int i = 0; i < rects.size(); i++) {
			if (i % 100 == 0)
				cout << i << endl;
			Mat temp = imOut(rects[i]);
			resize(temp, temp, Size(224, 224), INTER_AREA);
			Mat blob = blobFromImage(temp, 1.0, Size(224, 224));
			model.setInput(blob);
			rectangle(imOut, rects[i], Scalar(0, 0, 255));
			Mat output = model.forward();
			if (output.at<float>(0, 0) > 0.8) {
				bounday_box.push_back(rects[i]);
				scores.push_back(output.at<float>(0, 0));
				//rectangle(imOut, rects[i], Scalar(255, 100, 0));
			}
	}
	vector<int> indices;
	NMSBoxes(bounday_box, scores, 0.3f, 1.0f, indices);
	for (size_t i = 0; i < indices.size(); i++) {
		int idx = indices[i];
		Rect box = bounday_box[idx];
		rectangle(imOut, box, Scalar(0, 255, 0));
	}

	imshow("Output", imOut);
	waitKey(0);

}

double intersection_over_union(vector<int> boxA, vector<int> boxB) {

	int xA = max(boxA[0], boxB[0]);
	int xB = max(boxA[1], boxB[1]);
	int yA = min(boxA[2], boxB[2]);
	int yB = min(boxA[3], boxB[3]);

	//compute the area of intersection rectangle
	int interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);
	int boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1);
	int boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1);

	return double(interArea / double(boxAArea + boxBArea - interArea));
}

void preprocess_image(Mat input, Mat &result, double sigma, Range hue_range, Range value_range, int delta_brightness)
{
	// split into RGB channels
	vector<Mat> img_channels;
	split(input, img_channels);

	// equalize histograms
	equalizeHist(img_channels[0], img_channels[0]);
	equalizeHist(img_channels[1], img_channels[1]);
	equalizeHist(img_channels[2], img_channels[2]);

	// merge back into single equalized RGB image
	Mat equalized_img;
	merge(img_channels, equalized_img);

	result = equalized_img;
}

void wt(Mat src, Mat &fres, Mat &bin) {


	// Show the source image
	imshow("Source Image", src);
	waitKey(0);
	// Change the background from white to black, since that will help later to extract
	// better results during the use of Distance Transform
	Mat mask;
	inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
	src.setTo(Scalar(0, 0, 0), mask);
	// Show output image
	imshow("Black Background Image", src);
	waitKey(0);
	// Create a kernel that we will use to sharpen our image
	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1); 

	Mat imgLaplacian;
	filter2D(src, imgLaplacian, CV_32F, kernel);
	Mat sharp;
	src.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	// imshow( "Laplace Filtered Image", imgLaplacian );
	imshow("New Sharped Image", imgResult);
	waitKey(0);
	// Create binary image from source image
	Mat bw;
	cvtColor(imgResult, bw, COLOR_BGR2GRAY);
	threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("Binary Image", bw);
	waitKey(0);
	//bin = bw;
	// Perform the distance transform algorithm
	Mat dist;
	distanceTransform(bw, dist, DIST_L2, 3);
	// Normalize the distance image for range = {0.0, 1.0}
	// so we can visualize and threshold it
	normalize(dist, dist, 0, 5.0, NORM_MINMAX);
	imshow("Distance Transform Image", dist);
	waitKey(0);
	bin = dist;
	// Threshold to obtain the peaks
	// This will be the markers for the foreground objects
	threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
	// Dilate a bit the dist image
	Mat kernel1 = Mat::ones(3, 3, CV_8U);
	dilate(dist, dist, kernel1);
	imshow("Peaks", dist);
	waitKey(0);
	bin = dist;
	return;
	// Create the CV_8U version of the distance image
	// It is needed for findContours()
	Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);
	// Find total markers
	vector<vector<Point> > contours;
	findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	Mat markers = Mat::zeros(dist.size(), CV_32S);
	// Draw the foreground markers
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
	}
	// Draw the background marker
	circle(markers, Point(5, 5), 3, Scalar(255), -1);
	Mat markers8u;
	markers.convertTo(markers8u, CV_8U, 10);
	imshow("Markers", markers8u);
	// Perform the watershed algorithm
	watershed(src, markers);
	Mat mark;
	markers.convertTo(mark, CV_8U);
	bitwise_not(mark, mark);
	//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
	// image looks like at that point
	// Generate random colors
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++)
	{
		int b = theRNG().uniform(0, 256);
		int g = theRNG().uniform(0, 256);
		int r = theRNG().uniform(0, 256);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	// Create the result image
	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	// Fill labeled objects with random colors
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size()))
			{
				dst.at<Vec3b>(i, j) = colors[index - 1];
			}
		}
	}
	// Visualize the final image
	imshow("Final Result", dst);
	waitKey(0);
	fres = dst;
}