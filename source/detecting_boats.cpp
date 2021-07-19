#include "detecting_boats.h"

int main(int argc, char** argv)
{
	Timer timer = Timer();
	start(timer);

	//const string testIoU = "C:/data/test_IoU.txt";
	const string testIoU = "";
	const string pb_model = "C:/data/model21/model.pb";
	const string pbtxt_model = "C:/data/model21/model.pbtxt";
	//const string img_path = "C:/data/train/20130412_153327_37259.jpg";   //20130412_153327_37259.jpg";20130412_084036_47574
	const string img_path = "C:/data/venice_dataset/venice_dataset/11.png";
	//const string img_path = "C:/data/Kaggle_ships/06.jpg";
	
	//setUseOptimized(true);
	//setNumThreads(4);

	Mat image = imread(img_path);
	Mat imOut = image.clone();
	Net model = readNetFromTensorflow(pb_model, pbtxt_model);


	//for(Rect rect : bb_in)
	//	rectangle(image, rect, Scalar(0, 255, 0));
	//imshow("Output", image);
	//waitKey(0);

	

	//Mat result;
	//preprocess_image(image, result, 94, Range(25, 50), Range(25, 50), 10);
	/*Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(1);

	Mat fin;
	cvtColor(image, fin, COLOR_BGR2GRAY);
	clahe->apply(fin, fin);
	//imshow("Output", fin);
	//waitKey(0);
	cvtColor(fin, image, COLOR_GRAY2BGR);
	*/
	GaussianBlur(image, image, Size(7,7), 9,9, BORDER_DEFAULT);
	/*
	Mat fin;
	cvtColor(image, fin, COLOR_BGR2GRAY);
	equalizeHist(fin, fin);
	//imshow("Output", fin);
	//waitKey(0);
	cvtColor(fin, image, COLOR_GRAY2BGR);*/


	Mat fres;
	Mat bin;
	wt(image, fres, bin);
	//bin = image;
	cvtColor(bin, bin, COLOR_GRAY2BGR);
	int a = 124; //124
	resize(bin, bin, Size(a, a), INTER_AREA);
	//GaussianBlur(bin, bin, Size(3,3), 5,5, BORDER_DEFAULT);
	//imshow("OutputPreNN", bin);
	//waitKey(0);

	Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
	ss->setBaseImage(bin);
	ss->switchToSelectiveSearchFast();
	//ss->switchToSelectiveSearchQuality();
	//ss->switchToSingleStrategy(10, 0.001);
	//ss->switchToSingleStrategy();

	vector<Rect> rects;
	ss->process(rects);

	cout << "Total Number of Region Proposals: " << rects.size() << endl;
	//cout << "Total Number of Region Proposals with no mod: " << rects2.size() << endl;


	vector<Rect> bounday_box;
	vector<float> scores;
	for (int i = 0; i < rects.size(); i++) {
		if (i % 50 == 0)
			cout << i << endl;

		rects[i].x = (rects[i].x * imOut.size().width) / a;
		rects[i].y = (rects[i].y * imOut.size().height) / a;
		rects[i].width = (rects[i].width * imOut.size().width) / a;
		rects[i].height = (rects[i].height * imOut.size().height) / a;


		//rectangle(imOut, rects[i], Scalar(0, 0, 255));
		predict(model, imOut, rects[i], 0.85, bounday_box, scores);
	}
	vector<int> indices;
	NMSBoxes(bounday_box, scores, 0.3f, 0.1f, indices);
	for (size_t i = 0; i < indices.size(); i++) {
		Rect box = bounday_box[indices[i]];
		rectangle(imOut, box, Scalar(0, 255, 0));
	}

	//y, x, h, w
	//string bbinput = "(593,414,287,75);(243,585,423,124);(869,538,345,122)";
	if (!testIoU.empty()) {
		ifstream in(testIoU);
		vector<Rect> bb_in;
		parsing(testIoU,"06.jpg",bb_in);
		checkDisplayIoU(imOut, bounday_box, indices, bb_in);
	}

	cout << "time required: " << stop(timer) << "s" << endl;
	imshow("Output", imOut);
	waitKey(0);


}

void parsing(string testIoU, string name_image, vector<Rect> &bb_in) {
	
	ifstream in(testIoU);
	vector<string> labeled_images;
	string text_line;
	int flag_found = 0;
	while (in >> text_line)
		labeled_images.push_back(text_line);
	in.close();

	vector <string> tokens;
	string intermediate;

	//remove ";"
	for (string line : labeled_images) {
		tokens.clear();
		stringstream check1(line);
		while (getline(check1, intermediate, ';'))
			tokens.push_back(intermediate);
		if (tokens[0].compare(name_image) == 0) {
			flag_found = 1;
			break;
		}
		flag_found = 0;
	}

	if (!flag_found)
		printError("Image in input not present inside label txt file.");

	//remove paranthesis
	for (int i = 1; i < tokens.size(); i++)
		tokens[i] = tokens[i].substr(1, tokens[i].length() - 2);

	for (int i = 1; i < tokens.size(); i++) {
		vector<double> coords;
		stringstream check(tokens[i]);
		while (getline(check, intermediate, ','))
			coords.push_back(stoi(intermediate));
		bb_in.push_back(Rect(coords[0], coords[1], coords[2], coords[3]));
	}
 

}

void checkDisplayIoU(Mat &imOut, vector<Rect> bounday_box, vector<int> indices, vector<Rect> bb_in) {

	for (int index : indices){

		vector<double> ious;

		//computing the IoU between the input and predicted rects
		for(Rect rect : bb_in)
			ious.push_back(IoU(bounday_box[index], rect));

		//checking condition
		if (ious.size() == 0)
			continue;
		
		//max IoU found in %, e.g. 75%
		int max_iou = roundf(*max_element(ious.begin(), ious.end())*100);

		//index of the max IoU found
		int max_index = max_element(ious.begin(), ious.end()) - ious.begin();

		//draw rectangle
		rectangle(imOut, bb_in[max_index], Scalar(0, 0, 255));

		//draw on the image the IoU value
		string text = "IOU:" + to_string(max_iou)+"%";
		putText(imOut, text, Point2f(bounday_box[index].x, bounday_box[index].y), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255, 255), 1, 3);
	}

}

void predict(Net model, Mat &image, Rect rect, double th, vector<Rect> &bounday_box, vector<float> &scores){

	Mat blob = blobFromImage(image(rect), 1.0, Size(224, 224));
	model.setInput(blob);	
	Mat output = model.forward();
	if (output.at<float>(0, 0) > th) {
		bounday_box.push_back(rect);
		rectangle(image, rect, Scalar(0, 0, 255));
		scores.push_back(output.at<float>(0, 0));
	}

}

double IoU(Rect first_rect, Rect second_rect) {

	if (first_rect.empty() || second_rect.empty())
		printError("IOU function: one of the two rects is empty");

	int x_left = max(first_rect.x, second_rect.x);
	int y_top = max(first_rect.y, second_rect.y);
	int x_right = min((first_rect.x + first_rect.width), (second_rect.x + second_rect.width));
	int y_bottom = min((first_rect.y + first_rect.height), (second_rect.y + second_rect.height));

	//compute the area of intersection rectangle
	if ((x_right < x_left) ||  (y_bottom < y_top))
		return 0.0;

	double intersection_area = (x_right - x_left) * (y_bottom - y_top);
	double first_rect_area = (first_rect.width) * (first_rect.height);
	double second_rect_area = (second_rect.width) * (second_rect.height);
	double iou = intersection_area / double(first_rect_area + second_rect_area - intersection_area);

	if (iou > 1.0 || iou < 0.0)
		printError("IoU value is not between 0 and 1");

	return iou;
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
	//imshow("Source Image", src);
	//waitKey(0);
	// Change the background from white to black, since that will help later to extract
	// better results during the use of Distance Transform
	Mat mask;
	inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
	src.setTo(Scalar(0, 0, 0), mask);
	// Show output image
	//imshow("Black Background Image", src);
	//waitKey(0);
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
	//imshow("New Sharped Image", imgResult);
	//waitKey(0);
	// Create binary image from source image
	//bin = imgResult; return;
	Mat bw;
	cvtColor(imgResult, bw, COLOR_BGR2GRAY);
	//threshold(bw, bw, 30, 255, THRESH_BINARY | THRESH_OTSU);
	adaptiveThreshold(bw, bw, 250, BORDER_REPLICATE, THRESH_BINARY, 111, 50);
	//GaussianBlur(bw, bw, Size(3, 3), 7, 7);
	//imshow("Binary Image", bw);
	//waitKey(0);
	bin = bw;
	//return;
	//return;
	//waitKey(0);
	// Perform the distance transform algorithm
	Mat dist;
	distanceTransform(bw, dist, DIST_L2, 3);
	// Normalize the distance image for range = {0.0, 1.0}
	// so we can visualize and threshold it
	normalize(dist, dist, 0, 30, NORM_MINMAX);
	bin = dist;
	
	//imshow("Distance Transform Image", dist);
	//waitKey(0);
	return;
	//return;
	// Threshold to obtain the peaks
	// This will be the markers for the foreground objects
	threshold(dist, dist, 0.6, 1.0, THRESH_BINARY);
	// Dilate a bit the dist image
	Mat kernel1 = Mat::ones(3, 3, CV_8U);
	dilate(dist, dist, kernel1);
	bin = dist;
	//imshow("Distance Transform Image", dist);
	//waitKey(0);
	return;
	//imshow("Peaks", dist);
	//waitKey(0);
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
	//imshow("Markers", markers8u);
	bin = markers8u;
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

void printError(string error)
{
	printf("\nERROR: %s \n\n", error);
	exit(1);
}

void start(Timer timer) {
	timer.start = ((double)clock() / (double)CLK_TCK);
}

double stop(Timer timer) {
	// Get the time
	timer.stop = ((double)clock() / (double)CLK_TCK);
	double time = timer.stop - timer.start;
	return time;
}