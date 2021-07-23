#include "detecting_boats.h"

const string pb_model = "C:/data/model50/model.pb";
const string pbtxt_model = "C:/data/model50/model.pbtxt";

int main(int argc, char** argv)
{
	//const string img_path = "C:/data/train/20130412_153327_37259.jpg";   //20130412_153327_37259.jpg";20130412_084036_47574
	//const string directory = "C:/data/venice_dataset/venice_dataset/*";
	//const string directory = "C:/data/Kaggle_ships/*";
	//string directory = "C:/data/venice_dataset/venice_dataset/* C:/data/test_IoU.txt";

	//load the neural network
	Net model = readNetFromTensorflow(pb_model, pbtxt_model);
	 
	vector<String> paths;
	string testIoU;
	checkInput(argc, argv, paths, testIoU);

	//size of the image before the preprocessing and the threshold for selecting boat
	int size_processed_image = 150; //124
	double threshold = 0.70;
	//detecting images 
	for (String path : paths)
	{
		cout << "Image loaded: " << path << endl;
		detect(model, path, testIoU, threshold, size_processed_image);
	}

}

void detect(Net model, string path, string testIoU, double threshold, int size_processed_image) {

	Timer timer = Timer();
	timer.start = ((double)clock() / (double)CLK_TCK);

	//load the input image
	Mat input_image = imread(path);
	Mat output_image = input_image.clone();

	//preprocessing phase
	Mat processed_image;
	preprocessig(input_image, processed_image, size_processed_image);


	//finding the regions 
	vector<Rect> rects;
	selectiveSearch(processed_image, rects, 'F');

	
	//predict the regions
	vector<Rect> bounding_box;
	vector<float> scores;
	predictRegions(output_image, model, bounding_box, scores, rects, size_processed_image, threshold);

	//Non maxima suppression and drawing rects
	vector<int> indices;
	NMSandDrawing(output_image, indices, bounding_box, scores);

	//y, x, h, w
	//Computing the IoU given the file of bounding boxes
	if (!testIoU.empty()) {
		vector<Rect> bounding_box_input;
		parsingInputIOU(testIoU, path, bounding_box_input);
		checkDisplayIoU(output_image, bounding_box, indices, bounding_box_input);
	}

	//display time 
	timer.stop = ((double)clock() / (double)CLK_TCK);
	double time = timer.stop - timer.start;
	cout << "time required: " << time << "s" << endl;

	//display out
	imshow("Output", output_image);
	waitKey(0);
}

void checkInput(int argc, char** argv, vector<String> &paths, string &testIoU) {

	//usage -> C:\file\venice_dataset\* C:\file\IoU.txt"
	if (argc < 2)
		printError("checkInput: missing or input file wrong, usage -> directory or image and IoU file (not mandatory)");

	//save the directory
	string directory = argv[1];
	glob(directory, paths);
	if(paths.size()==0)
		printError("chekInput: empty directory");

	//save testIoU if it is given
	if (argc == 3)
		testIoU = argv[2];
	else
		testIoU = "";

}

void NMSandDrawing(Mat &output_image, vector<int> &indices, vector<Rect> bounding_box, vector<float> scores) {

	NMSBoxes(bounding_box, scores, 0.3f, 0.03f, indices);
	for (size_t i = 0; i < indices.size(); i++) {
		Rect box = bounding_box[indices[i]];
		rectangle(output_image, box, Scalar(0, 255, 0));
	}
}

void predictRegions(Mat &image, Net model, vector<Rect> &bounding_box, vector<float> &scores, vector<Rect> rects, int size_processed_image, double threshold) {

	cout << "Starting processing each region..." << endl;

	for (Rect rect : rects) {
	
		//resize the regions rects
		rect.x = (rect.x * image.size().width) / size_processed_image;
		rect.y = (rect.y * image.size().height) / size_processed_image;
		rect.width = (rect.width * image.size().width) / size_processed_image;
		rect.height = (rect.height * image.size().height) / size_processed_image;

		//predict the region
		regionPrediction(model, image, rect, threshold, bounding_box, scores);
	}

	cout << "Prediction phase completed" << endl;
}

void selectiveSearch(Mat image, vector<Rect> &rects, char method) {

	//definition of selecetive search
	Ptr<SelectiveSearchSegmentation> selective_search = createSelectiveSearchSegmentation();
	selective_search->setBaseImage(image);

	//selecte the method for finding the regions (defualt fast implmentation)
	switch (method)
	{
		case 'F':
			selective_search->switchToSelectiveSearchFast();
			break; 
		
		case 'Q':
			selective_search->switchToSelectiveSearchQuality();
			break; 

		case 'N':
			selective_search->switchToSingleStrategy();
			break;

		case 'SF':
			selective_search->switchToSingleStrategy(10, 0.001);
			break;

		default: 
			selective_search->switchToSelectiveSearchFast();
			break;
	}

	//process the image and return the regions
	selective_search->process(rects);

	cout << "Total number of region proposals found: " << rects.size() << endl;
}

void parsingInputIOU(string testIoU, string name_image, vector<Rect> &bounding_box_input) {
	
	//open the file
	ifstream input(testIoU);
	if (!input.is_open())
		printError("parsingInputIOU: unable to open text file");

	vector <string> tokens;
	string intermediate;

	//parsing the name image 
	stringstream check_name_input(name_image);
	while (getline(check_name_input, intermediate, '\\'))
		tokens.push_back(intermediate);
	name_image = tokens[1 ];

	//read the lines
	vector<string> labeled_images;
	string text_line;
	while (input >> text_line)
		labeled_images.push_back(text_line);
	input.close();



	//check if the input image is present inside the txt file and then remove ";" from the input
	int flag_found = 0;
	for (string line : labeled_images) {
		tokens.clear();
		stringstream check_line(line);
		while (getline(check_line, intermediate, ';'))
			tokens.push_back(intermediate);
		if (tokens[0].compare(name_image) == 0) {
			flag_found = 1;
			break;
		}
		flag_found = 0;
	}

	//image not found 
	if (!flag_found)
		printError("parsingInputIOU: Image in input not present inside label txt file");


	//remove paranthesis
	for (int i = 1; i < tokens.size(); i++)
		tokens[i] = tokens[i].substr(1, tokens[i].length() - 2);


	//save each value insaide the vector 
	for (int i = 1; i < tokens.size(); i++) {
		vector<double> coords;
		stringstream check(tokens[i]);
		while (getline(check, intermediate, ','))
			coords.push_back(stoi(intermediate));
		bounding_box_input.push_back(Rect(coords[0], coords[1], coords[2], coords[3]));
	}
 

}

void checkDisplayIoU(Mat &output_image, vector<Rect> bounding_box, vector<int> indices, vector<Rect> bounding_box_input) {

	for (int index : indices){

		vector<double> ious;

		//computing the IoU between the input and predicted rects
		for(Rect rect : bounding_box_input)
			ious.push_back(iou(bounding_box[index], rect));

		
		//max IoU found in %, e.g. 75% and control if it is greater than a given threshold
		int max_iou = roundf(*max_element(ious.begin(), ious.end())*100);
		if ((max_iou < 20) || (ious.size() == 0))
			continue;

		//index of the max IoU found
		int max_index = max_element(ious.begin(), ious.end()) - ious.begin();

		//draw rectangle
		rectangle(output_image, bounding_box_input[max_index], Scalar(0, 0, 255));

		//draw on the image the IoU value
		string text = "IOU:" + to_string(max_iou)+"%";
		putText(output_image, text, Point2f(bounding_box[index].x, bounding_box[index].y), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255, 255), 1, 3);
	}

}

void regionPrediction(Net model, Mat &image, Rect rect, double th, vector<Rect> &bounday_box, vector<float> &scores){

	//set blob using the input image
	Mat blob = blobFromImage(image(rect), 1.0, Size(224, 224));
	model.setInput(blob);	

	Mat output = model.forward();
	if (output.at<float>(0, 0) > th) {
		bounday_box.push_back(rect);
		//rectangle(image, rect, Scalar(0, 0, 255));
		scores.push_back(output.at<float>(0, 0));
	}

}

double iou(Rect first_rect, Rect second_rect) {

	if (first_rect.empty() || second_rect.empty())
		printError("iou: one of the two rects is empty");

	//define the max/min values
	int x_left = max(first_rect.x, second_rect.x);
	int y_top = max(first_rect.y, second_rect.y);
	int x_right = min((first_rect.x + first_rect.width), (second_rect.x + second_rect.width));
	int y_bottom = min((first_rect.y + first_rect.height), (second_rect.y + second_rect.height));

	//check the value
	if ((x_right < x_left) ||  (y_bottom < y_top))
		return 0.0;
	
	//compute the area of intersection rectangle
	double intersection_area = (x_right - x_left) * (y_bottom - y_top);
	double first_rect_area = (first_rect.width) * (first_rect.height);
	double second_rect_area = (second_rect.width) * (second_rect.height);
	double iou = intersection_area / double(first_rect_area + second_rect_area - intersection_area);

	//check the iou value
	if (iou > 1.0 || iou < 0.0)
		printError("iou: IoU value is not between 0 and 1");

	return iou;
}

void preprocessig(Mat input_image, Mat &processed_image, int size_processed_image) {

	//check inputs
	if (size_processed_image < 30 || size_processed_image > 600)
		printError("Preprocessing: Input 'size_processed_image' too small or too high");

	//smooth image using the Gaussian filter
	GaussianBlur(input_image, input_image, Size(7, 7), 11, 11, BORDER_DEFAULT);


	//change the background from white to black since performs better during the use of Distance Transform
	Mat mask;
	inRange(input_image, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
	input_image.setTo(Scalar(0, 0, 0), mask);


	//create a kernel for sharpe input image
	Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1); 

	
	//compute the sharpe image
	Mat laplacina_image, sharped_image;
	filter2D(input_image, laplacina_image, CV_32F, kernel);

	input_image.convertTo(sharped_image, CV_32F);
	Mat result_image = sharped_image - laplacina_image;


	// convert back to 8bits gray scale
	result_image.convertTo(result_image, CV_8UC3);
	laplacina_image.convertTo(laplacina_image, CV_8UC3);

	
	//compute the adaptive threshold
	Mat binary_image;
	cvtColor(result_image, binary_image, COLOR_BGR2GRAY);
	adaptiveThreshold(binary_image, binary_image, 250, BORDER_REPLICATE, THRESH_BINARY, 111, 50);


	// Perform the distance transform algorithm
	Mat distance_image;
	distanceTransform(binary_image, processed_image, DIST_L2, 3);


	// Normalize the distance image
	normalize(processed_image, processed_image, 0, 30, NORM_MINMAX);

	//Resize image in order to speed up
	cvtColor(processed_image, processed_image, COLOR_GRAY2BGR);
	resize(processed_image, processed_image, Size(size_processed_image, size_processed_image), INTER_AREA);
	
	cout << "Preprocessing completed" << endl;
}

void printError(string error)
{
	cout << "\n ERROR: " << error << endl;
	exit(1);
}
