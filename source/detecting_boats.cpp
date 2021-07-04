#include "detecting_boats.h"

int main(int argc, char** argv)
{
	const string pb_model = "C:/data/model/model.pb";
	const string pbtxt_model = "C:/data/model/model.pbtxt";
	const string img_path = "C:/data/train/20130304_054304_04214.jpg";
	
	Mat image = imread(img_path);
	Net model = readNetFromTensorflow(pb_model, pbtxt_model);

	setUseOptimized(true);
	setNumThreads(4);

	/*
	Mat blob2 = blobFromImage(image, 1.0, Size(299, 299), Scalar(127.5, 127.5, 127.5), true, false);
	model.setInput(blob2);
	Mat output2 = model.forward();
	std::cout
		<< "output.size(): " << output2.size() << '\n'
		<< "output.elemSize(): " << output2.elemSize() << '\n'
		<< "output.data():\n";
	for (size_t i = 0; i < output2.cols; ++i)
		std::cout << output2.at<float>(0, i) << ' ';
	cout << output2 << endl;
	cout << "-----" << output2.at<float>(0,0); //<< "-------" << output2.at<double>(1,0) << "------";
	*/

	Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
	ss->setBaseImage(image);
	ss->switchToSelectiveSearchFast();

	vector<Rect> rects;
	ss->process(rects);
	cout << "Total Number of Region Proposals: " << rects.size() << endl;

	while (1) {

		Mat imOut = image.clone();
		for (int i = 0; i < rects.size(); i++) {
			if (i < 500){
				Mat temp = imOut(rects[i]);
				resize(temp, temp, Size(299, 299), INTER_AREA);
				Mat blob = blobFromImage(temp, 1.0, Size(299, 299), Scalar(127.5, 127.5, 127.5), true, false);
				model.setInput(blob);
				Mat output = model.forward();
				if(output.at<float>(0, 1) > 0.8)
					rectangle(imOut, rects[i], Scalar(0, 255, 0));
			}
			else
				break;
		}
		imshow("Output", imOut);
		break;
	}
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