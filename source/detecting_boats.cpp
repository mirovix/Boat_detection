#include "detecting_boats.h"

int main(int argc, char** argv)
{
	const string pb_model = "C:/data/model/model.pb";
	const string pbtxt_model = "C:/data/model/model.pbtxt";
	//const string img_path = "C:/data/train/20130412_153327_37259.jpg";
	const string img_path = "C:/data/venice_dataset/venice_dataset/06.png";
	
	Mat image = imread(img_path);
	Net model = readNetFromTensorflow(pb_model, pbtxt_model);

	setUseOptimized(true);
	setNumThreads(4);

	Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
	ss->setBaseImage(image);
	ss->switchToSelectiveSearchFast();

	vector<Rect> rects;
	ss->process(rects);
	cout << "Total Number of Region Proposals: " << rects.size() << endl;


	vector<Rect> bounday_box;
	vector<float> scores;
	float th_score = 0.15;
	
	Mat imOut = image.clone();
	for (int i = 0; i < rects.size(); i++) {
		if (i < 800){
			if (i % 20 == 0)
				cout << i << endl;
			Mat temp = imOut(rects[i]);
			resize(temp, temp, Size(299, 299), INTER_AREA);
			Mat blob = blobFromImage(temp, 1.0, Size(299, 299));
			model.setInput(blob);
			Mat output = model.forward();
			if (output.at<float>(0, 0) > 0.6) {
				bounday_box.push_back(rects[i]);
				scores.push_back(output.at<float>(0, 0));
			}
		}
		else
			break;
	}
	vector<int> indices;
	NMSBoxes(bounday_box, scores, 0.1f, 0.3f, indices);
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