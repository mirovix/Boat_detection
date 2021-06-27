#include "detecting_boats.h"

int main(int argc, char** argv)
{

	/*
	string line;
	FILE* fmyfile = fopen("C:\data\label.json", "r");
	while (getline(myfile, line)) {

		cout << line << endl << "-";
		printf("%s", line);
	}
	myfile.close();
	system("PAUSE");
	return 0;
	*/

}

//void training_generation() {}

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