#include <iostream>
#include <fstream>
#include <vector>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <ctime>
using namespace std;
using namespace cv;
using namespace ml;

string get_today_time_for_spec()
{
	//��ȡ��ǰʱ��
	time_t now = time(0);
	//��ȡ�ṹ��ʱ��
	tm ltm;
	localtime_s(&ltm, &now);
	string time = to_string(ltm.tm_year) + to_string(ltm.tm_mon) + to_string(ltm.tm_mday) + 
		to_string(ltm.tm_hour) + to_string(ltm.tm_min) + to_string(ltm.tm_sec);
	return time;
}

Ptr<TrainData> generate_fake_data()
{
	return TrainData::loadFromCSV("result.csv",1,0,1);
	
}

vector<string> getLabels()
{
	ifstream fp("result.csv");
	vector<string> labels;
	string line;
	getline(fp, line);
	while (getline(fp, line)) {
		stringstream ss(line);
		string str;
		getline(ss, str, ',');
		labels.push_back(str);
	}
	return labels;
}

int main()
{
	string columns[] = {"radarType", "RFMean", "RFMax", "RFMin", "PRIMean", "PRIMax", "PRIMin", "PWMean", "PWMax", "PWMin"};
	string error_columns[] = {"RFMean", "RFMax", "RFMin", "PRIMean", "PRIMax", "PRIMin", "PWMean", "PWMax", "PWMin", "radarType", "predict_type"};

	//��ȡ��ͬģ��·��
	string model_path = "model";
	string info_path = "info";
	string error_path = "error";
	//��ȡʱ��
	string time = get_today_time_for_spec();

	bool duplicate_types[] = {true,false};
	for (bool duplicate_type : duplicate_types) {
		if (duplicate_type)
			string duplicate_flag = "dup";
		else
			string duplicate_flag = "dedup";
		//��ȡ����
		Ptr<TrainData> data = generate_fake_data();
		//ɾ��ȱʧ����
		//ɾ���ظ���
		//��������
		//�ָ��ǩ
		//���ֲ��Լ���ѵ����
		data->setTrainTestSplitRatio(0.7);
		Mat x_train = data->getTrainSamples();
		Mat x_train_idx = data->getTrainSampleIdx();
		Mat x_test = data->getTestSamples();
		Mat x_test_idx = data->getTestSampleIdx();
		Mat y_train = data->getTrainResponses();
		Mat y_test = data->getTestResponses();
		cout << x_train << endl;
		cout << x_train_idx << endl;
		cout << x_test << endl;
		cout << x_test_idx << endl;
		//TODO������������ı�ǩ����������Ҫ���´��ļ��ж�ȡ��ǩ
		cout << y_train << endl;
		cout << y_test << endl;
	}
	return 0;
}