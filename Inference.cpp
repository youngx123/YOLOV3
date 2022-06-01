#include <algorithm>
#include <functional>
#include "Inference.h"


bool GreaterCompare(const cv::Vec<float, 7>&a, const cv::Vec<float, 7>&b)
{
	return a[4] > b[4];
}


Inference::CVdnnInfer::CVdnnInfer(const std::string &onx, const int size)
	: onx_Path(onx), baseClass(size)
{
	cvNet = cv::dnn::readNetFromONNX(onx_Path);
	if (cvNet.empty())
	{
		Log("load file error !");
	}
	else
	{
		Log("load model sucess ");
	}
}

Inference::CVdnnInfer::~CVdnnInfer()
{

}
void Inference::CVdnnInfer:: Detection(const std::string file, const std::string savename)
{
	imageData = cv::imread(file);
	if (imageData.empty())
	{
		Log("\n load image data error");
	}
	int orig_H, orig_W;
	orig_H = imageData.rows;
	orig_W = imageData.cols;

	//imageData.convertTo(imageData, CV_32FC3, 1.0 / 255);

	cv::Mat input;
	input = cv::dnn::blobFromImage(imageData, 1.0 / 255, cv::Size(targetSize, targetSize), cv::Scalar(0, 0, 0));
	cvNet.setInput(input);

	cv::Mat outpred;

	//get output node name
	//auto outputNode = cvNet.getUnconnectedOutLayersNames();
	//for (auto i :outputNode)
	//	std::cout << i << std::endl;

	outpred = cvNet.forward("out_pred");

	if (outpred.empty())
	{
		Log("\n net predict error");
	}

	int pred_Num = 3 * (pow(targetSize / 32, 2) + pow(targetSize / 16, 2) + pow(targetSize / 8, 2));

	typedef cv::Vec<float, 25> MyData;
	std::vector<cv::Vec<float, 7>> pred_result;
	MyData pred1;
	for (int i = 0; i < pred_Num; i++)
	{
		pred1 = outpred.at<MyData>(i);
		cv::Vec<float, 2> xy, wh;
		cv::Vec<float, 80> prob;
		float score;
		for (int j = 0; j < 85; j++)
		{
			if (j == 0 || j == 1)
			{
				xy[j] = Sigmoid(pred1[j]);
			}
			else if (j == 2 || j == 3)
			{
				wh[j - 2] = pred1[j];
			}
			else if (j == 4)
			{
				score = pred1[j];
			}
			else
			{
				prob[j - 5] = pred1[j];
			}
		}
		int index;
		float max_score;
		Max_Score_Index(prob, index, max_score);
		if (score*max_score < 0.4)
			continue;
		else
		{
			cv::Vec<float, 7 > temp;
			temp[0] = xy[0] - wh[0] / 2;
			temp[1] = xy[1] - wh[1] / 2;
			temp[2] = xy[0] + wh[0] / 2;
			temp[3] = xy[1] + wh[1] / 2;
			temp[4] = score;
			temp[5] = index;
			temp[6] = max_score;
			pred_result.push_back(temp);
		}
	}
	if (pred_result.size() > 0)
	{
		sort(pred_result.begin(), pred_result.end(), GreaterCompare);
		
		// nms
		std::vector<cv::Vec<float, 7>>nms_result;
		NMS(pred_result, nms_result, 0.3);
		
		// draw detection result on image and save 
		if (nms_result.size() > 0)
		{
			drawImage(imageData, nms_result, savename);
		}
	}
}

