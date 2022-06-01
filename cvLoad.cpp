#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<string>
#include<cmath>
#include<algorithm>
#include<functional>

#include "cvLoad.h"

bool Greater(const cv::Vec<float, 7>&a, const cv::Vec<float, 7>&b)
{
	return a[4] > b[4];
}


DNN::YOLOV3::YOLOV3(const std::string &onx, const int size)
	: onx_Path(onx)
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

DNN::YOLOV3::~YOLOV3()
{
}


void DNN::YOLOV3::Soft_NMS()
{
}

void DNN::YOLOV3::onnxDetection(const std::string file, const std::string savename)
{
	imageData = cv::imread(file);
	if (imageData.empty())
	{
		Log("\n load data image error");
	}
	int orig_H, orig_W;
	orig_H = imageData.rows;
	orig_W = imageData.cols;

	//imageData.convertTo(imageData, CV_32FC3, 1.0 / 255);

	cv::Mat input;
	input = cv::dnn::blobFromImage(imageData, 1.0/255 ,cv::Size(targetSize, targetSize),cv::Scalar(0, 0, 0));
	cvNet.setInput(input);
	
	cv::Mat outpred;
	outpred = cvNet.forward("output");

	if (outpred.empty())
	{
		Log("\noutpred error");
	}
	
	int pred_Num = 3*(pow(targetSize / 32, 2)+pow(targetSize / 16, 2)+pow(targetSize / 8, 2));

	typedef cv::Vec<float, 85> MyData;
	std::vector<cv::Vec<float, 7>> pred_result;
	//MyData *pred;
	MyData pred1;
	for (int i = 0; i < pred_Num; i++)
	{
		//pred = outpred.ptr<MyData>(i);
		pred1 = outpred.at<MyData>(i);
		cv::Vec<float, 2> xy, wh;
		cv::Vec<float, 80> prob;
		float score;
		for (int j = 0; j < 85; j++)
		{
			if (j == 0 || j == 1)
			{
				xy[j] = pred1[j];
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
		sort(pred_result.begin(), pred_result.end(), Greater);
		// nms
		std::vector<cv::Vec<float, 7>>nms_result;
		NMS(pred_result, nms_result,0.3);
		if (nms_result.size() > 0)
		{
			drawImage(imageData, nms_result, savename);
		}
	}
}


float DNN::YOLOV3::Sigmoid(float &s)
{
	s = 1 / (1 + expf(-s));
	return s;
}


void DNN::YOLOV3::Max_Score_Index(const cv::Vec<float, 80> prob, int &index, float &score)
{
	score = -9999.0;
	for (int k = 0; k < 80; k++)
	{
		if (score < prob[k])
		{
			score = prob[k];
			index = k;
		}
	}
}

void DNN::YOLOV3::NMS(std::vector<cv::Vec<float, 7>>&det, std::vector<cv::Vec<float, 7>>&result, const float nms_thred)
{
	while (det.size())
	{
		result.push_back(det[0]);
		if (det.size() == 1)
			break;
		det.erase(det.begin());
		float ious_score[100] = { 0.0 };
		Bbox_IOU(result.back(), det, ious_score);

		std::vector<cv::Vec<float, 7>> det2;
		for (int i = 0; i < 100; i++)
		{
			if (ious_score[i] != 0.0 && ious_score[i] < nms_thred)
			{
				det2.push_back(det[i]);
			}
		}
		det.clear();
		det = det2;
	}
}

void  DNN::YOLOV3::Bbox_IOU(const cv::Vec<float, 7>&b1,
	const std::vector<cv::Vec<float, 7>>&b2, float *ious_score)
{
	cv::Vec<float, 7> bb1, bb2;
	bb1 = b1;
	for (int j = 0; j < b2.size(); j++)
	{
		bb2 = b2[j];

		float xmin = MAX(bb1[0], bb2[0]);
		float xmax = MIN(bb1[2], bb2[2]);
		float ymin = MAX(bb1[1], bb2[1]);
		float ymax = MIN(bb1[3], bb2[3]);

		float w = xmax - xmin;
		float h = ymax - ymin;
		float union_area = w * h;
		float area1 = (bb1[2] - bb1[0])*(bb1[3] - bb1[1]);
		float area2 = (bb2[2] - bb2[0])*(bb2[3] - bb2[1]);
		ious_score[j] = union_area / (area1 + area2 - union_area);
	}
}

void DNN::YOLOV3::drawImage(cv::Mat imageData, std::vector<cv::Vec<float, 7>>&nms_result,
	const std::string saveFile)
{
	int w, h;
	h = imageData.rows;
	w = imageData.cols;
	for (auto item : nms_result)
	{
		int x1 = w * item[0] / targetSize;
		int y1 = h * item[1] / targetSize;
		int x2 = w * item[2] / targetSize;
		int y2 = h * item[3] / targetSize;
		float class_prob = item[5];
		int class_id = item[6];
		int detW = (x2 - x1) < w ? (x2 - x1) : w;
		int detH = (y2 - y1) < h ? (y2 - y1) : h;
		cv::rectangle(imageData, cv::Rect(x1, y1, detW, detH), cv::Scalar(0, 0, 255), 2);

		//std::string mess = CLASS_NAME[class_id] + ": " + std::to_string(class_prob);
		//cv::putText(imageData, mess, cv::Point(int(x1), int(y1 - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
		//	cv::Scalar(0, 255, 0), 2);

		cv::imwrite(saveFile, imageData);
	}
}

void DNN::YOLOV3::Log(const std::string mess)
{
	std::cout << mess << std::endl;
}