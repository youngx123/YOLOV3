#pragma once
#include<iostream>
#include<ncnn/net.h>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<string>
#include <math.h>
#include <algorithm>
#include <functional>

#include "baseClass.h"


baseClass::baseClass(const int imagesize)
	       :targetSize(imagesize)
{
}

void baseClass::Max_Score_Index(const cv::Vec<float, 80> prob, int &index, float &score)
{
	score = -99999.0;
	for (int i = 0; i < prob.rows; i++)
	{
		if (score < prob[i])
		{
			score = prob[i];
			index = i;
		}
	}
}

float baseClass:: Sigmoid(float &s)
{

	s = 1 / (1 + expf(-s));
	return s;
}

void baseClass::Log(const std::string mess)
{
	std::cout << mess + "\n";
}

void baseClass::drawImage(cv::Mat imageData, std::vector<cv::Vec<float, 7>>&nms_result,
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

		std::string mess = CLASS_NAME[class_id] + ": " + std::to_string(class_prob);
		cv::putText(imageData, mess, cv::Point(int(x1), int(y1 - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
			cv::Scalar(0, 255, 0), 2);

		cv::imwrite(saveFile, imageData);
	}
}


void baseClass::NMS(std::vector<cv::Vec<float, 7>>&det, std::vector<cv::Vec<float, 7>>&result, const float nms_thred)
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

void  baseClass::Bbox_IOU(const cv::Vec<float, 7>&b1, const std::vector<cv::Vec<float, 7>>&b2,
						  float *ious_score)
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
