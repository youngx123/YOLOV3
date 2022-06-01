#pragma once
#include<iostream>
#include<ncnn/net.h>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<string>
#include <math.h>
#include <algorithm>
#include <functional>

class baseClass
{
public:
	baseClass(const int imagesize);
	float Sigmoid(float &s);
	void Max_Score_Index(const cv::Vec<float, 80> prob, int &index, float &score);
	void NMS(std::vector<cv::Vec<float, 7>>&det, std::vector<cv::Vec<float, 7>>&result, const float nms_thred = 0.5);
	void Bbox_IOU(const cv::Vec<float, 7>&b1, const std::vector<cv::Vec<float, 7>>&b2, float *ious_score);
	void Log(const std::string mess);
	void drawImage(cv::Mat imageData, std::vector<cv::Vec<float, 7>>&nms_result, const std::string saveFile);

public:
	const std::string CLASS_NAME[20] = { "aeroplane", "bicycle", "bird", "boat",
								"bottle", "bus", "car", "cat", "chair",
								"cow", "diningtable", "dog", "horse",
								"motorbike", "person", "pottedplant",
								"sheep", "sofa", "train", "tvmonitor"
	};
	cv::Mat imageData;
	int targetSize;
};

