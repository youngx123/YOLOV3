#pragma once
#include<iostream>
#include<ncnn/net.h>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<string>
#include<vector>

namespace DNN {
	class YOLOV3
	{
	public:
		YOLOV3(const std::string &onx, const int size);
		~YOLOV3();
		void onnxDetection(const std::string file, const std::string savename);
		float Sigmoid(float &s);
		void Max_Score_Index(const cv::Vec<float, 80> prob,int &index, float &score);
		void NMS(std::vector<cv::Vec<float, 7>>&det, std::vector<cv::Vec<float, 7>>&result, const float nms_thred=0.5);
		void Bbox_IOU(const cv::Vec<float, 7>&b1, const std::vector<cv::Vec<float, 7>>&b2, float *ious_score);
		void Log(const std::string mess);
		void drawImage(cv::Mat imageData, std::vector<cv::Vec<float, 7>>&nms_result,const std::string saveFile);
		
		void Soft_NMS();


	private:
		const std::string onx_Path;
		cv::dnn::Net cvNet;
		cv::Mat imageData;
		const int targetSize = 640;
	};

}


