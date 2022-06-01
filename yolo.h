#pragma once
#include<iostream>
#include<ncnn/net.h>
#include<opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<map>

namespace NCNN {
	class YOLOV3
	{
	public:
		YOLOV3(const std::string &param, const std::string &m_bin, const int size);
		YOLOV3(const std::string &onx, const int size);
		~YOLOV3();
		void Process(const ncnn::Mat& out,const float conf_thred, std::vector<cv::Vec<float, 7>>&dect_result);
		void ncnnDetection(const std::string file, const std::string saveFile, const float conf_thred=0.5);
		void NMS(std::vector<cv::Vec<float, 7>>&det, std::vector<cv::Vec<float, 7>>&result, const float nms_thred = 0.4);
		void Soft_NMS();
		void Bbox_IOU(const cv::Vec<float, 7>&b1, const std::vector<cv::Vec<float, 7>>&b2, float *ious_score);
		void drawImage(cv::Mat imageData,std::vector<cv::Vec<float, 7>>&nms_result, const std::string saveFile);
		void inline Sigmoid(cv::Vec<float, 20>&prob_class);
		void inline Sigmoid(cv::Vec<float, 1>&score);
		void inline Sigmoid(cv::Vec<float, 2>&xy);
		void inline Sigmoid(float &score);
		void inline makeGrid(cv::Vec<float, 2>&wh, const int id, const int i, const float scale);
		void inline Max_score_index(const cv::Vec<float, 20>&prob_class, float &score, int &index);
		void Log(const std::string mess);
		void ncnnMatRead(const ncnn::Mat &out5, std::vector<cv::Vec<float, 7>>&dect_result2);

	private:
		const std::string param_Path;
		const std::string bin_Path;
		const std::string onx_Path;
		ncnn::Net ncNet;

		cv::dnn::Net cvNet;
		cv::Mat imageData;
		int targetSize;
		const float anchor[3][3][2] = {
								{{206.0, 154.0}, {174.0, 298.0}, {343.0, 330.0 }},
								{{46.0, 133.0 }, {88.0, 93}, {94.0, 207.0}},
								{{15.0, 27.0 }, {25.0, 72}, {49, 43.0}}
								};

		const std::string CLASS_NAME[20] = { "aeroplane", "bicycle", "bird", "boat",
									"bottle", "bus", "car", "cat", "chair",
									"cow", "diningtable", "dog", "horse",
									"motorbike", "person", "pottedplant",
									"sheep", "sofa", "train", "tvmonitor"
		};
		std::map<int, int> Grid_Scale = { {13,0}, {26,1},{52, 2} };
		const float mean[3] = { 0.f, 0.f, 0.f };
		const float norm[3] = { 1 / 255.f,1 / 255.f,1 / 255.f };

	};
}

