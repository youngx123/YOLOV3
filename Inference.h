#pragma once
#include"baseClass.h"

namespace Inference
{
	class CVdnnInfer :public baseClass
	{
	public:
		CVdnnInfer(const std::string &onx, const int size);
		~CVdnnInfer();
		void Detection(const std::string file, const std::string savename);
	private:
		const std::string onx_Path;
		cv::dnn::Net cvNet;
	};


	class  NcnnInfer :public baseClass
	{
	public:
		NcnnInfer(const std::string &onx, const int size);
		~NcnnInfer();
		void Detection(const std::string file, const std::string savename);
		void inline makeGrid(cv::Vec<float, 2>&wh, const int id, const int i, const float scale);
		void ncnnMatRead(const ncnn::Mat &out5, std::vector<cv::Vec<float, 7>>&dect_result2);

	private:
		std::string param_Path;
		std::string bin_Path;
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





