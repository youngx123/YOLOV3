#pragma once
#include<iostream>
#include<ncnn/net.h>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<string>
#include <math.h>
#include <algorithm>
#include <functional>

#include "yolo.h"


bool greater(const cv::Vec<float, 7>&a, const cv::Vec<float, 7>&b)
{
	return a[4] > b[4];
}


NCNN::YOLOV3::YOLOV3(const std::string &param, const std::string &m_bin, const int size)
	:param_Path(param), bin_Path(m_bin),targetSize(size)
{ 
	ncNet.opt.use_vulkan_compute = 1; //开启vulkan
	//ncnn::VulkanDevice vkdev(0);
	//ncNet.set_vulkan_device(&vkdev);

	ncNet.set_vulkan_device(0);

	ncNet.load_param(param_Path.c_str());
	ncNet.load_model(bin_Path.c_str());
	int gpu_count = ncnn::get_gpu_count();
	std::cout << "gpu num: " << gpu_count << std::endl;
	Log("load ncnn model");

}

NCNN::YOLOV3::~YOLOV3()
{
	ncNet.opt.use_vulkan_compute = 0; // close vulkan
	ncNet.clear();
}

void NCNN::YOLOV3::ncnnDetection(const std::string file, const std::string saveFile, 
								const float conf_thred)
{
	imageData = cv::imread(file);
	if (imageData.empty())
	{
		Log("image load failed, check file path");
	}
	int H = this->imageData.rows;
	int W = this->imageData.cols;
	ncnn::Mat inputData = ncnn::Mat::from_pixels_resize(this->imageData.data, ncnn::Mat::PixelType::PIXEL_BGR,
										W, H,this->targetSize,this->targetSize);


	inputData.substract_mean_normalize(mean, norm);

	// input and detect
	ncnn::Extractor ext = ncNet.create_extractor();
	ext.input("Input_Image", inputData);
	ext.set_vulkan_compute(true);

	ncnn::Mat out5, out4, out3;
	//ncnn::VkMat out5, out4, out3;
	ext.extract("out", out5, 1);
	ext.extract("762", out4, 1);
	ext.extract("766", out3, 1);

	std::vector<cv::Vec<float, 7>>dect_result;
	
	//std::cout << "convert opencv Mat data format ...\n";
	//Process(out5, conf_thred, dect_result);
	//Process(out4, conf_thred, dect_result);
	//Process(out3, conf_thred, dect_result);

	//std::vector<cv::Vec<float, 7>>dect_result2;
	//std::cout << "use ncnn data format ...\n";
	ncnnMatRead(out3, dect_result);
	ncnnMatRead(out4, dect_result);
	ncnnMatRead(out5, dect_result);

	std::vector<cv::Vec<float, 7>>nms_result;

	sort(dect_result.begin(), dect_result.end(), greater);

	for (auto &item : dect_result)
	{
		auto temp = item;
		item[0] = temp[0] - temp[2] / 2; //cx -> x1
		item[1] = temp[1] - temp[3] / 2; //cy -> y1
		item[2] = temp[0] + temp[2] / 2; //w -> x2 
		item[3] = temp[1] + temp[3] / 2; //h -> y2
		item[4] = temp[4];
		item[5] = temp[5];
		item[6] = temp[6];
	}
	//std::cout << "NMS\n" ;

	NMS(dect_result, nms_result);

	if(nms_result.size()>0)
	{
		drawImage(imageData, nms_result, saveFile);
	 }
}

void NCNN::YOLOV3::Process(const ncnn::Mat& out, const float conf_thred,
	                       std::vector<cv::Vec<float, 7>>&dect_result)
{
	int cnum = out.c;
	std::vector<cv::Mat> preddata;
	for (int c = 0; c < cnum; c++)
	{
		ncnn::Mat chanledata = out.channel(c);
		cv::Mat a(out.h, out.w, CV_32FC1);
		memcpy((uchar*)a.data, chanledata.data, chanledata.w * chanledata.h * sizeof(float));
		preddata.push_back(a);
	}

	cv::Mat data123;
	cv::merge(preddata, data123);

	// 遍历行，列, 波段
	int gridx_num = data123.rows;
	int gridy_num = data123.cols;
	const int size = 75;
	typedef cv::Vec<float, size> myVector;
	myVector reslt, reslt2;


	for (int gy = 0; gy < gridy_num; gy++)
	{
		myVector * reslt = data123.ptr<myVector>(gy);
		for (int gx = 0; gx < gridx_num; gx++)
		{
			reslt2 = reslt[gx];
			for (int c = 0; c < 3; c++)
			{
				cv::Vec<float, 2>xy, wh;
				cv::Vec<float, 20>prob_class;
				cv::Vec<float, 1> score;

				xy[0] = reslt2[c * 25 + 0];
				xy[1] = reslt2[c * 25 + 1];

				wh[0] = reslt2[c * 25 + 2];
				wh[1] = reslt2[c * 25 + 3];

				score[0] = reslt2[c * 25 + 4];
				for (int i = 0; i < prob_class.rows; i++)
				{
					prob_class[i] = reslt2[c * 25 + i+5];
				}

				Sigmoid(xy);
				Sigmoid(score);
				Sigmoid(prob_class);
				float obj_conf;
				int index;
				Max_score_index(prob_class, obj_conf, index);
				if (score[0] * obj_conf < conf_thred)
					continue;
				else
				{
					float scale = targetSize / gridx_num;
					int id;
					if (gridx_num == 13)
					{
						id = 0;
					}
					else if (gridx_num == 26)
					{
						id = 1;
					}
					else
					{
						id = 2;
					}
					cv::Vec<float, 7>single_result;
					single_result[0] = (xy[0] + gx) * scale;
					single_result[1] = (xy[1] + gy) * scale;

					makeGrid(wh, id, c, scale);

					single_result[2] = wh[0] * scale;
					single_result[3] = wh[1] * scale;
					single_result[4] = score[0];
					single_result[5] = obj_conf;
					single_result[6] = index;

					dect_result.push_back(single_result);
				}
			}
		}
	}
}

void NCNN::YOLOV3::NMS(std::vector<cv::Vec<float, 7>>&det, std::vector<cv::Vec<float, 7>>&result, const float nms_thred)
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

void  NCNN::YOLOV3::Bbox_IOU(const cv::Vec<float, 7>&b1,
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

void NCNN::YOLOV3::drawImage(cv::Mat imageData, std::vector<cv::Vec<float, 7>>&nms_result,
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
		cv::rectangle(imageData,cv::Rect(x1,y1, detW, detH), cv::Scalar(0,0,255),2);

		std::string mess = CLASS_NAME[class_id] + ": " + std::to_string(class_prob);
		cv::putText(imageData, mess, cv::Point(int(x1), int(y1 - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
			cv::Scalar(0, 255, 0), 2);

		cv::imwrite(saveFile, imageData);
	}
}


void NCNN::YOLOV3::Log(const std::string mess)
{
	std::cout << mess << std::endl;
}

void inline NCNN::YOLOV3::Sigmoid(cv::Vec<float, 20>&prob_class)
{
	for (int i = 0; i < prob_class.rows; i++)
	{
		prob_class[i] = 1 / (1 + expf(-prob_class[i]));
	}
}

void inline NCNN::YOLOV3::Sigmoid(cv::Vec<float, 1>&score)
{
	for (int i = 0; i < score.rows; i++)
	{
		score[i] = 1 / (1 + expf(-score[i]));
	}
}

void inline NCNN::YOLOV3::Sigmoid(float &score)
{

	score= 1 / (1 + expf(-score));

}

void inline NCNN::YOLOV3::Sigmoid(cv::Vec<float, 2>&xy)
{
	for (int i = 0; i < xy.rows; i++)
	{
		xy[i] = 1 / (1 + expf(-xy[i]));
	}
}

void inline NCNN::YOLOV3::Max_score_index(const cv::Vec<float, 20>&prob_class, float &score, int &index)
{
	score = -99999.0;
	for (int i = 0; i < prob_class.rows; i++)
	{
		if (score < prob_class[i])
		{
			score = prob_class[i];
			index = i;
		}
	}
}

void inline NCNN::YOLOV3::makeGrid(cv::Vec<float, 2>&wh, const int id, const int i, const float scale)
{
	const float *anc = anchor[id][i];

	wh[0] = expf(wh[0]) * anc[0] / scale;
	wh[1] = expf(wh[1]) * anc[1] / scale;
}


void NCNN::YOLOV3::Soft_NMS()
{
}

void NCNN::YOLOV3::ncnnMatRead(const ncnn::Mat &out, std::vector<cv::Vec<float, 7>>&dect_result2)
{
	int gridx_num = out.h;
	float scale = targetSize / gridx_num;
	const int id = Grid_Scale[gridx_num];

	cv::Vec<float, 2> xy;
	cv::Vec<float, 2> wh;
	cv::Vec<float, 1> score;
	cv::Vec<float, 20>prob_class;

	for (int gy = 0; gy < out.h; gy++)
	{
		for (int gx = 0; gx < out.w; gx++)
		{
			for (int c = 0; c < 3; c++) // 3个尺度
			{
				for (int num = 0; num < 25; num++) //每个尺度25个预测结果
				{
					const float* cdata1 = out.channel(c*25+num);
					cdata1 = cdata1 + gy * out.w;
					if (num <2)
					{
						xy[num] = cdata1[gx];
					}
					else if(num>1 && num <4)
					{
						wh[num-2] = cdata1[gx];
					}
					else if (num == 4)
					{
						score[0] = cdata1[gx];
					}
					else
					{
						prob_class[num-5] = cdata1[gx];
					}
				}
				
				Sigmoid(xy);
				Sigmoid(score);
				Sigmoid(prob_class);
				float obj_conf;
				int index;
				Max_score_index(prob_class, obj_conf, index);
				if (score[0] * obj_conf < 0.5)
					continue;
				else
				{
					cv::Vec<float, 7>single_result;
					single_result[0] = (xy[0] + gx) * scale;
					single_result[1] = (xy[1] + gy) * scale;

					makeGrid(wh, id, c, scale);

					single_result[2] = wh[0] * scale;
					single_result[3] = wh[1] * scale;
					single_result[4] = score[0];
					single_result[5] = obj_conf;
					single_result[6] = index;

					dect_result2.push_back(single_result);
				}
			}
		}
	}
}
