#include<iostream>
#include<ncnn/net.h>
#include<opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<fstream>
#include<chrono>

//#include "yolo.h"
//#include "cvLoad.h"
#include"Inference.h"

int main()
{
	//std::string paramName = R"(D:\MyNAS\SynologyDrive\Object_Detection\v3\yv3\convetModel\yolov3_ncnn.param)";
	//std::string binName = R"(D:\MyNAS\SynologyDrive\Object_Detection\v3\yv3\convetModel\yolov3_ncnn.bin)";
	
	std::string onnxPath = R"(D:\MyNAS\SynologyDrive\Object_Detection\v3\yv3\convetModel\yolov3.onnx)";
	std::string testDir =  R"(D:\MyNAS\SynologyDrive\Object_Detection\v3\yv3\voc2007test.txt)";
	std::string saveDir = "" ;
	const int trainSize = 416;

	std::ifstream fidRead;
	std::vector <std:: string> fileNames;
	fidRead.open(testDir);
	if (!fidRead.is_open())
	{
		std::cout << "open file : " << testDir << " error !" << std::endl;
		exit(0);
	}

	std::string tempName;
	while (fidRead >> tempName)
	{
		int index = tempName.find("jpg");
		if (index != -1)
		{
			tempName = tempName.substr(0, index+4);
			fileNames.push_back(tempName);
		}
	}

	// start to detect image
	//NCNN::YOLOV3 *model = new NCNN::YOLOV3(paramName, binName, trainSize);
	//YOLOV3 *model = new YOLOV3(onnxPath, trainSize);

	//auto time1 = std::chrono::high_resolution_clock::now();
	//{
	//	//NCNN::YOLOV3 model(paramName, binName, trainSize);
	//	DNN::YOLOV3 model(onnxPath, trainSize);
	//	int index;
	//	std::chrono::time_point<std::chrono::steady_clock> start_time;
	//	std::chrono::time_point<std::chrono::steady_clock> end_time;
	//	std::chrono::duration<float>  duration;


	//	for (auto file : fileNames)
	//	{
	//		index = file.rfind("/");
	//		std::string baseName = file.substr(index+1, file.size());
	//		//std::cout << baseName << std::endl;
	//		std::string saveName = saveDir + baseName;
	//		start_time = std::chrono::high_resolution_clock::now();
	//		std::cout <<"detection image :" << baseName;

	//		//model.ncnnDetection(file, saveName);
	//		model.onnxDetection(file, saveName);

	//		end_time = std::chrono::high_resolution_clock::now();
	//		duration = end_time - start_time;
	//		std::cout << "\ntime consuming : " << duration.count() << " s\n\n";//<< std::endl;

	//	}
	//	//delete model;
	//}
	//auto time2 = std::chrono::high_resolution_clock::now();
	//auto  a  = time2 - time1;
	//std::cout << " total time : " << a.count() << std::endl;

	// use subclass dnnInference
	auto time1 = std::chrono::high_resolution_clock::now();
	{
		Inference::CVdnnInfer model(onnxPath, trainSize);
		int index;
		std::chrono::time_point<std::chrono::steady_clock> start_time;
		std::chrono::time_point<std::chrono::steady_clock> end_time;
		std::chrono::duration<float>  duration;


		for (auto file : fileNames)
		{
			index = file.rfind("/");
			// get input file base name
			std::string baseName = file.substr(index + 1, file.size());

			// file name of deteciton result
			std::string saveName = saveDir + baseName;
			start_time = std::chrono::high_resolution_clock::now();
			std::cout << "detection image :" << baseName;

			model.Detection(file, saveName);

			end_time = std::chrono::high_resolution_clock::now();
			duration = end_time - start_time;
			std::cout << "\ntime consuming : " << duration.count() << " s\n\n";//<< std::endl;

		}
		//delete model;
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	auto  a = time2 - time1;
	std::cout << " total time : " << a.count() << std::endl;

	return 0;
};