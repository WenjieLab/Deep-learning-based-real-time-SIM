// DLsim_test_v1.cpp 


#include <stdio.h>
#include "tensorflow/c/c_api.h"
#include <iostream>
#include <string>
#include <iterator>
#include <valarray>
#include <array>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <windows.h>
#include <io.h>
#define PREDICT_ERROR 'A'
using namespace std;
// 初始化TensorFlow
TF_Graph* graph = TF_NewGraph();
TF_Status* status = TF_NewStatus();
TF_SessionOptions* session_opts = TF_NewSessionOptions();
TF_Buffer* run_opts = NULL;
const char* saved_model_dir = "saved_model/save_model";
const char* tags = "serve";
int ntags = 1;
int pixelsize = 128;
int outsize = 256;
const char* input_filepath = "img/128/";
const char* inpath = "img/128\\*.tif";
TF_Session* session;
// 记录时间
clock_t start_time, end_time;
// GPU设置
//shijei uint8_t config[16] = { 0x32, 0xe, 0x9, 0x1d, 0x5a, 0x64, 0x3b, 0xdf, 0x4f, 0xd5, 0x3f, 0x20, 0x1, 0x2a, 0x1, 0x30 };
uint8_t config[16] = { 0x32, 0xe, 0x9, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0xd3, 0x3f, 0x20, 0x1, 0x2a, 0x1, 0x30 };
//0.7 uint8_t config[16] = { 0x32, 0xe, 0x9, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0xe6, 0x3f, 0x20, 0x1, 0x2a, 0x1, 0x30 };

// 打印tensorflow版本
void version()
{
	printf("This TensorFlow Version is %s.\n", TF_Version());
}

// 创建TF_Tensor参数
static void DeallocateTensor(void* data, std::size_t, void*)
{
	std::free(data);
#ifdef _DEBUG
	std::cout << "Deallocate tensor" << std::endl;
#endif
}

// 识别逻辑                     
char predict_sim(vector<float> vecs)
{
	/*
	验证模型是否加载成功
	*/
	char f = 'K';
	if (TF_GetCode(status) == TF_Code::TF_OK)
	{
#ifdef _DEBUG
		cout << "Load success!" << endl;
#endif
	}
	else
	{
#ifdef _DEBUG
		printf("%s\n", TF_Message(status));
#endif
		return PREDICT_ERROR;
	}
	// 输出层的数量
	int num_outputs = 1;
	// 申请内存
	TF_Output* output = (TF_Output*)malloc(sizeof(TF_Output) * num_outputs);
	TF_Output out = { TF_GraphOperationByName(graph,"StatefulPartitionedCall"), 0 };
	// 验证Graph是否获取成功
	if (!out.oper) {
#ifdef  _DEBUG
		printf("load graph output error\n");
#endif
		return PREDICT_ERROR;
	}
	else {
#ifdef _DEBUG
		printf("load graph output ok\n");
#endif
	}
	output[0] = out;
	// 输入层的数量
	int num_inputs = 1;
	TF_Output* input = (TF_Output*)malloc(sizeof(TF_Output) * num_inputs);
	TF_Output in = { TF_GraphOperationByName(graph,"serving_default_input_1"), 0 };
	// 验证Graph是否获取成功
	if (!in.oper) {
#ifdef _DEBUG
		printf("load graph input error\n");
#endif
		return PREDICT_ERROR;
	}
	else {
#ifdef _DEBUG
		printf("load graph input ok\n");
#endif
	}
	input[0] = in;

	// 申请内存
	TF_Tensor** input_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * num_inputs);
	TF_Tensor** output_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * num_outputs);
	const array<int64, 4> dims = { 1,pixelsize,pixelsize,9 };
	const array<int64, 4> dim = { 1,outsize,outsize,1 };
	size_t size = sizeof(float);
	// 1*pixelsize*pixelsize*9*size
	for (auto i : dims) {
		size *= abs(i);
	}
	auto data = static_cast<float*>(malloc(size));
	std::copy(vecs.begin(), vecs.end(), data);
	/*
	创建TF_Tensor
	*/
	TF_Tensor* tensor = TF_NewTensor(TF_DataType::TF_FLOAT,
		dims.data(),
		static_cast<int>(dims.size()),
		data,
		size,
		DeallocateTensor,
		nullptr
	);
	// 验证TensorType是否为FLOAT类型
	if (TF_TensorType(tensor) != TF_DataType::TF_FLOAT) {
#ifdef _DEBUG
		cout << "Wrong tensor type" << endl;
#endif
		return PREDICT_ERROR;
	}
	// 验证矩阵维度数是否相符
	if (TF_NumDims(tensor) != dims.size())
	{
#ifdef _DEBUG
		cout << "Wrong number of dimensions" << endl;
#endif
		return PREDICT_ERROR;
	}
	// 验证图片矩阵的尺寸是否相符
	for (int i = 0; i < dims.size(); i++) {
		if (TF_Dim(tensor, i) != dims[i]) {
#ifdef _DEBUG
			cout << "Wrong dimensions size for dim: " << i << endl;
#endif
			return PREDICT_ERROR;
		}
	}
	auto tf_data = static_cast<float*>(TF_TensorData(tensor));
	for (int i = 0; i < vecs.size(); i++) {
		if (tf_data[i] != vecs[i]) {
#ifdef _DEBUG
			cout << "Element: " << i << "does not match" << endl;
#endif
			return PREDICT_ERROR;
		}
	}
	input_values[0] = tensor;
	//start_time = clock();
	TF_SessionRun(session, NULL, input, input_values, num_inputs, output, output_values, num_outputs, NULL, 0, NULL, status);
	// 验证状态
	//end_time = clock();
	//printf("time: %fs\n", (float)(end_time - start_time) / CLOCKS_PER_SEC);
	if (TF_GetCode(status) == TF_Code::TF_OK)
	{
#ifdef _DEBUG
		printf("Session is OK\n");
#endif
	}
	else {
#ifdef _DEBUG
		printf("%s\n", TF_Message(status));
#endif
		return PREDICT_ERROR;
	}
	// 获取模型输出数据
	TF_Tensor * inp = output_values[0];
	auto tf_da = static_cast<float*>(TF_TensorData(inp));
	auto inpp = TF_TensorData(inp);
	// 验证模型输出数据
	if (TF_TensorType(inp) != TF_DataType::TF_FLOAT) {
#ifdef _DEBUG
		cout << "Wrong tensor type" << endl;
#endif
		return PREDICT_ERROR;
	}
	// 验证矩阵维度数是否相符
	if (TF_NumDims(inp) != dim.size())
	{
#ifdef _DEBUG
		cout << "Wrong number of dimensions" << endl;
#endif
		return PREDICT_ERROR;
	}
	// 验证图片矩阵的尺寸是否相符
	for (int i = 0; i < dim.size(); i++) {
		if (TF_Dim(inp, i) != dim[i]) {
#ifdef _DEBUG
			cout << "Wrong dimensions size for dim: " << i << endl;
#endif
			return PREDICT_ERROR;
		}
	}
	//float* results = static_cast<float*>(TF_TensorData(output_values[0]));
	//const float* camBuf = (float*)TF_TensorData(*output_values);
	//cout << *((float*)results)<< endl;
	cv::Mat mat(outsize, outsize, CV_32F);
	int ind = 0;
	float tst = 0.0;
	for (int i = 0; i < outsize; i++) {
		for (int j = 0; j < outsize; j++) {
			ind = i * outsize + j;
			//tst= *(camBuf + ind);
			mat.at<float>(i, j) = tf_da[ind];
			//cout << *((float*)inpp+ ind) << endl;
		}
	}
	cv::Mat o;
	//cout << *((float*)inpp) << endl;

	//std::memcpy(mat.data, camBuf, sizeof(TF_Tensor*) * num_outputs);
	//cv::normalize(mat, mat, 0, 255, CV_MINMAX);
	cv::normalize(mat, mat, 0.0, 255.0, CV_MINMAX);
	mat.convertTo(o, CV_8U);
	cv::imwrite("2.tif", o);
	//cv::normalize(mat, mat, 0.0, 1, CV_MINMAX);
	//cv::imwrite("2.tif", mat);
	//cv::imshow("out", mat);
	//cv::waitKey(0);
#ifdef _DEBUG
	 //cout << max_value << endl;
	 //cout << letter[max_location] << endl;
#endif
	// 释放内存
	free(input);
	free(output);
	free(input_values);
	free(output_values);
	//free(tensor);
	// 返回结果
	//char fin = 'f';
	//return fin;
	return 'f';
}

// 用于释放TF模型的内存
void freesession() {
	TF_DeleteGraph(graph);
	TF_DeleteSession(session, status);
	TF_DeleteSessionOptions(session_opts);
	TF_DeleteStatus(status);
}

// 用于分割图片
cv::Mat split_img(cv::Mat img, int shape[]) {
	cv::Mat region_img = cv::Mat(cv::Size(shape[1] - shape[0], 25), CV_8UC3, cv::Scalar(255, 255, 255));
	for (int i = 0; i < img.rows; i++) {
		for (int j = shape[0]; j < shape[1]; j++) {
			region_img.at<cv::Vec3b>(i, j - shape[0])[0] = img.at<cv::Vec3b>(i, j)[0];
			region_img.at<cv::Vec3b>(i, j - shape[0])[1] = img.at<cv::Vec3b>(i, j)[1];
			region_img.at<cv::Vec3b>(i, j - shape[0])[2] = img.at<cv::Vec3b>(i, j)[2];
		}
	}
	return region_img;
}
//测试vector数据是否正确
void te(vector<float> vecs) {
	cv::Mat m(pixelsize, pixelsize, CV_8U); ;
	for (int h = 0; h < pixelsize; h++) {
		for (int w = 0; w < pixelsize; w++) {
			//id = m_3.step[0] * n + m_3.step[1] * c + m_3.step[2] * h + w*m_3 .step[3];
			//id = m_3.step[0] * c + m_3.step[1] * h + w * m_3.step[2];
			int id1 = pixelsize * h + w + 262144;
			m.at<uchar>(h, w) = vecs[id1] * 255;

		}
	}
	cv::imshow("test", m);
	cv::waitKey(0);

}
string predict(std::string inPath) {
	cv::Mat img;
	int samples_size[3];
	// 存储九张图;
	samples_size[2] = pixelsize;
	samples_size[1] = pixelsize;
	samples_size[0] = 9;
	cv::Mat m_3 = cv::Mat::zeros(3, samples_size, CV_32F);

	string result = "";
	vector<float> temp = {};
	intptr_t handle;
	int c = 0;
	struct _finddata_t fileinfo;
	handle = _findfirst(inPath.c_str(), &fileinfo);
	if (handle == -1)
		return "Error: no such file";
	do
	{
		string filepath = input_filepath+ (string)fileinfo.name;
		// 读取到存储矩阵
		img = cv::imread(filepath,CV_16U);
		cv::normalize(img, img, 0.0, 255.0, CV_MINMAX);
		img.convertTo(img, CV_8U);
		int id = 0;
		for (int h = 0; h < pixelsize; h++) {
			for (int w = 0; w < pixelsize; w++) {
				id = pixelsize * h + w;
				uchar pix = img.data[id];
				m_3.at<float>(c, h, w) = pix;
				//cout <<(float)m_3.at<uchar>(c, h, w) << endl;
			}
		}
		c++;
	} while (!_findnext(handle, &fileinfo));
	// 关闭查找句柄
	_findclose(handle);
	//cout << "[" << img.cols << ","     //宽度
	//	<< img.rows << "]" << endl;    //高度
	//cout << img.channels() << endl;

	int id1;
	for (int h = 0; h < pixelsize; h++) {
		for (int w = 0; w < pixelsize; w++) {
			for (int c = 0; c < 9; c++) {
				//存储为vector
				id1 = pixelsize * h + w;
				temp.push_back((float)((m_3.at<float>(c, h, w)) / 255.0));
			}
		}
	}
	vector<float> temp1 = temp;
	start_time = clock();
	result = predict_sim(temp);
	end_time = clock();
	printf("predict_sim_time: %fs\n", (float)(end_time - start_time) / CLOCKS_PER_SEC);

	if (result.find(PREDICT_ERROR) == result.npos) //没有找到PREDICT_ERROR
	{
		return "ok";
	}
	else
		return "Error";
}
//模型初始化流程
int main(int nargv, const char* argvs[]) {
	// 打印版本
	version();
	TF_SetConfig(session_opts, (void*)config, 16, status);
	session = TF_NewSession(graph, session_opts, status);
	if (nargv > 1) {
		session = TF_LoadSessionFromSavedModel(session_opts, run_opts, saved_model_dir, &tags, ntags, graph, NULL, status);
		string result = "";
		result = predict(argvs[1]);
		const char* test = result.c_str();
		printf("%s\n", test);
	}
	else {
		// 记录开始时间
		// 加载模型
		session = TF_LoadSessionFromSavedModel(session_opts, run_opts, saved_model_dir, &tags, ntags, graph, NULL, status);
		string result = "";
		// 遍历文件夹下的所有.tif文件
		std::string inPath = inpath;
		for (int te = 0; te < 10; te++) {
			//start_time = clock();
			result = predict(inPath);
			//end_time = clock();
			//printf("predict_time: %fs\n", (float)(end_time - start_time) / CLOCKS_PER_SEC);
		}
	}

	// 释放内存
	freesession();
}
