//测试opencv加载pytorch模型
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;


int main()
{
	String modelFile = "./torch.onnx";
	String imageFile = "./dog.jpg";

	dnn::Net net = cv::dnn::readNetFromONNX(modelFile); //读取网络和参数
	
	Mat image = imread(imageFile); // 读取测试图片
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	Mat inputBolb = blobFromImage(image, 0.00390625f, Size(32, 32), Scalar(), false, false); //将图像转化为正确输入格式

	net.setInput(inputBolb); //输入图像

	Mat result = net.forward(); //前向计算

	cout << result << endl;
}

