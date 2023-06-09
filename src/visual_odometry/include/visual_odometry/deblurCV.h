#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include<opencv2/highgui/highgui.hpp>
#include<ros/ros.h>
using namespace cv;
using namespace std;

void help();
void calcPSF(Mat& outputImg, Size filterSize, int len, double theta); //计算点扩散图像
void fftshift(const Mat& inputImg, Mat& outputImg);//傅里叶之后得到的结果频率范围是0到fs，为了便于进行频率域滤波，也便于观察频谱信息 ，通常将频率范围调整至-fs/2到fs/2，这样就将零频分量（直流分量）迁移到了图像中心，呈现的效果就是中心低频信息，四周外围是高频信息，这个实现我们就称为fftshift。
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);//频域滤波后输出时域图像
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);//计算维纳滤波器
void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma = 5.0, double beta = 0.2);//使输入图像的边缘逐渐变细，以减少恢复图像中的振铃效应