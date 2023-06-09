#define _CRT_SECURE_NO_WARNINGS

#ifndef ACT_D435_H_
#define ACT_D435_H_

#include <iostream>
#include <thread> 
#include <mutex>
#include <math.h>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
//#include "kalman_filter.h"
#include "CameraApi.h"
#include <time.h>


#define PI 3.1415926

#define DEBUG			// 测试模式
// #define OBSERVE			// 观察调试模式
// #define COMMUN			// 进程通讯
// #define IFSERIAL		// 串口通讯

#define TIME			// 时间输出
#define CAPTURE			// 截图模式
#define TXT				// 记录数据
// #define RECORD		// 录制视频模式
// #define VIDEO		// 播放视频

//任务
#define MODETOWER 0
#define MODEBALL 1
#define MODEREST 2
 
//红蓝场
#define RED 0
#define BLUE 1

#define IMAGE_WIDTH 	1280*0.75
#define IMAGE_HEIGHT 	1024*0.75

// #define RECORD_WIDTH	1280
// #define RECORD_HEIGHT	1024

#define RECORD_WIDTH	640
#define RECORD_HEIGHT	512

// #define FOURmm
// #define SIXmm
// #define EIGHTmm
#define SIXTEEN

#ifdef FOURmm
#define INTRIN_FX 1040.619
#define INTRIN_FY 1041.951
#define INTRIN_CX 319.9145
#define INTRIN_CY 281.0621
#elif defined(SIXmm)
#define INTRIN_FX 1040.619
#define INTRIN_FY 1041.951
#define INTRIN_CX 319.9145
#define INTRIN_CY 281.0621
#elif defined(EIGHTmm)

// long camera
// #define INTRIN_FX 2051.78198471129
// #define INTRIN_FY 2050.43833080590
// #define INTRIN_CX 673.208502259030
// #define INTRIN_CY 562.648383316971

// short camera
#define INTRIN_FX 2066.31461605229
#define INTRIN_FY 2065.84434756357
#define INTRIN_CX 650.307707858815
#define INTRIN_CY 531.462824412728

#elif defined(SIXTEEN)
// #define INTRIN_FX 4042.55641477483
// #define INTRIN_FY 4042.13585297481
// #define INTRIN_CX 666.064855435824
// #define INTRIN_CY 540.726891236806

#define INTRIN_FX 4037.23335666286
#define INTRIN_FY 4038.12153811515
#define INTRIN_CX 672.352660621964
#define INTRIN_CY 543.217040973380
#endif

// TOWER
#define MAX_BLUE_H 119
#define MIN_BLUE_H 98
#define MAX_RED_H 180
#define MIN_RED_H 175
#define MAX_L_RED_H 5
#define MIN_L_RED_H 0
#define MAX_S 255
#define MIN_S 70
#define MAX_V 255
#define MIN_V 90

// BALL
#define BALL_MAX_BLUE_H 128
#define BALL_MIN_BLUE_H 98
#define BALL_MAX_RED_H 180
#define BALL_MIN_RED_H 170
#define BALL_MAX_L_RED_H 12
#define BALL_MIN_L_RED_H 0
#define BALL_MAX_S 255
#define BALL_MIN_S 80
#define BALL _MAX_V 255
#define BALL_MIN_V 90

#define RX 672.352660621964
#define RY 543.217040973380

using namespace std;
using namespace cv;


//-- ROI of an object
typedef struct
{
	double xMin;
	double xMax;

	double yMin;
	double yMax;

	double zMin;
	double zMax;

} ObjectROI;

//相机内参
struct cameraIntrix
{
	float fx;
	float fy;
	float cx;
	float cy;
};

class MindVisionInit
{
private:
	int					iCameraCounts = 1;
	int 				iStatus = -1;
	tSdkCameraDevInfo 	tCameraEnumList;
	int					hCamera;
	tSdkCameraCapbility	tCapability;
	tSdkFrameHead		sFrameInfo;
	BYTE*				pbyBuffer;
	// IplImage 			*iplImage = NULL;
	int 				channel = 3;//3
	unsigned char       *g_pRgbBuffer; 
	tSdkImageResolution imageResplution;

public:
	MindVisionInit();
	~MindVisionInit();

	bool init(void);

	bool updateImage(void);
	
	void recordImage(Mat recordImage, VideoWriter Vwriter);
	void releaseBuffer(void);

	char Camera = 0; // 判断相机在不在的判断;1正常，0异常；
	int picNum = 217;
	string picName;
	
	Mat srcImage;
	Mat hsvImage;
	Mat dstImage;


	VideoWriter writer;


	
	TickMeter Imgtime;
};
#endif
