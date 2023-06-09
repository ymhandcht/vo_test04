#include<ros/ros.h>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<deque>
#include<numeric>

using namespace cv;
using namespace std;
#define downWidth  640*2
#define downHeigth 512*2
#define x_point downWidth/2
#define y_point downHeigth/2

//安装高度为10cm，相机照地面的实际大小
// #define actual_X 55//55mm
// #define actual_Y 68.75//68.75mm
//像素坐标 640*512 则x方向每个像素点对应坐标为 55/512mm Y方向每个像素点对应坐标为68.75/640mm
#define realDis1 0.334
#define realDis2 0.3344

float calculateRevolveAngle(float coordArrA1[2],float coordArrB1[2],float coordArrA2[2],float coordArrB2[2]);

float calculateDirectionAngle(float y,float x);
float pointToLineDistance(float x1,float y1,float x2,float y2,float x3,float y3);

float getPointToPointDistance(float x1,float y1,float x2,float y2);


Point3f getCenterCoordinate(vector<Point3f>vec1,vector<Point3f>vec2,int i,int j);
Point2f getCrossPoint(float k1,float b1,float k2,float b2);
Point2f getSolution(float a,float b,float c);
Point3f updateCoord(deque<Point3f> &d);
Point3f getNowCoord(Point3f last_point,Point3f new_point);

class CoordCaculate
{
public:
    vector<Point3f> vec1; //记录第一幅图像特征点容器
    vector<Point3f> vec2;  //记录匹配图像特征点容器
    Point3f mappedPoint;  //映射的中心点坐标
    deque<Point3f> mappedPoints; //经过筛选后得到的所有映射中心点的坐标
    Point3f filterPoints;//经过滤波过后的映射点坐标
    float sumX = 0.0;
    float sumY = 0.0;
    float sumZ = 0.0;
    
    //将特征点存放到vec1 vec2两个容器内
    void storageFeaturePoints(vector<DMatch> &matches,vector<KeyPoint> &keyPoints_1,vector<KeyPoint> &keyPoints_2);
    //计算匹配图像对应第一幅图像中心点的映射点  输入为两个位置 且这两个点不是同一个点
    void getMappedPoint(int firstPoint,int secondPoint);
    //筛选得到所有映射中心点坐标
    void getAllMappedPoints();
    //对所有的坐标进行滤波处理
    void filterMappedPoints();
    //初始化
    void initCoordCaculate();

};

