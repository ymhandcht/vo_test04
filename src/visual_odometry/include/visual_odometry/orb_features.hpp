#include<ros/ros.h>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>

#include "opencv2/cudabgsegm.hpp"
#include "opencv2/core/cuda.hpp"
#include"opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudawarping.hpp"

using namespace cv;
using namespace std;
using namespace cuda;

void knnMatches(Mat &query,Mat &train,vector<DMatch>&matches)
{
    vector<vector<DMatch>> knn_matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.knnMatch(query,train,knn_matches,2);

    float min_dist = FLT_MAX;
    for (int r = 0; r < knn_matches.size(); ++r)
    {
        if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
            continue;

        float dist = knn_matches[r][0].distance;
        if (dist < min_dist) min_dist = dist;
    }

    matches.clear();
    for (size_t r = 0; r < knn_matches.size(); ++r)
    {
        if (
            knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
            knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
            )
            continue;
        matches.push_back(knn_matches[r][0]);
    }

}
class orb_features
{
public:
    Mat orignalImage_1; //待匹配图像
    Mat orignalImage_2;

    vector<KeyPoint> keyPoints_1;//存储图像特征点的数组
    vector<KeyPoint> keyPoints_2;
    
    Mat descriptors_1;  //描述子是矩阵表示原因在于：每一行表示一个特征点的描述子 每一列是描述子具体值 1 0
    Mat descriptors_2;
    vector<DMatch> matches;//匹配结果
    
    Mat image_match;//画出匹配图像存储的图像数组

    orb_features(Mat image1,Mat image2);//构造函数
    void orb_features_matching();//特征提取的成员函数
};

orb_features::orb_features(Mat image1,Mat image2)
{
    orignalImage_1 = image1;
    orignalImage_2 = image2;
}

void orb_features::orb_features_matching()
{
    float mStart = (float)getTickCount();
    Ptr<FeatureDetector> detector = cv::ORB::create();
    //auto detector = cv::cuda::ORB::create();

    Ptr<DescriptorExtractor> descriptor = cv::ORB::create();
    //auto descriptor  = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);
    
  

    //Ptr<DescriptorMatcher>macher = DescriptorMatcher::create("BruteForce-Hamming");
  
    //检测orinented fast角点位置
    detector->detect(orignalImage_1,keyPoints_1);
    detector->detect(orignalImage_2,keyPoints_2);
    float t_End1 = (float)getTickCount();
    float tDetector = 1000*(t_End1-mStart)/getTickFrequency();
    cout<<"提取特征点耗时："<<tDetector<<endl;
    //cout<<"keypoint"<<keyPoints_1.size()<<endl;
    //根据角点位置计算brief描述子
    descriptor->compute(orignalImage_1,keyPoints_1,descriptors_1);
    descriptor->compute(orignalImage_2,keyPoints_2,descriptors_2);
    float t_End2 = (float)getTickCount();
    float tCompute = 1000*(t_End2-t_End1)/getTickFrequency();
   // cout<<"计算描述子用时："<<tCompute<<endl;
    // float mEnd = (float)getTickCount();
    // float tM = -1000*(mStart-mEnd)/getTickFrequency();
    // cout<<"特征匹配程序用时："<<tM<<"ms"<<endl;



    // float mStartmatch = (float)getTickCount();
    //对两幅图像进行匹配，使用汉明距离
    BFMatcher matcher(NORM_HAMMING,true);
    matcher.match(descriptors_1,descriptors_2,matches);
    float mEnd = (float)getTickCount();
    cout<<"普通用时："<<1000*(mEnd-mStart)/(float)getTickFrequency()<<endl;
    knnMatches(descriptors_1,descriptors_2,matches);
    cout<<"普通好的匹配点个数："<<matches.size()<<endl;
    // float mEndmatch = (float)getTickCount();
    // float tM1 = -1000*(mStartmatch-mEndmatch)/getTickFrequency();
    // cout<<"特征匹配程序用时123："<<tM1<<"ms"<<endl;

    
    
    drawMatches(orignalImage_1,keyPoints_1,orignalImage_2,keyPoints_2,matches,image_match);
    
    imshow("匹配图",image_match);
    
}
