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


void knnMatches2(Mat &query,Mat &train,vector<DMatch>&matches)
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

class orb_cuda_features
{
public:
    orb_cuda_features(Mat img1,Mat img2);  //构造函数声明
    void orb_cuda_features_matching();          //特征提取
    
    Mat orignalImage_1; //待匹配图像
    Mat orignalImage_2;
    cuda::GpuMat G_img1,G_img2;    //gpu格式的图像数据
    cuda::GpuMat G_imggray1,G_imggray2;//将图像转化成灰度图像
    cuda::GpuMat G_keypoints1, G_keypoints2; //GPU格式两幅图像的特征点
	cuda::GpuMat G_descriptors1, G_descriptors2, G_descriptors1_32F, G_descriptors2_32F;  //描述子
	vector<KeyPoint> keyPoints_1, keyPoints_2;
	//Mat descriptors1,descriptor2;
	vector<DMatch> matches;
    Mat matchImage;

};
orb_cuda_features::orb_cuda_features(Mat img1,Mat img2)
{
    this->orignalImage_1 = img1;
    this->orignalImage_2 = img2;
    this->G_img1.upload(orignalImage_1);
    this->G_img2.upload(orignalImage_2);
    cuda::cvtColor(G_img1, G_imggray1, COLOR_BGR2GRAY);
	cuda::cvtColor(G_img2, G_imggray2, COLOR_BGR2GRAY);
}

void orb_cuda_features::orb_cuda_features_matching()
{
    auto orb_cuda_detector = cuda::ORB::create(1000,1.2f,8,31,0,2,0,31,20,true);
    
    Ptr<cv::cuda::DescriptorMatcher> G_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(4);

    orb_cuda_detector->detectAndComputeAsync(G_imggray1,cuda::GpuMat(),G_keypoints1,G_descriptors1);
   
    
    orb_cuda_detector->convert(G_keypoints1,keyPoints_1);
   

    G_descriptors1.convertTo(G_descriptors1_32F,CV_32F);
  
    Mat descriptors1(G_descriptors1);

    orb_cuda_detector->detectAndComputeAsync(G_imggray2,cuda::GpuMat(),G_keypoints2,G_descriptors2);
    orb_cuda_detector->convert(G_keypoints2,keyPoints_2);
    G_descriptors1.convertTo(G_descriptors2_32F,CV_32F);
    

    Mat descriptors2(G_descriptors2);

    G_matcher->match(G_descriptors1_32F,G_descriptors2_32F,this->matches);
   

   // cout<<"特征匹配用时："<<time<<"ms"<<endl;
    knnMatches2(descriptors1,descriptors2,matches);
    //cout<<"好的匹配点个数："<<matches.size()<<endl;
    // drawMatches(orignalImage_1,keyPoints_1,orignalImage_2,keyPoints_2,matches,this->matchImage);
    // imshow("src",matchImage);
}