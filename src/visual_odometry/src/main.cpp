#include<ros/ros.h>
#include<orb_features.hpp>
#include<coordCaculate.h>
#include<mindVision_init.h>
#include<iostream>
#include<orb_cuda_features.hpp>

//相机内参和畸变参数
Mat cameraMatrix = (Mat_<double>(3,3)<<1038.445291,0.000000,660.017113,0.000000,1038.776144,530.520263,0.000000,0.000000,1.000000);
Mat distortionMatrix = (Mat_<double>(1,5)<<-0.101756,0.086948,0.000260,0.000568,0.000000);

void distortionCorrection(Mat &inputImage,Mat &outputImage)
{
    undistort(inputImage,outputImage,cameraMatrix,distortionMatrix);
}

int main(int argc,char *argv[])
{

    ros::init(argc,argv,"main");
    //定义初始坐标为（0，0，0）
    Point3f realCoor(0.0,0.0,0.0);
    Point3f lastcastCoord(downWidth/2, downHeigth/2, 0);//图像坐标中心点
    MindVisionInit mindvision;
    mindvision.init();
    Mat image1 = mindvision.dstImage;  //首先记录第一张图像
    Mat image2 = image1;
        
    while(ros::ok())
    {
        mindvision.updateImage();
   /*以下是cuda版本测试*/
   
        orb_cuda_features ocf(image1,image2);  
        ocf.orb_cuda_features_matching();
        
        CoordCaculate cc;
        cc.initCoordCaculate();
        //将匹配的特征点放到容器里 分别存放第一幅和第二幅图像匹配的特征点
        cc.storageFeaturePoints(ocf.matches,ocf.keyPoints_1,ocf.keyPoints_2);
        cc.getAllMappedPoints();
        deque<Point3f> result = cc.mappedPoints;
        
        cc.filterMappedPoints();
        cout << "mapped point coord " << cc.mappedPoint << endl;
        Point3f coord = getNowCoord(lastcastCoord, cc.filterPoints);

        realCoor+=coord;  //得到真实坐标
        
        cout<<"当前坐标为："<<realCoor<<endl;

        
        image1 = mindvision.dstImage;
        image2 = image1;
     
    }

    return 0;
}

