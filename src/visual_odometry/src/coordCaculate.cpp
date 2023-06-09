#include<ros/ros.h>
#include<coordCaculate.h>
#include<math.h>
using namespace cv;
using namespace std;

//计算向量角度
float calculateDirectionAngle(float y,float x)
{   
    float angle=atan2(y,x);;
    if(angle>180) angle-=360;
    else if(angle<-180) angle+=360;
    return angle;  
}

//计算旋转角度
//功能：通过两组坐标求出图像旋转的角度
//参数说明：4个参数分别是第一幅图像的两个特征点坐标和第二幅图像的两个坐标
//返回值是逆时针旋转的角度
float calculateRevolveAngle(float coordArrA1[2],float coordArrB1[2],float coordArrA2[2],float coordArrB2[2])
{   
    float deltay1 = coordArrB1[1]-coordArrA1[1];
    float deltax1 = coordArrB1[0]-coordArrA1[0];
    float deltay2 = coordArrB2[1]-coordArrA2[1];
    float deltax2 = coordArrB2[0]-coordArrA2[0];
    //代表矢量方向，true表示0-180 flase表示0-（-180）
    float angle1 = -calculateDirectionAngle(deltay1,deltax1);
    float angle2 = -calculateDirectionAngle(deltay2,deltax2);
    //计算旋转角度，认为顺时针或者逆时针最多不会超过180度
    cout<<angle2<<"   "<<angle1<<endl;
    float angleResult;
    if(( angle2 - angle1 )>180) angleResult = (angle2-angle1-360);
    else if ((angle2 - angle1)<-180) angleResult = (angle2-angle1+360);
    else angleResult = angle2 - angle1;
    return angleResult; 
}
//点到直线距离
//返回值是距离的平方
//x1 x1 y1 y2 是直线上的两点，x3 y3是要求距离的点
float pointToLineDistance(float x1,float y1,float x2,float y2,float x3,float y3)
{
    float distance;
    if(x1==x2)
    {
        distance = (x3-x1)*(x3-x1);
    }
    else
    {
    float k = (y2-y1)/(x2 - x1);
    float b = y1-k*x1;
    distance = (( k*x3 - y3 +b) *( k*x3 - y3 +b))/(1+k*k);
    }
    
    return distance;
}
//求两条直线交点坐标
//输入 k1 b1 k2 b2 输出 x y
Point2f getCrossPoint(float k1,float b1,float k2,float b2)
{
    Point2f point;
    float x,y;
    if(k1!=k2)
    {
    x = (b2-b1)/(k1-k2);
    y = k1*x+b1;
    }
    else 
    {
        x=0;
        y=0;
    }
    point.x = x;
    point.y = y;
    return point;
}
//求一元二次方程根
//x y 分别是方程的两个根
Point2f getSolution(float a,float b,float c)
{
    Point2f result;
    if(a==0)
    {
        result.x = -(c/b);
        result.y = -(c/b);
    }
    else
    {
        result.x = (-b-sqrt(b*b-4*a*c))/(2*a);
        result.y = (-b+sqrt(b*b-4*a*c))/(2*a);
    }
    return result;
}
//两点之间距离公式
//返回值是距离的平方
float getPointToPointDistance(float x1,float y1,float x2,float y2)
{
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
}

Point3f getNowCoord(Point3f last_point,Point3f new_point)
{
    static Point3f resultCoord;
    float deltaX = new_point.x - last_point.x;
    float deltaY = new_point.y - last_point.y;
    float deltaZ = new_point.z - last_point.z;

    float displacementX = deltaX*realDis1;
    float displacementY = deltaY*realDis2;
    resultCoord.x = displacementX;
    resultCoord.y = displacementY;
    resultCoord.z = deltaZ;
    return resultCoord;
}

void CoordCaculate::storageFeaturePoints(vector<DMatch> &matches,vector<KeyPoint> &keyPoints_1,vector<KeyPoint> &keyPoints_2)
{
    for(int i = 0;i<matches.size();i++)
    {
        int index1 = matches[i].queryIdx;
        int index2 = matches[i].trainIdx;
        float x1 = keyPoints_1[index1].pt.x;
        float y1 = keyPoints_1[index1].pt.y;
        float z1 = 0.0f;
        float x2 = keyPoints_2[index2].pt.x;
        float y2 = keyPoints_2[index2].pt.y;
        float z2 = 0.0f;
        Point3f p1(x1,y1,z1);
        Point3f p2(x2,y2,z2);
        this->vec1.push_back(p1);
        this->vec2.push_back(p2);
    }
}

void CoordCaculate::getMappedPoint(int firstPoint,int secondPoint)
{
    
   // float t_Start = (float)getTickCount();
    Point3f result; //坐标结果
    int i = firstPoint;
    int j = secondPoint;
    //k1代表第一幅图像的两个特征点所成直线的斜率 b1表示截距 直线记为l1
    float k1 = (vec1.at(i).y - vec1.at(j).y)/(vec1.at(i).x - vec1.at(j).x); 
    float b1 = vec1.at(i).y - k1*vec1.at(i).x; 
    //k2表示过图像中点的与l1垂直的直线l2
    float k2 = (-1)/k1;
    float b2 = y_point -k2*x_point;
    //求出两条直线的交点 用数组记录 point[0]表示交点x坐标 point[1]表示交点y坐标
    Point2f crossPoint;
    crossPoint = getCrossPoint(k1,b1,k2,b2);
    //cout<<"交点是："<<crossPoint<<endl;
    //d1表示第一个特征点与交点之间的距离平方
    float d1 = getPointToPointDistance(crossPoint.x,crossPoint.y,vec1.at(i).x,vec1.at(i).y);
   //cout<<"d1 = "<<d1<<endl;
    //d2表示交点与第二个特征点之间的距离平方
    float d2 = getPointToPointDistance(crossPoint.x,crossPoint.y,vec1.at(j).x,vec1.at(j).y);
   // cout<<"d2 = "<<d2<<endl;
    //d3表示图像中心点与第二个特征点距离的平方
    float d3 = getPointToPointDistance(x_point,y_point,vec1.at(j).x,vec1.at(j).y);
    //d4表示第一幅图像中心点到两个特征点所成直线l1的距离
    double d4 = pointToLineDistance(vec1.at(i).x,vec1.at(i).y,vec1.at(j).x,vec1.at(j).y,x_point,y_point);
    //将中心点与第一个特征点所成角度定义为angle1 中心点与第二个特征点所成角度定义为angle2
    float deltaY11 = vec1.at(i).y - y_point;
    float deltaX11 = vec1.at(i).x - x_point;
    float angle1 = cv::fastAtan2(deltaY11,deltaX11);
    float deltaY12 = vec1.at(j).y - y_point;
    float deltaX12 = vec1.at(j).x - x_point;
    float angle2 = cv::fastAtan2(deltaY12,deltaX12);
    

    if(angle1>180) angle1-=360;
    else if(angle1<-180) angle1+=360;
    if(angle2>180) angle2-=360;
    else if(angle2<-180) angle2+=360;
   // cout<<"angle1 = "<<angle1<<" angle2 = "<<angle2<<endl;

    //根据第一幅图像信息计算第二幅图像中心坐标在第一幅图像上面的映射坐标
    //k3是第二幅图像前两个匹配特征点的连线的斜率 b3是对应的斜率 直线记为l2
    float k3 = (vec2.at(i).y - vec2.at(j).y)/(vec2.at(i).x - vec2.at(j).x);
    float b3 = vec2.at(i).y - k3*vec2.at(i).x;
    
    //在直线l2上找到第一幅图像交点对应的点x0 y0   两个距离约束
    
    Point2f rootResult1;
    rootResult1 = getSolution(1+k3*k3,2*k3*(b3-vec2.at(i).y)-2*vec2.at(i).x,vec2.at(i).x*vec2.at(i).x+(b3-vec2.at(i).y)*(b3-vec2.at(i).y)-d1);
    
    //(x11,y11)或者(x12,y12)其中一个是第一幅图像交点的映射点
    float x11 = rootResult1.x;
    float x12 = rootResult1.y;
    float y11 = k3*x11+b3;
    float y12 = k3*x12+b3;
   // cout<<"可能的交点1:"<<"("<<x11<<","<<y11<<")"<<endl;
   // cout<<"可能的交点2:"<<"("<<x12<<","<<y12<<")"<<endl;

    //判断哪个是正确的交点
    float x1,y1; //正确的交点
    float d11 = getPointToPointDistance(x11,y11,vec2.at(j).x,vec2.at(j).y);
    float d12 = getPointToPointDistance(x12,y12,vec2.at(j).x,vec2.at(j).y);
  //  cout<<"d11 = "<<d11<<" d12 = "<<d12<<endl;
    x1 = abs(d11-d2)<abs(d12-d2)?x11:x12;
    y1 = k3*x1 + b3;
  //  cout<<"预测的交点是：("<<x1<<","<<y1<<")"<<endl;
    //k4是与匹配图像两个匹配特征点所成直线垂直直线的斜率
    float k4 = -(1/k3);
    float b4 = y1 - k4*x1;
    //此时找到过原图像中心点映射到匹配图像点的直线，且此直线与两个特征点所成直线垂直
    //在此直线上找到实际映射的第一幅图像中心点的坐标
    Point2f rootResult2 = getSolution(1+k4*k4,2*k4*(b4-y1)-2*x1,x1*x1+(b4-y1)*(b4-y1)-d4);
    float x2,y2;//第一幅图像映射到第二幅图像上点的坐标
    float x21 = rootResult2.x;
    float y21 = k4*x21+b4;
    float x22 = rootResult2.y;
    float y22 = k4*x22+b4;
   // cout<<"映射可能的坐标中心点1是:("<<x21<<","<<y21<<")"<<endl;
  //  cout<<"映射可能的坐标中心点2是:("<<x22<<","<<y22<<")"<<endl;

    //根据与两个特征点之间的角度差值判断准确的映射点
    float deltaY21 = vec2.at(i).y - y21;
    float deltaX21 = vec2.at(i).x - x21;
    float deltaY22 = vec2.at(j).y - y21;
    float deltaX22 = vec2.at(j).x - x21;
    float angle3 = cv::fastAtan2(deltaY21,deltaX21);
    float angle4 = cv::fastAtan2(deltaY22,deltaX22);

    float deltaY23 = vec2.at(i).y - y22;
    float deltaX23 = vec2.at(i).x - x22;
    float deltaY24 = vec2.at(j).y - y22;
    float deltaX24 = vec2.at(j).x - x22;
    float angle5 = cv::fastAtan2(deltaY23,deltaX23);
    float angle6 = cv::fastAtan2(deltaY24,deltaX24);

    if(angle3>180) angle3-=360;
    else if(angle3<-180) angle3+=360;
    if(angle4>180) angle4-=360;
    else if(angle4<-180) angle4+=360;
    //cout<<"angle3 = "<<angle3<<" angle4 = "<<angle4<<endl;
    //cout<<"angle5 = "<<angle5<<" angle6 = "<<angle6<<endl;
    x2 = abs((angle3-angle4)-(angle1 - angle2))<15?x21:x22;
    y2 = k4*x2 + b4;

    
    result.x = x2;
    result.y = y2;
    result.z = 0;
    this->mappedPoint = result;
    //  double t_End = (double)getTickCount();

    //     float tM = 1000*(-t_Start+t_End)/getTickFrequency();
    //     cout<<tM<<" ms"<<endl;
}

void CoordCaculate::getAllMappedPoints()  //只保留50个点 算出的数据是50！
{
    // if(vec1.size()<=50)
    // {
        for(int i = 0;i<vec1.size();i++)
    {
        Point3f castPoint;
        for(int j=i+1;j<vec1.size();j++)
        {
            //判断两个特征点之间距离，太近就滤除 阈值设定500
            if(getPointToPointDistance(vec1.at(i).x,vec1.at(i).y,vec1.at(j).x,vec1.at(j).y)>500
            &&pointToLineDistance(vec1[i].x,vec1[i].y,vec1[j].x,vec1[j].y,downWidth/2,downHeigth/2)>200
            &&abs(vec1[i].x-vec1[j].x)>5)
            {
                this->getMappedPoint(i,j);
                castPoint = mappedPoint;
                if(castPoint.x>=0&&castPoint.x<=downWidth&&castPoint.y>=0&&castPoint.y<=downHeigth)
               this->mappedPoints.emplace_back(castPoint);
            }
        }
    }  
    // }
    /*存在bug*/
    // else
    // {
    //     int num = 0;
    //     for(int i = 0;i<50;i++)
    // {
    //     Point3f castPoint;
    //     for(int j=i+1;j<51;j++)
    //     {
    //         //判断两个特征点之间距离，太近就滤除 阈值设定500
    //         if(getPointToPointDistance(vec1.at(i).x,vec1.at(i).y,vec1.at(j).x,vec1.at(j).y)>500
    //         &&pointToLineDistance(vec1[i].x,vec1[i].y,vec1[j].x,vec1[j].y,downWidth/2,downHeigth/2)>200
    //         &&abs(vec1[i].x-vec1[j].x)>5)
    //         {
    //             this->getMappedPoint(i,j);
    //             castPoint = mappedPoint;
    //             if(castPoint.x>=0&&castPoint.x<=downWidth&&castPoint.y>=0&&castPoint.y<=downHeigth)
    //            this->mappedPoints.emplace_back(castPoint);
    //            num++;
    //         }
    //     }
    // }  
    // }
     
}

bool cmp1(const Point3f &A,const Point3f &B)
    {
        return A.x>B.x;//x降序排列
    }
bool cmp2(const Point3f &A,const Point3f &B)
    {
        return A.y>B.y;//x降序排列
    }
void CoordCaculate::filterMappedPoints()
{
    sort(this->mappedPoints.begin(),this->mappedPoints.end(),cmp1);  //将获取的中心点坐标x进行排序
    //删除掉头尾1/3
    int num1 = this->mappedPoints.size();
    for(int i = 0;i < num1/3;i++)
    {
        this->mappedPoints.pop_back();//删除头部1/3的数据
        this->mappedPoints.pop_front();//删除尾部的1/3数据
    }
    sort(this->mappedPoints.begin(),this->mappedPoints.end(),cmp2);//将获取的中心点坐标y进行排序
    //删除掉头尾1/3
    int num2 = this->mappedPoints.size();
    for(int i = 0;i < num2/3;i++)
    {
        this->mappedPoints.pop_back();//删除头部1/3的数据
        this->mappedPoints.pop_front();//删除尾部的1/3数据
    }
    // //测试输出
    // for(deque<Point3f>::iterator it = this->mappedPoints.begin();it!=this->mappedPoints.end();it++)
    // {
    //     cout<<*it<<endl;
    // }
    //求均值
   for(int i = 0;i<this->mappedPoints.size();i++)
   {
    sumX+=this->mappedPoints[i].x;
    sumY+=this->mappedPoints[i].y;
    sumZ+=this->mappedPoints[i].z;
   }
   if(sumZ<0.3) sumZ = 0.0;
   if(sumX<0.5*this->mappedPoints.size()) sumX = 0.0;
   if(sumY<0.5*this->mappedPoints.size()) sumY = 0.0;
   this->filterPoints.x = sumX/this->mappedPoints.size();
   this->filterPoints.y = sumY/this->mappedPoints.size();
   this->filterPoints.z = sumZ/this->mappedPoints.size();
  
}

void CoordCaculate::initCoordCaculate()
{
    this->mappedPoints.clear();
    this->vec1.clear();
    this->vec2.clear();

}
