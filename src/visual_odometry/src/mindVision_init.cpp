#include<ros/ros.h>
#include "mindVision_init.h"
#include "CameraApi.h"

#ifdef RECORD
VideoWriter writer("../src/cv/videos/dst.avi", VideoWriter::fourcc('P', 'I', 'M', '1'), 30.0, Size(RECORD_WIDTH, RECORD_HEIGHT));

#endif

#ifdef VIDEO
VideoCapture cap("../src/cv/videos/src.avi");
#endif

MindVisionInit::MindVisionInit(/* args */)
{
}

MindVisionInit::~MindVisionInit()
{
}

bool MindVisionInit::init(void)
{
    cout << "Camera initing ······" << endl;
    CameraSdkInit(1);
    iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);

    printf("state = %d\n", iStatus);

    printf("count = %d\n", iCameraCounts);

    if (iCameraCounts == 0)
    {
        return false;
    }

    iStatus = CameraInit(&tCameraEnumList, -1, -1, &hCamera);

    printf("status = %d\n", iStatus);
    if (iStatus != CAMERA_STATUS_SUCCESS)
    {
        return false;
    }
    //CameraSetExposureTime(hCamera,10000);

    CameraGetCapability(hCamera, &tCapability);

    CameraGetImageResolution(hCamera, &imageResplution);
    imageResplution.iIndex = 0xff;
    imageResplution.iHeight = IMAGE_HEIGHT;
    imageResplution.iWidth = IMAGE_WIDTH;
    imageResplution.iWidthFOV = IMAGE_WIDTH;
    imageResplution.iHeightFOV = IMAGE_HEIGHT;
    if (CameraSetImageResolution(hCamera, &imageResplution) == CAMERA_STATUS_SUCCESS)
    {
        cout << "success" << endl;
    }
    else
    {
        init();
    }

    g_pRgbBuffer = (unsigned char *)malloc(tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);

    CameraPlay(hCamera);

    if (tCapability.sIspCapacity.bMonoSensor)
    {
        channel = 1;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
    }
    else
    {
        channel = 3;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
    }
    double Exptime;

#ifdef FOURmm
    //设置手动曝光
    CameraSetAeState(hCamera, false); //false 手动 true 自动

    //设置曝光时间
    CameraSetExposureTime(hCamera, 20000); //单位us

    //设置色温
    CameraSetPresetClrTemp(hCamera, 1);

    //设置模拟增益
    CameraSetAnalogGain(hCamera, 64); //64 是1倍增益

    //设置饱和度
    CameraSetSaturation(hCamera, 125);

    //设置对比度
    CameraSetContrast(hCamera, 100);
#elif defined(SIXmm)
    //设置手动曝光
    CameraSetAeState(hCamera, false); //false 手动 true 自动

    //设置曝光时间
    CameraSetExposureTime(hCamera, 20000); //单位us

    //设置色温
    CameraSetPresetClrTemp(hCamera, 1);

    //设置模拟增益
    CameraSetAnalogGain(hCamera, 128); //64 是1倍增益

    //设置饱和度
    CameraSetSaturation(hCamera, 125);

    //设置对比度
    CameraSetContrast(hCamera, 100);
#elif defined(EIGHTmm)
    //设置手动曝光
    CameraSetAeState(hCamera, false); //false 手动 true 自动

    //设置曝光时间
    CameraSetExposureTime(hCamera, 25000); //单位us

    //设置色温
    CameraSetPresetClrTemp(hCamera, 1);

    //设置模拟增益
    CameraSetAnalogGain(hCamera, 144); //64 是1倍增益

    //设置饱和度
    CameraSetSaturation(hCamera, 125);

    //设置对比度
    CameraSetContrast(hCamera, 100);
#elif defined(SIXTEEN)

    //设置手动曝光
    CameraSetAeState(hCamera, false); //false 手动 true 自动

    //设置曝光时间
    CameraSetExposureTime(hCamera, 8000); //单位us 28000

    //设置色温
    CameraSetPresetClrTemp(hCamera, 1);

    //设置模拟增益
    CameraSetAnalogGain(hCamera, 144); //64 是1倍增益

    //设置饱和度
    CameraSetSaturation(hCamera, 125);

    //设置对比度
    CameraSetContrast(hCamera, 100);

#endif

    CameraGetImageResolution(hCamera, &imageResplution);
    CameraGetExposureTime(hCamera, &Exptime);

    this->updateImage();
    this->updateImage();
    this->updateImage();
    this->updateImage();

    cout << "camera init done !!" << endl;
    cout << "Exptime: " << Exptime << " width: " << imageResplution.iWidth << " height: " << imageResplution.iHeight << endl;

    return true;
}

bool MindVisionInit::updateImage(void)
{
#ifdef VIDEO
    cap >> srcImage;
    // return true;
#else
    Imgtime.start();
    if (CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 400) == CAMERA_STATUS_SUCCESS)
    {

        CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);

        srcImage = cv::Mat(
            cv::Size(sFrameInfo.iWidth, sFrameInfo.iHeight),
            sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
            g_pRgbBuffer);
        CameraReleaseImageBuffer(hCamera, pbyBuffer);
        Imgtime.stop();
       // cout << "updateImg:" << Imgtime.getTimeMilli() << endl;
        Imgtime.reset();
    }
    else
    {
        Imgtime.stop();
        cout << "updateError" << endl;
        Imgtime.reset();
        releaseBuffer();
        return false;
    }
#endif

    dstImage = srcImage.clone();

#ifdef CAPTURE
    if (char(cv::waitKey(1)) == 's')
    {
        picNum++;
        picName = "../src/cv/images/" + std::to_string(picNum) + ".jpg";
        cv::imwrite(picName, srcImage);
    }

#endif
    return true;
}

void MindVisionInit::recordImage(Mat recordImage, VideoWriter Vwriter)
{
    resize(recordImage, recordImage, Size(RECORD_WIDTH, RECORD_HEIGHT));
    Vwriter << recordImage;

    return;
}

void MindVisionInit::releaseBuffer(void)
{
    CameraReleaseImageBuffer(hCamera, pbyBuffer);
}