#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

//Point2f point;

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const int MAX_COUNT = 50;

int main()
{
    VideoCapture capture;
    Mat cameraFrame, grayFrame, grayFrameOld;
    bool initializeFeatureTracking = true;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    vector<Point2f> pointsOld;
    vector<Point2f> pointsNew;
    Size subPixWinSize(10,10);
    Size subPixZeroZone(-1,-1);
    Size winSizeForOpticalFlow(31,31);
    
    vector<uchar> status;
    vector<float> err;

    capture.open(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);

    capture.read(cameraFrame);
    //cameraFrame.copyTo(grayFrame);
    cvtColor(cameraFrame, grayFrame, COLOR_BGR2GRAY);

    while(1)
    {
        capture.read(cameraFrame);
        cvtColor(cameraFrame, grayFrame, COLOR_BGR2GRAY);

        if(initializeFeatureTracking)
        {
            goodFeaturesToTrack(grayFrame, pointsNew, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
            cornerSubPix(grayFrame, pointsNew, subPixWinSize, subPixZeroZone, termcrit);
            initializeFeatureTracking = false;
        }

        if(pointsOld.empty())
        {
            pointsOld = pointsNew;
        }

        if(grayFrameOld.empty())
        {
            grayFrame.copyTo(grayFrameOld);
        }
        calcOpticalFlowPyrLK(grayFrameOld, grayFrame, pointsOld, pointsNew, status, err, winSizeForOpticalFlow, 3, termcrit, 0, 0.001);

        for(int i = 0; i < pointsNew.size(); i++)
        {
            circle( cameraFrame, pointsNew[i], 3, Scalar(0,255,0), -1, 8);
        }

        imshow("KLT Tracker Plain Image", cameraFrame);
        imshow("KLT Tracker Gray Image", grayFrame);
        waitKey(30);
        swap(pointsNew, pointsOld);
        swap(grayFrameOld, grayFrame);
    }
}
