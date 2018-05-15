#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

Point2f centerOfFeatureTrackingArea;
bool initializeFeatureTracking = false;

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const int MAX_COUNT = 15;

int main()
{
	//**********Swap the file path to switch between the Drone tracking video or Thermal Video
	string filePath = "bebop_images/frame";
	//string filePath = "robot_thermal_data/tau2_fog_";
	string fileExtension = ".jpg";
	int frameIndex = 1;
	stringstream frameFile;
	//stringstream stringFrameIndex;

	VideoCapture capture;
    Mat cameraFrame, grayFrame, grayFrameOld;
    //Mat mask;
    Rect2d rect;
    //bool initializeFeatureTracking = false;
    bool pauseVideo = false;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    vector<Point2f> pointsOld;
    vector<Point2f> pointsNew;
    Size subPixWinSize(10,10);
    Size subPixZeroZone(-1,-1);
    Size winSizeForOpticalFlow(31,31);
    
    vector<uchar> status;
    vector<float> err;
    namedWindow( "KLT Tracker", 1 );

    capture.open(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);

    capture.read(cameraFrame);
    cameraFrame.copyTo(grayFrame);
    cvtColor(cameraFrame, grayFrame, COLOR_BGR2GRAY);

    while(1)
    {
    	if(!pauseVideo)
    	{
    		//**********Various lines below need to be commented/uncommented to get image frames from camera or from file input
	    	frameFile << filePath << setw(4) << setfill('0') << frameIndex << fileExtension; //build string file path to drone file
	    	//frameFile << filePath << frameIndex << fileExtension; //build string file path to thermal file
	    	frameIndex = (frameIndex + 1) % 451;//increment image frame number
	    	//frameIndex = (frameIndex + 1) % 4974;//increment image frame number
	    	frameFile.str("");
	    	capture.read(cameraFrame);
	    	//cameraFrame = imread(frameFile.str(), CV_LOAD_IMAGE_COLOR);
    		cvtColor(cameraFrame, grayFrame, COLOR_BGR2GRAY);

	        if(!pointsNew.empty())
	        {
	    		if(pointsOld.empty())
		        {
		            pointsOld = pointsNew;
		        }

		        if(grayFrameOld.empty())
		        {
		            grayFrame.copyTo(grayFrameOld);
		        }
	        	calcOpticalFlowPyrLK(	grayFrameOld, /*Old Frame*/
										grayFrame, /*New Frame*/
										pointsOld, /*Old Points Vector*/
										pointsNew, /*New Points Vector*/
										status, /*Status Vector - T/F if flow for corresponding corner has been found*/
										err, /*Error Vector - */
										winSizeForOpticalFlow, /*size of the search window at each pyramid level*/
										6, /*pyramid level number*/
										termcrit, 
										0, 
										0.0000001); /*minimum eigen value of 2x2 matrix. sets a standard (lower = more strict) for filtering out bad points.*/
	        }

	    }
        
        if(initializeFeatureTracking)
        {
        	cvtColor(cameraFrame, grayFrame, COLOR_BGR2GRAY);
        	imshow("KLT Tracker", cameraFrame);//refresh the image so we draw the roi in the most recent frame
        	pointsOld.clear();//we are grabbing new points so all old points will be irrelevant
        	grayFrame.copyTo(grayFrameOld);
        	rect = selectROI("KLT Tracker", cameraFrame);//roi selection box to grab the object of interest
        	Mat mask = Mat::zeros(grayFrame.size(), CV_8U);//declare matrix of same size as our frame
        	Mat roi(mask, rect);//declare matrix roi that will mask the region of interest
        	roi = Scalar(255, 255, 255);//roi should be "binary"
            goodFeaturesToTrack(	grayFrame, 
            						pointsNew, 
            						MAX_COUNT, /*max number of corners*/
            						0.35, /*corner quality*/ 
            						5, /*min distance between corners*/
            						mask, /*roi mask*/
            						3, /*block size*/
            						//3, /*unknown variable found in examples?*/
            						false, /*use Harris Detector*/
            						0.1); /*parameter for Harris Detector*/
            cornerSubPix(grayFrame, pointsNew, subPixWinSize, subPixZeroZone, termcrit);
            pointsOld = pointsNew;
            initializeFeatureTracking = false;
        }
        for(int i = 0; i < pointsNew.size(); i++)
		{
	    	circle( cameraFrame, pointsNew[i], 3, Scalar(0,255,255), -1, 8);
	    }
	    imshow("KLT Tracker", cameraFrame);

        char c = (char)waitKey(30);
        if(c == 27)
            break;
        switch(c)
        {
        case 'p': //pause
			pauseVideo = !pauseVideo;
            break;
        case 'c'://clear points
            pointsOld.clear();
            pointsNew.clear();
            break;
        case 's'://initialize feature tracking
        	initializeFeatureTracking = true;
            break;
        }
		swap(pointsNew, pointsOld);
		swap(grayFrameOld, grayFrame);
    }
}
