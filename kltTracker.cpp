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
const int MAX_COUNT = 50;

static void onMouse(int, int, int, int /*flags*/, void* /*param*/);

int main()
{
    //VideoCapture capture;
	string filePath = "bebop_images/frame";
	string fileExtension = ".jpg";
	int frameIndex = 0;
	stringstream frameFile;
	//stringstream stringFrameIndex;

    Mat cameraFrame, grayFrame, grayFrameOld;
    //Mat mask;
    Rect2d rect;
    //bool initializeFeatureTracking = false;
    bool pauseVideo = false;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    vector<Point2f> pointsOld;
    vector<Point2f> pointsNew;
    //Point2f centerOfFeatureTrackingArea;
    Size subPixWinSize(10,10);
    Size subPixZeroZone(-1,-1);
    Size winSizeForOpticalFlow(31,31);
    
    vector<uchar> status;
    vector<float> err;
    namedWindow( "KLT Tracker", 1 );
    setMouseCallback("KLT Tracker", onMouse, 0);

    //capture.open(0);
    //capture.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
    //capture.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);

    //capture.read(cameraFrame);
    //cameraFrame.copyTo(grayFrame);
    //cvtColor(cameraFrame, grayFrame, COLOR_BGR2GRAY);

    while(1)
    {
    	if(!pauseVideo)
    	{
	        //capture.read(cameraFrame);
	        //cvtColor(cameraFrame, grayFrame, COLOR_BGR2GRAY);
	    	frameFile << filePath << setw(4) << setfill('0') << frameIndex << fileExtension; //build string file path/name of the image frame
	    	//cout << frameFile.str() << endl;
	    	frameIndex = (frameIndex + 1) % 451;//increment image frame number
	    	cameraFrame = imread(frameFile.str(), CV_LOAD_IMAGE_COLOR);
	    	cvtColor(cameraFrame, grayFrame, COLOR_BGR2GRAY);
	    	frameFile.str("");

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
										5, /*pyramid level number*/
										termcrit, 
										0, 
										0.000001); /*minimum eigen value of 2x2 matrix. sets a standard (lower = more strict) for filtering out bad points.*/
	        }

	    }
        
        if(initializeFeatureTracking)
        {
        	cvtColor(cameraFrame, grayFrame, COLOR_BGR2GRAY);
        	imshow("KLT Tracker", cameraFrame);//refresh the image so we draw the roi in the most recent frame
        	pointsOld.clear();//we are grabbing new points so all old points will be irrelevant
        	//grayFrameOld.release();
        	grayFrame.copyTo(grayFrameOld);
        	rect = selectROI("KLT Tracker", cameraFrame);//roi selection box to grab the object of interest
        	Mat mask = Mat::zeros(grayFrame.size(), CV_8U);//declare matrix of same size as our frame
        	Mat roi(mask, rect);//declare matrix roi that will mask the region of interest
        	roi = Scalar(255, 255, 255);//roi should be "binary"
            goodFeaturesToTrack(	grayFrame, 
            						pointsNew, 
            						MAX_COUNT, /*max number of corners*/
            						0.2, /*corner quality*/ 
            						3, /*min distance between corners*/
            						mask, /*roi mask*/
            						3, /*block size*/
            						//3, /*unknown variable found in examples?*/
            						true, /*use Harris Detector*/
            						0.1); /*parameter for Harris Detector*/
            //pointsNew.push_back(centerOfFeatureTrackingArea);
            cornerSubPix(grayFrame, pointsNew, subPixWinSize, subPixZeroZone, termcrit);
            pointsOld = pointsNew;
            initializeFeatureTracking = false;
            //centerOfFeatureTrackingArea = Point2f(-1,-1);
        }
        for(int i = 0; i < pointsNew.size(); i++)
		{
	    	circle( cameraFrame, pointsNew[i], 3, Scalar(0,255,255), -1, 8);
	    }
	    imshow("KLT Tracker", cameraFrame);

        char c = (char)waitKey(60);
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
        	//pauseVideo = !pauseVideo;
        	initializeFeatureTracking = true;
            break;

        // case 'n':
        //     nightMode = !nightMode;
        //     break;
        }
		swap(pointsNew, pointsOld);
		swap(grayFrameOld, grayFrame);
    }
}

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
	if( event == EVENT_LBUTTONDOWN )
	{
   		//cout << "mouse click" << endl;
		//centerOfFeatureTrackingArea = Point2f((float)x, (float)y);
		//initializeFeatureTracking = true;
	}
}
