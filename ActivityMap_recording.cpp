#include "BackgroundDepthSubtraction.h"
#include <ActivityMap_Utils.h>
#include "KinectSensor.h"
#include "Plane.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <list>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <ctype.h>
#include "XnCppWrapper.h"


const int NUM_SENSORS = 3;

void updateActivityMap(Mat& activityMap, Mat& activityMap_back, const ActivityMap_Utils* am, const XnPoint3D* p3D, const int nP, const XnPoint3D* points2D, const XnRGB24Pixel* rgbMap)
{
	Mat hightMap = Mat::Mat(activityMap.size(), CV_32F);
	Utils::initMatf(hightMap, -5000);
	for (int i = 0; i < nP; i++)
	{
		int xCoor = am->findCoordinate(p3D[i].X, MIN_X, MAX_X, am->xStep);
		int yC = am->findCoordinate(p3D[i].Z, MIN_Z, MAX_Z, am->depthStep);
		int yCoor = (YRes-1) - yC; //flip around X axis.


		XnRGB24Pixel color = rgbMap[(int)points2D[i].Y*XN_VGA_X_RES+(int)points2D[i].X];

		uchar* ptr = activityMap.ptr<uchar>(yCoor);
		uchar* ptr_back = activityMap_back.ptr<uchar>(yCoor);
		float* ptrH = hightMap.ptr<float>(yCoor);

		if (ptrH[xCoor] < (float)p3D[i].Y)
		{
			ptrH[xCoor] = (float)p3D[i].Y;

			ptr[3*xCoor] = color.nBlue;
			ptr[3*xCoor+1] = color.nGreen;
			ptr[3*xCoor+2] = color.nRed;

			ptr_back[3*xCoor] = color.nBlue;
			ptr_back[3*xCoor+1] = color.nGreen;
			ptr_back[3*xCoor+2] = color.nRed;
		}
	}
}

void createDepthMatrix(const XnDepthPixel* dMap, Mat& depthMat)
{
	for (int i = 0; i < XN_VGA_Y_RES; i++)
	{
		ushort* ptr = depthMat.ptr<ushort>(i);
		for (int j = 0; j < XN_VGA_X_RES; j++)
		{
			ptr[j] = dMap[i*XN_VGA_X_RES+j];
		}
	}

}

int main()
{
	char* paths[3];
	paths[0] = "d:/Emilio/Tracking/DataSet/s_2pers_occl/kinect0_calib.oni";
	paths[1] = "d:/Emilio/Tracking/DataSet/s_2pers_occl/kinect1_calib.oni";
	paths[2] = "d:/Emilio/Tracking/DataSet/s_2pers_occl/kinect2_calib.oni";

	ActivityMap_Utils actMapCreator(NUM_SENSORS);

	KinectSensor kinects[NUM_SENSORS];
	const XnDepthPixel* depthMaps[NUM_SENSORS];
	const XnRGB24Pixel* rgbMaps[NUM_SENSORS];

	for (int i = 0; i < NUM_SENSORS; i++)
	{
//		kinects[i].initDevice(i, REF_CAM, true, paths[i]);
		kinects[i].initDevice(i, REF_CAM, true);
		kinects[i].startDevice();
		kinects[i].tilt(-10);
	}

	namedWindow("Activity Map");
	Mat *activityMap, *activityMap_Back;
	Mat background, whiteBack, colorMap;

	//flags
	bool bShouldStop = false;
	bool trans = false;
	bool bgComplete = false;
	bool deleteBG = false;

	Mat depthImages[NUM_SENSORS];
	Mat depthMat[NUM_SENSORS];
	Mat masks[NUM_SENSORS];
	Mat grey;

	BackgroundDepthSubtraction subtractors[NUM_SENSORS];
	int numberOfForegroundPoints[NUM_SENSORS];

	XnPoint3D* pointsFore2D [NUM_SENSORS];
	XnPoint3D* points3D[NUM_SENSORS];

	for (int i = 0; i < NUM_SENSORS; i++)
	{
		depthImages[i] = Mat(XN_VGA_Y_RES, XN_VGA_X_RES, CV_8UC3);
		depthMat[i] = Mat(XN_VGA_Y_RES, XN_VGA_X_RES, CV_16U);
		pointsFore2D[i] = new XnPoint3D[MAX_FORGROUND_POINTS];
		numberOfForegroundPoints[i] = 0;
	}

	Point centres[10];
	double angles[10];


//	VideoWriter w ("d:/Emilio/Tracking/DataSet/s_2pers_occl/MoA_2pers_occl.avi", CV_FOURCC('M','J','P','G'), 20.0, actMapCreator.getResolution(), true);

	bool first = true;
	int cont = 0;
	clock_t total = 0;
	while (!bShouldStop)
	{
		for (int i = 0; i < NUM_SENSORS; i++)
			kinects[i].waitAndUpdate();

		for (int i = 0; i < NUM_SENSORS;  i++)
		{
			depthMaps[i] = kinects[i].getDepthMap();
			rgbMaps[i] = kinects[i].getRGBMap();
			//new part
			kinects[i].getDepthImage(depthImages[i]);
			//Creates a matrxi with depth values (ushort)
			createDepthMatrix(depthMaps[i], depthMat[i]);

			//to create a mask for the noise (depth img is threhold)
			cvtColor(depthImages[i],grey,CV_RGB2GRAY);
			masks[i] = grey > 250;
		}
				
		if (bgComplete && trans) //Trans must be true
		{
			cont++;
			if (first)
			{
				whiteBack = Mat::Mat(actMapCreator.getResolution(), CV_8UC3);
				activityMap = new Mat(actMapCreator.getResolution(), CV_8UC3);
				activityMap_Back = new Mat(actMapCreator.getResolution(), CV_8UC3);
				Utils::initMat3u(whiteBack, 255);
				first = false;
			}
			whiteBack.copyTo(*activityMap);
			background.copyTo(*activityMap_Back);

			for (int i = 0; i < NUM_SENSORS; i++)
				numberOfForegroundPoints[i] = subtractors[i].subtraction(pointsFore2D[i], &(depthMat[i]), &(masks[i]));
			
			for (int i = 0; i < NUM_SENSORS; i++)
			{
				points3D[i] = kinects[i].arrayBackProject(pointsFore2D[i], numberOfForegroundPoints[i]);
				if (kinects[i].getIdCam() != kinects[i].getIdRefCam())
						kinects[i].transformArray(points3D[i], numberOfForegroundPoints[i]);

				updateActivityMap(*activityMap, *activityMap_Back, &actMapCreator, points3D[i], numberOfForegroundPoints[i], pointsFore2D[i], rgbMaps[i]);
			}
			
			if (deleteBG)
				imshow("Activity Map", *activityMap);
			else
				imshow("Activity Map", *activityMap_Back);
			
//			w << *activityMap;
		}
		else
		{
			background = *(actMapCreator.createActivityMap(kinects, depthMaps, rgbMaps, trans));
			imshow("Activity Map", background);
		}
		

	
		int c = waitKey(1);
		switch (c)
		{
		case 27: //esc
			{
				bShouldStop = true;
				break;
			}

		case 99: //c
			{
				bgComplete = true;
				break;
			}
		case 116: //t
			{
				trans = !trans;
				break;		
			}
		case 13: //enter
			{
				deleteBG = !deleteBG;
				break;
			}
		}
	}

	for (int i = 0; i < NUM_SENSORS; i++)
	{
		kinects[i].stopDevice();
  		kinects[i].shutDown();
	}
	return 0;
}