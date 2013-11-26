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
const int TOTAL_SUBINTERVAL_MOA = 4;
float DEPTH_SCALE = 1; // decide the scaling factor with respect to MAX_RANGE
int KINECTS_DISPLACEMENT = 0; 
float MAX_RANGE;
float totalSubIntervalsMOA[TOTAL_SUBINTERVAL_MOA] = {0,0,0,0};
char* windMoA = "Activity Map";
KinectSensor kinects[NUM_SENSORS];


/*
activityMap(out): Updates the positions where the 3D points project.
actimityMap_back(out): With the background
featureSpace(out): Feature space of colour and height. Modelled over a plan view
am(in): 
p3D(in): List of foreground points for a particular kinect
nP(in): Number of foreground points.
points2D(in): List of 2D points of the image plane for a particular kinect
rgbMap(in): Map of rgb colours for a particular kinect
*/
void updateActivityMap(Mat& activityMap, Mat& activityMap_back, const ActivityMap_Utils* am, const XnPoint3D* p3D, const int nP, const XnPoint3D* points2D)
{
	for (int i = 0; i < nP; i++)
	{
		Point p2D = ActivityMap_Utils::findMoACoordinate(&p3D[i], MAX_RANGE);

		if (p2D.x != -1)
		{
			uchar* ptr = activityMap.ptr<uchar>(p2D.y);
			uchar* ptr_back = activityMap_back.ptr<uchar>(p2D.y);

			ptr[3*p2D.x] = ptr_back[3*p2D.x] = 0;
			ptr[3*p2D.x+1] = ptr_back[3*p2D.x+1] = 0;
			ptr[3*p2D.x+2] = ptr_back[3*p2D.x+2] = 0;
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

Rect calculateRect(KinectSensor* kOut, KinectSensor* k, bool left)
{
	int angle = 29;
	if (!left)
		angle = -angle;

	XnPoint3D edge3D[1];
	edge3D[0].X = 9700*tanf(angle*CV_PI/180);
	float range = sqrtf( powf(9700,2) + powf(edge3D[0].X,2) );
	edge3D[0].Y = 0;//range * tanf(21.5*CV_PI/180);
	edge3D[0].Z = 9700;
	
	kOut->transformArrayNoTilt(edge3D, 1);
	XnPoint3D* edge2D = k->arrayProject(edge3D, 1);
	int x = edge2D[0].X;
	delete []edge2D;
	Rect out;

	if (left)
		out = Rect(0,0, x, XN_VGA_Y_RES);
	else
		out = Rect(x, 0, XN_VGA_X_RES-x, XN_VGA_Y_RES);

	return out;
}

int fillArrayPointSel(const Mat& roi, XnPoint3D* points, int origX)
{
	int ttlRows = roi.rows;
	int ttlCols = roi.cols;

	ushort* d_data = (ushort*)roi.data;
	int d_step = roi.step/sizeof(ushort);

	int id = 0;
	for (int i = 0; i < ttlRows; i++)
	{
		ushort* ptr = d_data + i*d_step;
		for (int j = 0; j < ttlCols; j++)
		{
			int depth = ptr[j];
			if (depth != 0)
			{
				//int id = i*ttlCols + j;
				points[id].X = j+origX;
				points[id].Y = i;
				points[id++].Z = depth;
			}
		}
	}
	return id;
}

void fillArray(const Mat& roi, XnPoint3D* points, int origX)
{
	int ttlRows = roi.rows;
	int ttlCols = roi.cols;

	ushort* d_data = (ushort*)roi.data;
	int d_step = roi.step/sizeof(ushort);

	for (int i = 0; i < ttlRows; i++)
	{
		ushort* ptr = d_data + i*d_step;
		for (int j = 0; j < ttlCols; j++)
		{
			int id = i*ttlCols + j;
			points[id].X = j+origX;
			points[id].Y = i;
			points[id].Z = ptr[j];
		}
	}
}

void maskDepthImagePoint(Mat& dMat, const XnPoint3D* points3D, const XnPoint3D* points2D, int ttlP)
{
	int ttlRows = dMat.rows;
	int ttlCols = dMat.cols;
	ushort* d_data = (ushort*)dMat.data;
	int d_step = dMat.step/sizeof(ushort);

	for (int id = 0; id < ttlP; id++)
	{
		XnPoint3D p = points3D[id];
		if (p.X < XN_VGA_X_RES && p.X >= 0 && p.Y < XN_VGA_Y_RES && p.Y >= 0)
		{
			XnPoint3D p2d = points2D[id];
			ushort* ptr = d_data + (int)p2d.Y*d_step;
			ptr[(int)p2d.X] = 0;
		}
	}
}

void maskDepthImage(Mat& roi, const XnPoint3D* points)
{
	int ttlRows = roi.rows;
	int ttlCols = roi.cols;
	for (int i = 0; i < ttlRows; i++)
	{
		ushort* ptr = roi.ptr<ushort>(i);
		for (int j = 0; j < ttlCols; j++)
		{
			int id = i*ttlCols + j;
			XnPoint3D p = points[id];
			if (p.X < XN_VGA_X_RES && p.X >= 0 && p.Y < XN_VGA_Y_RES && p.Y >= 0)
			{
				ptr[j] = 0;
			}
		}
	}
}

void createROI(Mat& depthMat0, Mat& depthMat1, int x, bool left)
{
	KinectSensor* k;
	int step = -1;
	int step3D = 20;
	if (left)
		k = &kinects[0];
	else
	{
		k = &kinects[2];
		step = 1;
		step3D = -20;
	}

	XnPoint3D edge2D[480];
	XnPoint3D edge3D[480];

	for (int i = 40; i < 475; i++)
	{
		int depth = 0;
		int xId = x;
		while (depth == 0 && xId >= 0 && xId < XN_VGA_X_RES)
		{
			depth = (int)depthMat0.ptr<ushort>(i)[xId];
			xId += step;
		}
		if (depth == 0)
			cout << "hey" << endl;
		
		if (depth > 4000)
			xId +=step3D;
		edge2D[i].X = xId; edge2D[i].Y = i; edge2D[i].Z = depth;
	}
	k->arrayBackProject(edge2D, edge3D, 480);

	k->transformArrayNoTilt(edge3D, 480);
	XnPoint3D* p1 = kinects[1].arrayProject(edge3D, 480);
	for (int i = 0; i < 480; i++)
	{
		XnPoint3D p = p1[i];
		if (p.X >= 0 && p.X < XN_VGA_X_RES && p.Y >= 0 && p.Y < XN_VGA_Y_RES)
		{
			int xInit = 0;
			int width = p.X;
			if (!left)
			{
				width = XN_VGA_X_RES - p.X;
				xInit = p.X;
			}

			Rect r = Rect(xInit, p.Y, width, 1);
			Mat roi = depthMat1(r);
			roi = Mat::zeros(roi.size(), CV_16UC1);
		}
	}
	delete [] p1;

}

void maskOutOverlappingEdge(Mat& depthMat0, Mat& depthMat1, Mat& depthMat2)
{
	XnPoint3D edge2D[2];
	XnPoint3D edge3D[2];
	createROI(depthMat0, depthMat1, 615, true);
	createROI(depthMat2, depthMat1, 20, false);


}

void maskOutOverlappingPointSel(Mat& depthMat, Rect rLeft, Rect rRight)
{
	int tpLeft = rLeft.width*rLeft.height;
	int tpRight = rRight.width*rRight.height;
	//Go through al the points in the roi and create an array of XnDepth3D
	Mat leftRoi = depthMat(rLeft);
	Mat rightRoi = depthMat(rRight);
	//TODO: may be parameters of the function
	XnPoint3D* leftSide2D = new XnPoint3D[tpLeft];
	XnPoint3D* rightSide2D = new XnPoint3D[tpRight];

	int ttlLeft = fillArrayPointSel(leftRoi, leftSide2D, rLeft.x);
	int ttlRight = fillArrayPointSel(rightRoi, rightSide2D, rRight.x);

	//Back project to the 3D space
	XnPoint3D* leftSide3D = new XnPoint3D[ttlLeft];
	XnPoint3D* rightSide3D = new XnPoint3D[ttlRight];
	kinects[REF_CAM].arrayBackProject(leftSide2D, leftSide3D, ttlLeft);
	kinects[REF_CAM].arrayBackProject(rightSide2D, rightSide3D, ttlRight);

	// transform into the second CS (make sure it is calibrated the other way around)
//	XnPoint3D* leftCamP = new XnPoint3D[tpLeft];
//	XnPoint3D* rightCamP = new XnPoint3D[tpRight];
	kinects[REF_CAM].transformArrayNoTilt_rev(leftSide3D, ttlLeft, 0);
	kinects[REF_CAM].transformArrayNoTilt_rev(rightSide3D, ttlRight, 2);

	//Project into the image plane
	XnPoint3D* leftCamP = kinects[0].arrayProject(leftSide3D, ttlLeft);
	XnPoint3D* rightCamP = kinects[2].arrayProject(rightSide3D, ttlRight);

	maskDepthImagePoint(depthMat, leftCamP, leftSide2D, ttlLeft);
	maskDepthImagePoint(depthMat, rightCamP, rightSide2D, ttlRight);

	//Free memory
	delete []leftSide2D;
	delete []rightSide2D;
	delete []leftSide3D;
	delete [] rightSide3D;
	delete [] leftCamP;
	delete [] rightCamP;

}

void maskOutOverlappingSelective(Mat& depthMat, Rect rLeft, Rect rRight)
{
	XnPoint3D edge2D[1];
	XnPoint3D edge3D[1];
	edge2D[0].X = 615; edge2D[0].Y = 240; edge2D[0].Z = 4500;
	kinects[1].arrayBackProject(edge2D, edge3D, 1);
	kinects[1].transformArrayNoTilt_rev(edge3D, 1, 2);
	XnPoint3D* edge2Dp = kinects[2].arrayProject(edge3D, 1);

	int tpLeft = rLeft.width*rLeft.height;
	int tpRight = rRight.width*rRight.height;
	//Go through al the points in the roi and create an array of XnDepth3D
	Mat leftRoi = depthMat(rLeft);
	Mat rightRoi = depthMat(rRight);
	//TODO: may be parameters of the function
	XnPoint3D* leftSide2D = new XnPoint3D[tpLeft];
	XnPoint3D* rightSide2D = new XnPoint3D[tpRight];

	fillArray(leftRoi, leftSide2D, rLeft.x);
	fillArray(rightRoi, rightSide2D, rRight.x);

	//Back project to the 3D space
	XnPoint3D* leftSide3D = new XnPoint3D[tpLeft];
	XnPoint3D* rightSide3D = new XnPoint3D[tpRight];
	kinects[REF_CAM].arrayBackProject(leftSide2D, leftSide3D, tpLeft);
	kinects[REF_CAM].arrayBackProject(rightSide2D, rightSide3D, tpRight);

	// transform into the second CS (make sure it is calibrated the other way around)
//	XnPoint3D* leftCamP = new XnPoint3D[tpLeft];
//	XnPoint3D* rightCamP = new XnPoint3D[tpRight];
	kinects[REF_CAM].transformArrayNoTilt_rev(leftSide3D, tpLeft, 0);
	kinects[REF_CAM].transformArrayNoTilt_rev(rightSide3D, tpRight, 2);

	//Project into the image plane
	XnPoint3D* leftCamP = kinects[0].arrayProject(leftSide3D, tpLeft);
	XnPoint3D* rightCamP = kinects[2].arrayProject(rightSide3D, tpRight);

	maskDepthImage(leftRoi, leftCamP);
	maskDepthImage(rightRoi, rightCamP);

	//Free memory
	delete []leftSide2D;
	delete []rightSide2D;
	delete []leftSide3D;
	delete [] rightSide3D;
	delete [] leftCamP;
	delete [] rightCamP;

}

void convert16to8(const Mat* src, Mat& out)
{
	double max = 0;
	minMaxIdx(*src, NULL, &max);
	if (max != 0)
	{
		src->convertTo(out, CV_8UC1, 255/max);
			
		subtract(cv::Scalar::all(255),out, out);

	}
	else
	{
		Utils::initMat1u(out, 255);
	}
}


void maskOutOverlapping(Mat& depthMat, Rect rLeft, Rect rRight)
{
	Mat leftRoi = depthMat(rLeft);
	Mat rightRoi = depthMat(rRight);
	leftRoi = Mat::zeros(leftRoi.size(), CV_16U);
	rightRoi = Mat::zeros(rightRoi.size(), CV_16U);


}

void updateDepthImage(Mat& dImg, Mat& mask)
{
	for (int i = 0; i < dImg.rows; i++)
	{
		uchar* ptrD = dImg.ptr<uchar>(i);
		ushort* ptrM = mask.ptr<ushort>(i);
		for (int j = 0; j < dImg.cols; j++)
		{
			if (ptrM[j] == 0)
			{
				ptrD[j*3] = 255;
				ptrD[j*3+1] = 255;
				ptrD[j*3+2] = 255;
			}
				
		}
	}

}

const int TOTAL_INTERVALS = 2;
const int ROI = 0;
const int TOT_ID = 1;
char* titles[TOTAL_INTERVALS] = {"ROI MASK", " TOTAL"};
float totalIntervals[TOTAL_INTERVALS] = {0,0};

ofstream outDebugFile("d:/Debug.txt");
/*
Arg 1: 0:Video; 1:live
Arg 2: 0:No Record; 1:Record
*/
int main(int argc, char* argv[])
{
	
	bool saved = false;
	int fromVideo = 0;
	int recordOut = 1;
	int tilt = -40;

	char* paths[3];
	paths[0] = "d:/Emilio/Tracking/DataSet/kinect0_calib.oni";
	paths[1] = "d:/Emilio/Tracking/DataSet/kinect1_calib.oni";
	paths[2] = "d:/Emilio/Tracking/DataSet/kinect2_calib.oni";


	ActivityMap_Utils actMapCreator(DEPTH_SCALE, NUM_SENSORS);

	
	const XnDepthPixel* depthMaps[NUM_SENSORS];
	const XnRGB24Pixel* rgbMaps[NUM_SENSORS];

	for (int i = 0; i < NUM_SENSORS; i++)
	{
		if (fromVideo == 0)
			kinects[i].initDevice(i, REF_CAM, true, paths[i]);
		else
			kinects[i].initDevice(i, REF_CAM, true);

		kinects[i].startDevice();
		kinects[i].tilt(tilt);
	}


	KINECTS_DISPLACEMENT = max(abs(kinects[0].translation(0)), abs(kinects[2].translation(0))); //MAXIMUM TRANSLATION IN THE HORIZONTAL AXIS
	MAX_RANGE = ActivityMap_Utils::MAX_Z_TRANS + KINECTS_DISPLACEMENT; 
	

	//namedWindow(windMoA);
	Mat *activityMap, *activityMap_Back;
	Mat whiteBack, colorMap;
	Mat background = Mat(actMapCreator.getResolution(), CV_8UC3);

	//flags
	bool bShouldStop = false;
	bool trans = true;
	bool bgComplete = true;
	bool deleteBG = true;

	Mat depthImages[NUM_SENSORS];
	Mat rgbImages[NUM_SENSORS];
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
		rgbImages[i] = Mat(XN_VGA_Y_RES, XN_VGA_X_RES, CV_8UC3);
		depthMat[i] = Mat(XN_VGA_Y_RES, XN_VGA_X_RES, CV_16U);
		pointsFore2D[i] = new XnPoint3D[MAX_FORGROUND_POINTS];
		numberOfForegroundPoints[i] = 0;
	}


	//calculate mask out regions in middle kinect
	Rect rLeft, rRight;
	rLeft = calculateRect(&kinects[0], &kinects[1], true);
	rRight = calculateRect(&kinects[2], &kinects[1], false);

	//DEBUG
	XnPoint3D edge2D[2];
	XnPoint3D edge3D[2];
	edge2D[0].X = 640; edge2D[0].Y = 480; edge2D[0].Z = 9700;
	edge2D[1].X = 0; edge2D[1].Y = 480; edge2D[1].Z = 9700;

	kinects[0].arrayBackProject(&edge2D[0], &edge3D[0], 1);
	kinects[2].arrayBackProject(&edge2D[1], &edge3D[1], 1);

	kinects[0].transformArrayNoTilt(&edge3D[0], 1);
	kinects[2].transformArrayNoTilt(&edge3D[1], 1);

	XnPoint3D* pr0 = kinects[1].arrayProject(&edge3D[0], 1);
	XnPoint3D* pr1 = kinects[1].arrayProject(&edge3D[1], 1);
	//END DEBUG




	bool first = true;

	Mat* outMoA;
	Mat outMoAScaled;
	int waitTime = 1;

	
	VideoWriter w;
	if (recordOut == 1)
	{
		//VideoWriter	w("d:/Emilio/Tracking/DataSet/MoA2.mpg",CV_FOURCC('P','I','M','1'), 20.0, actMapCreator.getResolution(), true);
		//Size sz = actMapCreator.getResolution();
	//	Size sz = Size(181, 500);
		w.open("d:/MoA_Detection.mpg",CV_FOURCC('P','I','M','1'), 20.0, actMapCreator.getResolution(), true);
	}	

	clock_t startTime;
	clock_t startTotalTime = clock();

	int nPoints = 0;
	int frames = 0;
	while (!bShouldStop && frames <= 530)
	{		
		cout << "Frames: " << frames << endl;
		for (int i = 0; i < NUM_SENSORS; i++)
			kinects[i].waitAndUpdate();
		
		for (int i = 0; i < NUM_SENSORS;  i++)
		{
			depthMaps[i] = kinects[i].getDepthMap();
			rgbMaps[i] = kinects[i].getRGBMap();
			//new part
			kinects[i].getDepthImage(depthImages[i]);
			kinects[i].getRGBImage(rgbImages[i]);
			
			//Creates a matrxi with depth values (ushort)
			createDepthMatrix(depthMaps[i], depthMat[i]);

			if (i == REF_CAM)
			{
				startTime = clock();
				//Mask out the complete ROI
				//maskOutOverlapping(depthMat[i], rLeft, rRight);
				//Selective Mask out
				//maskOutOverlappingSelective(depthMat[i], rLeft, rRight);
				//Selective Mask Out Edge
				//maskOutOverlappingEdge(depthMat[0], depthMat[1], depthMat[2]);
				//Selective Points Mask out 
				maskOutOverlappingPointSel(depthMat[i], rLeft, rRight);

				updateDepthImage(depthImages[i], depthMat[i]);
				totalIntervals[ROI] += clock() - startTime; //time debugging
				line(depthImages[i], Point(pr0->X, 0), Point(pr0->X, 480), Scalar(0,0,255));
				line(depthImages[i], Point(pr1->X, 0), Point(pr1->X, 480), Scalar(0,0,255));
			}


			//to create a mask for the noise (depth img is threhold)
			cvtColor(depthImages[i],grey,CV_RGB2GRAY);
			masks[i] = grey > 250; //mask that identifies the noise (1)
		}
		

		nPoints = 0;
		if (bgComplete && trans)// && frames > 20) //Trans must be true
		{
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
			{
				numberOfForegroundPoints[i] = subtractors[i].subtraction(pointsFore2D[i], &(depthMat[i]), &(masks[i]));
				nPoints += numberOfForegroundPoints[i];
			}
			if (nPoints > 0)
			{
				int ttlPnts = 0;

				for (int i = 0; i < NUM_SENSORS; i++)
				{
					points3D[i] = new XnPoint3D[numberOfForegroundPoints[i]];
					kinects[i].arrayBackProject(pointsFore2D[i], points3D[i], numberOfForegroundPoints[i]);

					kinects[i].transformArray(points3D[i], numberOfForegroundPoints[i]);
										
					updateActivityMap(*activityMap, *activityMap_Back, &actMapCreator, points3D[i], numberOfForegroundPoints[i], pointsFore2D[i]);
					delete []points3D[i];
				}
		
				Mat *tmp = activityMap;
				if (!deleteBG)
					tmp = activityMap_Back;
				
			}
			outMoA = activityMap;
			if (!deleteBG)
					outMoA = activityMap_Back;

			if (recordOut == 1)
				w << *activityMap;
			
		}
		else
		{			
			actMapCreator.createActivityMap(kinects, depthMaps, rgbMaps, trans, background, frames, MAX_RANGE, totalSubIntervalsMOA); 
			
			outMoA = &background;
	/*		if (recordOut == 1 && trans)
					w << background;*/
		}

		if (DEPTH_SCALE < 1)
		{			
			resize(*outMoA, outMoAScaled, Size(outMoA->cols/DEPTH_SCALE, outMoA->rows/DEPTH_SCALE), 0,0, INTER_LINEAR);
			outMoA = &outMoAScaled;
		}
		imshow(windMoA, *outMoA);

		line(depthImages[0], Point(615, 0), Point(615, 480), Scalar(255,0,0));
		line(depthImages[2], Point(20, 0), Point(20, 480), Scalar(255,0,0));

		imshow("depth 0", depthImages[0]);
		imshow("depth 1", depthImages[1]);
		imshow("depth 2", depthImages[2]);

		int c = waitKey(waitTime);
		bShouldStop = (c == 27);
		if (c == 13)
			waitTime = 0;
		
		frames++;
	}

	totalIntervals[TOT_ID] = clock() - startTotalTime;
	//BUILD REPORT
	outDebugFile << "EXECUTION TIME REPORT" << endl;
	for (int i = 0; i < TOTAL_INTERVALS-1; i++)
	{
		float time_p = totalIntervals[i]*100/totalIntervals[TOT_ID];
		outDebugFile << titles[i] << ": " << time_p << " %" << endl;
	}
		
	double fps = frames/(double(totalIntervals[TOT_ID])/(double(CLOCKS_PER_SEC)));
	outDebugFile << "Total frames processed: " << frames << ". fps: " << fps << endl;
	
	for (int i = 0; i < NUM_SENSORS; i++)
	{
		kinects[i].stopDevice();
  		kinects[i].shutDown();
	}
	return 0;

}