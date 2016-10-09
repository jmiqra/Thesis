#include <iostream>
#include <string.h>
#include <conio.h>

#include<stdio.h>
#include<math.h>
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<opencv2\objdetect\objdetect.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<vector>

using namespace cv;
using namespace std;

CvHaarClassifierCascade *cascade;
CvMemStorage            *storage;

void detect(IplImage *img);

int main(int argc, char** argv)
{
	CvCapture *capture;
	IplImage  *frame;
	int       key;
	char      *filename = "C:\\haar\\1256617233-1-haarcascade_hand.xml"; //put the name of your classifier here

	cascade = (CvHaarClassifierCascade*)cvLoad(filename, 0, 0, 0);
	storage = cvCreateMemStorage(0);
	capture = cvCaptureFromCAM(0);

	assert(cascade && storage && capture);

	cvNamedWindow("video", 1);

	while (1) {
		frame = cvQueryFrame(capture);

		detect(frame);

		key = cvWaitKey(50);
	}

	cvReleaseImage(&frame);
	cvReleaseCapture(&capture);
	cvDestroyWindow("video");
	cvReleaseHaarClassifierCascade(&cascade);
	cvReleaseMemStorage(&storage);

	return 0;
}

void detect(IplImage *img)
{
	int i;

	CvSeq *object = cvHaarDetectObjects(
		img,
		cascade,
		storage,
		1.5, //-------------------SCALE FACTOR
		2,//------------------MIN NEIGHBOURS
		1,//----------------------
		// CV_HAAR_DO_CANNY_PRUNING,
		cvSize(30, 30), // ------MINSIZE
		cvSize(640, 480));//---------MAXSIZE

	for (i = 0; i < (object ? object->total : 0); i++)
	{
		CvRect *r = (CvRect*)cvGetSeqElem(object, i);
		cvRectangle(img,
			cvPoint(r->x, r->y),
			cvPoint(r->x + r->width, r->y + r->height),
			CV_RGB(255, 0, 0), 2, 8, 0);

		//printf("%d,%d\nnumber =%d\n",r->x,r->y,object->total);


	}

	cvShowImage("video", img);
}

