#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

// Global variables
String hand_cascade_name = "C:\\haar\\1256617233-1-haarcascade_hand.xml";
//String eyes_cascade_name = "C:\\haar\\haarcascade_eye.xml";
CascadeClassifier hand_cascade;
//CascadeClassifier eyes_cascade;
String window_name = "Capture - Hand detection";
// @function main 

int photocount = 0; //initialize image counter.
int fc = 0;
String imagename;
int key;

Mat src; Mat src_gray;
int thresh = 230;
int max_thresh = 255;
RNG rng(12345);


String inttostr(int input)
{
	stringstream ss;
	ss << input;
	return ss.str();
}

Mat thresh_callback(int, void*, Mat fra)
{
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Point result = 0;
	int largest = 0;
	int largest_index = 0;
	/// Detect edges using canny
	Canny(fra, canny_output, thresh, thresh * 2, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<Moments> mu(contours.size());
	Moments mm;
	
	for (size_t i = 0; i < contours.size(); i++)
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest){
			largest = a;
			mm = moments(contours[i], false);
			largest_index = i;                //Store the index of largest contour
			//bounding_rect = boundingRect(contours[i]);
		}
	}
	vector<Point2f> mc(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mm.m10 / mm.m00, mm.m01 / mm.m00);
	}
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (size_t i = 0; i< contours.size(); i++)
	{
		//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, (int)i, Scalar(255,0,0), 2, 8, hierarchy, 0, Point());
		circle(drawing, mc[i] , 4, Scalar(0, 255, 255) , -1, 8, 0);
	}

	//convex hull

	/*vector<vector<Point> >hull(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
	}
	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		drawContours(drawing, hull, i, color, 1, 8, hierarchy, 0, Point());

	}*/

	/*Moments mo = moments(drawing);
	result = Point(mo.m10 / mo.m00, mo.m01 / mo.m00);*/

	/// Show in a window
	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//imshow("Contours", drawing);
	return drawing;
}

void detectAndDisplay(Mat frame)
{
	vector<int> compression_params;
	//vector that stores the compression parameters of the image
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	//specify the compression technique
	compression_params.push_back(100); //specify the compression quality

	fc++;
	std::vector<Rect> hand;
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	// apply pre-processing functions
	IplImage* frame2 = cvCloneImage(&(IplImage)frame);
	Mat img;

	hand_cascade.detectMultiScale(frame_gray, hand, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	Point close, close1;
	close.x = 0;
	close.y = 0;
	close1.x = 0;
	close1.y = 0;
	for (size_t i = 0; i < hand.size(); i++)
	{
		//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		Point pt1, pt2;
		pt1.x = hand[i].x;
		pt1.y = hand[i].y;
		pt2.x = hand[i].width;
		pt2.y = hand[i].height;
		if (close1.x < pt2.x && close1.y < pt2.y)
		{
			close1.x = pt2.x;
			close1.y = pt2.y;
			close.x = pt1.x;
			close.y = pt1.y;
		}
		//rectangle(frame, pt1, pt2, Scalar(255 , 0 , 255),5,8,0);
		//Mat faceROI = frame_gray(hand[i]);

#if 0
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
#endif
	}
	//rectangle(frame_gray, close, close1, Scalar(255, 0, 255), 5, 8, 0);
	Rect myroi(close.x, close.y, close1.x, close1.y);
	vector <Point> contour;
	contour.push_back(close);
	contour.push_back(Point2f(close.x, close1.y));
	contour.push_back(Point2f(close1.x, close.y));
	contour.push_back(close1);
	double area0 = contourArea(contour);
	//originalImage(faceRect).copyTo(croppedImage);
	//frame_gray(myroi).copyTo(img);
	//img = frame_gray(myroi);
	Mat crop = frame_gray(myroi);
	Mat fra, fra1;
	threshold(frame_gray, fra, 127, 255, THRESH_BINARY);
	//threshold(frame_gray, fra1, 127, 255, THRESH_BINARY_INV);
	//if ((fc % 5) == 0)
	//{
	fra1=thresh_callback(0, 0 , fra);
	photocount++;
	//imagename = "F:/Thesis/thesis_project/haar_detection/haar_detection/pic" + inttostr(photocount) + ".jpg";
	//imwrite(imagename, frame, compression_params);
	//}
	imshow(window_name, frame);
}

int main(void)
{
	VideoCapture capture;
	Mat frame;
	//-- 1. Load the cascades
	if (!hand_cascade.load(hand_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };
	//if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };
	//-- 2. Read the video stream
	capture.open(1);
	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }
	time_t start, end;
	while (capture.read(frame))
	{
		time(&start);
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);
		time(&end);
		double sec = difftime(end, start);
		cout << "second" << sec;
		int c = waitKey(10);
		if ((char)c == 27) { break; } // escape
	}

	return 0;
}

//second e 5 ta chobi shudhu rectangle er/
//bhalo web cam
//manipulation
//ki jinish banabo
//gray scale try kore dekhbo

/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 240;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void*);

// @function main 
int main(int argc, char** argv)
{
	/// Load source image and convert it to gray
	src = imread("pic7.jpg", 1);

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	/// Create Window
	char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

	//createTrackbar(" Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);

	waitKey(0);
	return(0);
}

//@function thresh_callback 
void thresh_callback(int, void*)
{
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	//convex hull

	vector<vector<Point> >hull(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
	}
	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		drawContours(drawing, hull, i,color, 1, 8, hierarchy , 0, Point());
		
	}

	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}
*/