#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "opencv2/imgcodecs.hpp"

#define I_CIRCLE_RADIUS 100
#define CANNY_THRESHOLD 70
#define pi 3.1416

using namespace std;
using namespace cv;

// Global variables
//String hand_cascade_name = "C:\\haar\\handOpen.xml";
String hand_cascade_name = "C:\\haar\\1256617233-1-haarcascade_hand.xml";
//String hand_cascade_name = "C:\\haar\\palm.xml";

//String eyes_cascade_name = "C:\\haar\\haarcascade_eye.xml";
CascadeClassifier hand_cascade;
//CascadeClassifier eyes_cascade;
String window_name = "Capture - Hand detection";
// @function main 

int photocount = 0; //initialize image counter.
int fc = 0;
String imagename1, imagename2, imagename3, imagename4, imagename5, imagename6, imagename7;
int key;

Mat src; 
Mat src_gray;
int thresh = 230;
int max_thresh = 255;
RNG rng(12345);

bool in ;
bool mi ;
bool ri ;
bool li ;
String th ;
String sign ;


String inttostr(int input)
{
	stringstream ss;
	ss << input;
	return ss.str();
}

double dist(Point p1, Point p2)
{
	double d,m,n;
	m = (p1.x - p2.x)*(p1.x - p2.x);
	n = (p1.y - p2.y)*(p1.y - p2.y);
	d = sqrt(m + n);
	return d;
}

int findangle(Point cog1a, Point cogb, Point cofc)
{
	Point ab = { cogb.x - cog1a.x, cogb.y - cog1a.y };
	Point cb = { cogb.x - cofc.x, cogb.y - cofc.y };
	/*
	//POINTFLOAT ab = { b.x - a.x, b.y - a.y };
	//POINTFLOAT cb = { b.x - c.x, b.y - c.y };

	// dot product  
	float dot = (ab.x * cb.x + ab.y * cb.y);
	// length square of both vectors
	float abSqr = ab.x * ab.x + ab.y * ab.y;
	float cbSqr = cb.x * cb.x + cb.y * cb.y;

	// square of cosine of the needed angle    
	float cosSqr = dot * dot / abSqr / cbSqr;

	// this is a known trigonometric equality:
	// cos(alpha * 2) = [ cos(alpha) ]^2 * 2 - 1
	float cos2 = 2 * cosSqr - 1;

	// Here's the only invocation of the heavy function.
	// It's a good idea to check explicitly if cos2 is within [-1 .. 1] range

	const float pi = 3.141592f;

	float alpha2 =
		(cos2 <= -1) ? pi :
		(cos2 >= 1) ? 0 :
		acosf(cos2);

	float rslt = alpha2 / 2;

	float rs = rslt * 180. / pi;


	// Now revolve the ambiguities.
	// 1. If dot product of two vectors is negative - the angle is definitely
	// above 90 degrees. Still we have no information regarding the sign of the angle.

	// NOTE: This ambiguity is the consequence of our method: calculating the cosine
	// of the double angle. This allows us to get rid of calling sqrt.

	if (dot < 0)
		rs = 180 - rs;

	// 2. Determine the sign. For this we'll use the Determinant of two vectors.

	float det = (ab.x * cb.y - ab.y * cb.y);
	if (det < 0)
		rs = -rs;

	return (int)floor(rs + 0.5);
	*/
	float dot = ((ab.x * cb.x) + (ab.y * cb.y));
	float cross = ((ab.x * cb.y) - (ab.y * cb.x));
	float alpha = atan2(cross, dot);
	int angle = floor(alpha * 180. / pi + 0.5);
	//cout << endl;
	//cout << "Angle " << angle << endl;
	angle = angle*(-1);
	return angle;
}


void signDetect()
{
	if ((in == true) && (mi == false) && (ri == false) && (li == false) && (th == "close"))
	{
		sign = "One";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  1" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == false) && (li == false) && (th == "close"))
	{
		sign = "Two";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  2" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == true) && (li == false) && (th == "close"))
	{
		sign = "Three";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  3" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == true) && (li == true) && (th == "close"))
	{
		sign = "Four";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  4" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == true) && (li == true) && (th == "normal"))
	{
		sign = "Five";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  5" << endl;
	}
	else if ((in == false) && (mi == true) && (ri == false) && (li == false) && (th == "up"))
	{
		sign = "Six";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  6" << endl;
	}
	else if ((in == true) && (mi == false) && (ri == false) && (li == false) && (th == "normal"))
	{
		sign = "Seven";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  7" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == false) && (li == false) && (th == "normal"))
	{
		sign = "Eight";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  8" << endl;
	}
	else if ((in == false) && (mi == false) && (ri == false) && (li == false) && (th == "down"))
	{
		sign = "Nine";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  9" << endl;
	}
	else if ((in == false) && (mi == false) && (ri == false) && (li == false) && (th == "close"))
	{
		sign = "Zero";
		printf("%s %s %s %s %s ", in ? "false" : "true", mi ? "false" : "true", ri ? "false" : "true", li ? "false" : "true", th.c_str());
		cout << sign << "-------------------------------------  0" << endl;
	}
}


void cog_cof(Mat img)
{
	vector<Point> finger, fin;
	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	namedWindow("Binary", CV_WINDOW_AUTOSIZE);
	moveWindow("Binary", 570, 100);
	moveWindow("Original", 0, 100);
	int row = 3, col = 3;
	//Mat original = imread("F:/Thesis/thesis_project/ppp.jpg");
	// Define circle center
	// put the circle center at the center of image for test purposes
	// Scalar is the way to go CV_RGB is old format, openCV uses Scalar on BGR now
	Point circle_center((img.cols / 2)+10, (img.rows / 2)+25);
	// Calculate all points on that circle
	// Axes are half of the ellipse axes, a circle is a perfect ellipse, with radius = 2x half of the axes :)
	if (img.rows > 3 && img.cols > 3)
	{
		row = img.rows;
		col = img.cols;
		row = (row*2) / 5;
		col = (col*2) / 5;
	}
	Size axes(row, col);
	vector<Point> circle_points;
	ellipse2Poly(circle_center, axes, 0, 0, 360, 1, circle_points);

	//Make a grayscale copy
	//Mat gray(original.size(), CV_8UC1);
	//cvtColor(original, gray, CV_BGR2GRAY);

	//Find edges in grayscale image
	//Mat edges(original.size(), CV_8UC1);
	//Canny(gray, edges, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 3);

	// Only draw extra info right before visualisation
	//circle(original, circle_center, I_CIRCLE_RADIUS, Scalar(255, 0, 0), 2);

	// Lets make a edge map on a 3 channel color image for output purposes
	Mat img_color(img.rows, img.cols, CV_8UC3); //8UC3
	Mat inp[] = { img, img, img };
	int from_to[] = { 0, 0, 1, 1, 2, 2 };
	mixChannels(inp, 3, &img_color, 1, from_to, 3);
	//Iterate pixels in circle
	for (int i = 0; i < circle_points.size(); i++){
		Point current_point = circle_points[i];
		int value_at_that_location_in_edge_map = img.at<uchar>(circle_points[i].y, circle_points[i].x);
		//cout << value_at_that_location_in_edge_map << " ";
		if (value_at_that_location_in_edge_map == 0){
			circle(img_color, circle_points[i], 2, Scalar(0, 0, 255), -1); // filled circle at the position
			fin.push_back(circle_points[i]);
		}
	}
	circle(img_color,circle_center,5,Scalar(0,255,0),-1);
	//int l = 0;
	//add new code for finger
	for (int k = 1; k < fin.size(); k++)
	{
		double dis = dist(fin[k-1],fin[k]);
		if (dis>25)
		{
			finger.push_back(fin[k - 1]);
			finger.push_back(fin[k]);
		}
	}
	Point cog1;
	cog1.x = circle_center.x + row;
	cog1.y = circle_center.y;
	line(img_color, circle_center, cog1, Scalar(255, 0, 0), 2, 8, 0);

	in = false;
	mi = false;
	ri = false;
	li = false;
	th = "close";

	for (int f = 0; f < finger.size(); f++)
	{
		circle(img_color, finger[f], 4, Scalar(255, 255, 0), -1);
		//line(img_color, circle_center, finger[f], Scalar(255, 0, 0), 2, 8, 0);
		int angle = findangle(cog1, circle_center, finger[f]);
		if (angle > 0||(angle > -100 && angle < -80))
			line(img_color, circle_center, finger[f], Scalar(255, 0, 0), 2, 8, 0);
		//cout << endl;
		//cout << "Angle " << angle << endl;
		if (angle > 0 && angle <= 35)
		{
			th = "normal";
			//cout << "Thumb Normal" << endl;
		}
		else if (angle >= 55 && angle < 85)
		{
			in = true;
			//cout << "Index" << endl;
		}
		else if (angle >= 85 && angle < 115)
		{
			mi = true;
			th = "up";
			//cout << "Middle maybe thumbs up" << endl;
		}
		else if (angle >= 115 && angle < 140)
		{
			ri = true;
			//cout << "Ring" << endl;
		}
		else if (angle >= 140 && angle < 175)
		{
			li = true;
			//cout << "Little" << endl;
		}
		else if (angle > -105 && angle <= -80)
		{
			th = "down";
			//cout << "Thumbs Down" << endl;
		}
	}
	/*
	sign = "---";

	if ((in == true) && (mi == false) && (ri == false) && (li == false) && (th == "close"))
	{
		sign = "One";
		printf("\n%s %s %s %s %s ", in ? "true" : "false", mi ? "true" : "false", ri ? "true" : "false", li ? "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == false) && (li == false) && (th == "close"))
	{
		sign = "Two";
		printf("\n%s %s %s %s %s ", in ? "true" : "false", mi ? "true" : "false", ri ? "true" : "false", li ?  "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == true) && (li == false) && (th == "close"))
	{
		sign = "Three";
		printf("\n%s %s %s %s %s ", in ?  "true" : "false", mi ?  "true" : "false", ri ? "true" : "false", li ?  "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == true) && (li == true) && (th == "close"))
	{
		sign = "Four";
		printf("\n%s %s %s %s %s ", in ? "true" : "false", mi ?  "true" : "false", ri ?  "true" : "false", li ? "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == true) && (li == true) && (th == "normal"))
	{
		sign = "Five";
		printf("\n%s %s %s %s %s ", in ? "true" : "false", mi ? "true" : "false", ri ?  "true" : "false", li ?  "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}
	else if ((in == false) && (mi == true) && (ri == false) && (li == false) && (th == "up"))
	{
		sign = "Six";
		printf("\n%s %s %s %s %s ", in ? "true" : "false", mi ? "true" : "false", ri ?  "true" : "false", li ? "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}
	else if ((in == true) && (mi == false) && (ri == false) && (li == false) && (th == "normal"))
	{
		sign = "Seven";
		printf("\n%s %s %s %s %s ", in ?  "true" : "false", mi ?  "true" : "false", ri ?  "true" : "false", li ?  "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}
	else if ((in == true) && (mi == true) && (ri == false) && (li == false) && (th == "normal"))
	{
		sign = "Eight";
		printf("\n%s %s %s %s %s ", in ? "true" : "false", mi ? "true" : "false", ri ? "true" : "false", li ? "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}
	else if ((in == false) && (mi == false) && (ri == false) && (li == false) && (th == "down"))
	{
		sign = "Nine";
		printf("\n%s %s %s %s %s ", in ? "true" : "false", mi ? "true" : "false", ri ? "true" : "false", li ? "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}
	else if ((in == false) && (mi == false) && (ri == false) && (li == false) && (th == "close"))
	{
		sign = "Zero";
		printf("\n%s %s %s %s %s ", in ? "true" : "false", mi ? "true" : "false", ri ? "true" : "false", li ? "true" : "false", th.c_str());
		cout << sign << " is  Detected" << endl;
	}

	// Always add a waitKey(5) after an imshow so that the onpaint inner method can get executed without problems
	//signDetect();
	*/
	imshow("Binary", img_color);
	imshow("Original", img);
	//waitKey(5);
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

	hand_cascade.detectMultiScale(frame_gray, hand, 1.3, 2, 0 | CASCADE_SCALE_IMAGE, Size(64, 64)); //sir_chng
	Point close, close1;
	close.x = 0;
	close.y = 0;
	close1.x = 1;
	close1.y = 1;
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
		else
		{
			close.x = 0;
			close.y = 0;
			close1.x = 1;
			close1.y = 1;
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
	rectangle(frame, close, close1, Scalar(255, 0, 255), 5, 8, 0);

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
	Mat fra, fra2 ;
	//threshold(crop, fra, 127, 255, THRESH_BINARY); // somthing not right iqr
	//threshold(frame_gray, fra1, 127, 255, THRESH_BINARY_INV);
	//if ((fc % 5) == 0)
	//{
	//fra1=thresh_callback(0, 0 , fra2); //chng iqr fra1 er kaj baki ase cog contour
	Point ptc; // cog
	ptc.x = (close.x + close1.x) / 2 ;
	ptc.y = (close.y + close1.y) / 2;
	circle(frame, ptc , 4, Scalar(0, 255, 255), -1, 8, 0);
	circle(frame, ptc, I_CIRCLE_RADIUS, Scalar(255, 0, 0), 2);//new december
	//contour iqr new

	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	Mat dst(frame.rows, frame.cols, CV_8UC1, Scalar::all(0)); //8UC1
	//threshold(frame_gray, fra2, 25, 255, THRESH_BINARY); //Threshold the gray
	threshold(crop, fra2, 25, 255, THRESH_BINARY);

	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(fra2, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
	
	for (int i = 0; i< contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a>largest_area){
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}
	Scalar color(255, 255, 255);
	drawContours(dst, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
	Mat jan1;
	Mat morph1, morph2, morph3, morph4, erod, dilat;
	int erosion_size= 4;
	int dilation_size = 1;
	Mat element1 = getStructuringElement(MORPH_ELLIPSE,Size(2 * erosion_size + 1, 2 * erosion_size + 1),Point(erosion_size, erosion_size));
	/// Apply the erosion operation
	Mat element2 = getStructuringElement(MORPH_ELLIPSE,Size(2 * dilation_size + 1, 2 * dilation_size + 1),Point(dilation_size, dilation_size));
	/// Apply the dilation operation
	//imshow("contour", dst);
	dilate(dst, dilat, element2);
	erode(dilat, erod, element1);
	dilate(erod, dst, element2);
	//imshow("erosion", erod);
	//imshow("dilation", dilat);

	//int morph_size = 2;
	//Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	//morphologyEx(dst, morph1, MORPH_OPEN, element);
	//morphologyEx(dst, morph2, MORPH_CLOSE, element);
	//morphologyEx(dst, morph3, MORPH_TOPHAT, element);
	//morphologyEx(dst, morph4, MORPH_BLACKHAT, element);
	//imshow("1", morph1);
	//imshow("2", morph2);
	//imshow("3", morph3);
	//imshow("4", frame_gray);

	rectangle(frame, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);
	circle(dst, ptc, 4, Scalar(255, 255, 255), 10, 8, 0); //no need december
	
	imshow("primary", frame);
	//imshow("Gray", frame_gray);
	imshow("largest Contour Dilate-Erode-Dilate", dst);

	Rect newroi(0,0,myroi.width,myroi.height);
	Mat jan;
	dst.copyTo(jan);
	jan1 = jan(newroi);
	//imshow("Cropped", jan);
	/*
	photocount++;
	imagename1 = "F:/Thesis/test/pic" + inttostr(photocount) + ".jpg";
	imwrite(imagename1, jan1, compression_params);
	photocount++;
	imagename2 = "F:/Thesis/test/pic" + inttostr(photocount) + ".jpg";
	imwrite(imagename2, jan, compression_params);
	photocount++;
	imagename3 = "F:/Thesis/test/pic" + inttostr(photocount) + ".jpg";
	imwrite(imagename3, frame, compression_params);
	photocount++;
	imagename4 = "F:/Thesis/test/pic" + inttostr(photocount) + ".jpg";
	imwrite(imagename4, frame_gray, compression_params);
	*/
	////////////////// new code december /////////////////
	//Mat newimg = imread("F:/Thesis/thesis_project/pic82.jpg");
	//cog_cof(newimg);
	if(newroi.height>5 && newroi.width>5)
		cog_cof(jan1);

	/////////////////////////////////////////////////////////
}

int main(void)
{
	//cog_cof();
	VideoCapture capture;
	Mat frame;
	//freopen("out.txt", "w+", stdout);
	//-- 1. Load the cascades
	if (!hand_cascade.load(hand_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };
	//if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };
	//-- 2. Read the video stream
	capture.open(1);

	if (!capture.isOpened()) 
	{ 
		printf("--(!)Error opening video capture\n"); 
		return -1; 
	}
	
	time_t start, end;
	while (capture.read(frame))
	{
		time(&start);
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		//printf("\n \n \n true true true false up Three is Detected\n \n \n \n");
		//printf("\n \n true true false false down Two is Detected");
		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);
		//signDetect();
		cout << endl;
		time(&end);
		double sec = difftime(end, start);
		cout << "second " << sec << endl;
		int c = waitKey(10);
		if ((char)c == 27) { break; } // escape
	}

	return 0;
}
