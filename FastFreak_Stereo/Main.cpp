/**
* @file Threshold.cpp
* @brief Sample code that shows how to use the diverse threshold options offered by OpenCV
* @author OpenCV team
*/

#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2\world.hpp"
#include "opencv2\xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

/// helper functions
void do_main();

/**
* @function main
*/
int main(int argc, char** argv)
{
	try
	{
		do_main();
	}
	catch (const std::exception& e)
	{
		cout << "Oh no! An exception occurred:" << endl << e.what() << endl << endl;
	}

	// allow operator to view cout until key is pressed
	cout << endl << endl << "Ending Program, press any key to close...";
	waitKey();

	return 0;
}

ostringstream filename1, filename2;
// Mat img[4];
string windows[10] = {
	"Base Image", // window 0
	"Affine Transformed Image", // window 1
	"Keypoint Detailed Image", // window 2
	"Keypoint Detailed Transformed Image", // window 3
	"Ground Truth", // window 4
	"Descriptor Map", // window 5
	// "w5", // window 5 
	"w6", // window 6
	"w7", // window 7
	"w8", // window 8
	"w9" // window 9
};

void do_main() {

	/*for (int i = 0; i < 10; i++) {
		namedWindow(windows[i], WINDOW_AUTOSIZE);
	}*/

	// Load image
	Mat im0 = imread("im0.png", IMREAD_GRAYSCALE);
	Mat im1 = imread("im1.png", IMREAD_GRAYSCALE);
	Mat fast, ffreak, trans, save = im0;
	Mat im0k = im0, im1k = im1;

	// Display incoming Image
	namedWindow(windows[0], WINDOW_NORMAL);
	imshow(windows[0], im0);
	cout << "Hello!" << endl << "Here is a nice picture of a Bike..." << endl << "Press any key to continue... " << endl << endl;
	waitKey();

	// display stereo counterpart
	namedWindow(windows[1], WINDOW_NORMAL);
	imshow(windows[1], im1);
	cout << "Here is annother perspective" << endl << "Press any key to Continue... " << endl << endl;
	waitKey();

	// Keypoint analysis
	Ptr<SIFT> detector = SIFT::create();
	std::vector<KeyPoint> keypoints1, keypoints2;

	// do detection
	detector->detect(im0, keypoints1); // left image
	detector->detect(im1, keypoints2); // right image
										  
	// --  Draw keypoints
	Mat   img_keypoints_1;
	Mat   img_keypoints_2;
	drawKeypoints(im0, keypoints1, im0k,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
	drawKeypoints(im1, keypoints2, im1k,Scalar::all(-1),DrawMatchesFlags::DEFAULT);

	// --  Show detected (drawn) keypoints
	imshow(windows[2],im0k);
	imshow(windows[3],im1k);

	waitKey();
	
	//// sobel, lets try 2nd order... get a usefull map by basically making it a laplacian...
	//Mat soBird;
	//Sobel(img, soBird, img.depth(), 1, 1, 5,3);
	//imshow("Sobel Bird", soBird);
	//imwrite("soBird.jpg", soBird);

	//cout << "Here is a Sobel bird!" << endl << "Press any key to Continue... " << endl << endl;
	//waitKey();

	//// canny, lets try 1st order...
	//Mat cannyBird;
	//Canny(img, cannyBird, 50, 100);
	//imshow("Canny Bird", cannyBird);
	//imwrite("cannyBird.jpg", cannyBird);

	//cout << "Here is a Canny bird!" << endl << "Press any key to Continue... " << endl << endl;
	//waitKey();

	//cout << "Now here are the distance transforms of each..." << endl;

	//// invert the values
	//laBird = UINT8_MAX - laBird;
	//// convert so that we dont trip the assert in distanceTransform
	//laBird.convertTo(laBird, CV_8UC1);
	//distanceTransform(laBird, laBird, DIST_L1, 3);
	//normalize(laBird, laBird, 0, 1, NORM_MINMAX);
	//// disp
	//imshow("Laplacian Bird Distance", laBird);
	//imwrite("dTlaBird.jpg", laBird*255);
	//waitKey();

	//// invert
	//soBird = UINT8_MAX - soBird;
	//// convert
	//soBird.convertTo(soBird, CV_8UC1);
	//distanceTransform(soBird, soBird, DIST_L1, 3);
	//normalize(soBird, soBird, 0, 1, NORM_MINMAX);
	//// disp
	//imshow("Sobel Bird Distance", soBird);
	//imwrite("dTsoBird.jpg", soBird*255);
	//waitKey();

	//cannyBird = UINT8_MAX - cannyBird;
	//distanceTransform(cannyBird, cannyBird, DIST_L1, 3);
	//normalize(cannyBird, cannyBird, 0, 1, NORM_MINMAX);
	//imshow("Canny Bird Distance", cannyBird);
	//imwrite("dTcannyBird.jpg", cannyBird*255);
	//waitKey();
}
