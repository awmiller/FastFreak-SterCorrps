/**
* @file Threshold.cpp
* @brief Sample code that shows how to use the diverse threshold options offered by OpenCV
* @author OpenCV team
*/

#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

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

void do_main() {
	ostringstream filename1, filename2;
	Mat img;
	namedWindow("Bird", WINDOW_GUI_EXPANDED);
	namedWindow("Laplacian Bird", WINDOW_GUI_EXPANDED);
	namedWindow("Sobel Bird", WINDOW_GUI_EXPANDED);
	namedWindow("Canny Bird", WINDOW_GUI_EXPANDED);

	namedWindow("Laplacian Bird Distance", WINDOW_GUI_EXPANDED);
	namedWindow("Sobel Bird Distance", WINDOW_GUI_EXPANDED);
	namedWindow("Canny Bird Distance", WINDOW_GUI_EXPANDED);

	// Load image
	img = imread("Bird.jpg", 0);
	Mat save = img;

	// must get frame into correct data type
	//Mat float_bird;
	//img.convertTo(float_bird, CV_32F, 1.0/255.0, 0.0);
	// normalize(float_frame, float_frame, 1, NORM_MINMAX);

	// Display incoming Image
	imshow("Bird", img);
	cout << "Hello!" << endl << "Here is a nice picture of a Bird..." << endl << "Press any key to continue... " << endl << endl;
	waitKey();

	// laplacian
	Mat laBird;
	// using aparture of 3 to get something out... any less just saturates because it produces no zeros...
	Laplacian(img, laBird, img.depth(), 5);
	imshow("Laplacian Bird", laBird);
	imwrite("laBird.jpg", laBird);

	cout << "Here is a Laplacian bird!" << endl << "Press any key to Continue... " << endl << endl;
	waitKey();

	// sobel, lets try 2nd order... get a usefull map by basically making it a laplacian...
	Mat soBird;
	Sobel(img, soBird, img.depth(), 1, 1, 5,3);
	imshow("Sobel Bird", soBird);
	imwrite("soBird.jpg", soBird);

	cout << "Here is a Sobel bird!" << endl << "Press any key to Continue... " << endl << endl;
	waitKey();

	// canny, lets try 1st order...
	Mat cannyBird;
	Canny(img, cannyBird, 50, 100);
	imshow("Canny Bird", cannyBird);
	imwrite("cannyBird.jpg", cannyBird);

	cout << "Here is a Canny bird!" << endl << "Press any key to Continue... " << endl << endl;
	waitKey();

	cout << "Now here are the distance transforms of each..." << endl;

	// invert the values
	laBird = UINT8_MAX - laBird;
	// convert so that we dont trip the assert in distanceTransform
	laBird.convertTo(laBird, CV_8UC1);
	distanceTransform(laBird, laBird, DIST_L1, 3);
	normalize(laBird, laBird, 0, 1, NORM_MINMAX);
	// disp
	imshow("Laplacian Bird Distance", laBird);
	imwrite("dTlaBird.jpg", laBird*255);
	waitKey();

	// invert
	soBird = UINT8_MAX - soBird;
	// convert
	soBird.convertTo(soBird, CV_8UC1);
	distanceTransform(soBird, soBird, DIST_L1, 3);
	normalize(soBird, soBird, 0, 1, NORM_MINMAX);
	// disp
	imshow("Sobel Bird Distance", soBird);
	imwrite("dTsoBird.jpg", soBird*255);
	waitKey();

	cannyBird = UINT8_MAX - cannyBird;
	distanceTransform(cannyBird, cannyBird, DIST_L1, 3);
	normalize(cannyBird, cannyBird, 0, 1, NORM_MINMAX);
	imshow("Canny Bird Distance", cannyBird);
	imwrite("dTcannyBird.jpg", cannyBird*255);
	waitKey();
}
