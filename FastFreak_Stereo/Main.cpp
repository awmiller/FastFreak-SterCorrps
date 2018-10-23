/**
* @file Threshold.cpp
* @brief Sample code that shows how to use the diverse threshold options offered by OpenCV
* @author OpenCV team
*/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv2\core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2\xfeatures2d.hpp"
#include "opencv2\imgproc.hpp";

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

// DISPARITY_MAX is the maximum disparity allowed for any given feature
// this was selected based on the observation that a great number of outliers
// were produced in the initial implementation
#define DISPARITY_MAX 300
// SIFT_SCALEFACTOR is the fitted best scale factor applied the the computed
// SIFT disparity map. This value was found in Matlab via unconstrained 
// multivariable minimization using "[params,minimum]=fminsearch(@func,initial_params)
#define SIFT_SCALEFACTOR 3.94
// FFK_SCALEFACTOR is the best fit scale factor for FASK-FREAK disparity using
// the same method as for SIFT_SCALEFACTOR
#define FFK_SCALEFACTOR 3.74
// DISPARITY_ERR_THRSH is \delta_d as defined in the Middlebury taxonomy paper
// it is the threshold applied for deciding bad pixels
#define DISPARITY_ERR_THRSH 5
// DO_ALL_WAITKEYS is a flag that will enable all UI pause functions, this is set
// to false for data collection and faster iteration, but is reccommended to be 
// set true for presentation, so that console messages align with applications events
#define DO_ALL_WAITKEYS false
// SHOW_DEBUG_IMAGES is a flag that will disable most image generation all together,
// and my association, the wait key given for them. This flag should be true for presentation
#define SHOW_DEBUG_IMAGES false

// The following defines were added to list the best case metrics as found with Matlab,
// these are the values reported in the written report for this project
#define MATLAB_SIFT_RMS 15.0373
#define MATLAB_FFK_RMS 31.7032
#define MATLAB_SFT_BP 0.0273
#define MATLAB_FFK_BP 0.0174

// matchData is a typedef-struct that appends useful data next to a match object
// this allows me to aggregate useful information in a way that entries are aligned
// for loop iteration in post-processing.
typedef struct {
	DMatch match; Point2d trainedPt; int gtruth; float dy, dx;
} matchData;

/// ***helper functions*** ///
///<summary>
/// use this function in main, wrapped in any sort of error handling. This is where the real application takes place
///</summary>
void do_main();

///<summary>
/// This function takes two images, a matcher type defined in DescriptorMatcher, and two sets of keypoints for matching, 
/// returning the last parameter as a vector of matches found
/// <param @="im0">the first image is taken as the training image</param>
/// <param @="im1">the second image is taken as the querry or matching image</param>
/// <param @="matcher_t">this is a type that must be selected form DescriptorMatcher::</param>
/// <param @="keypoints1">these keypoints correspond to im0</param>
/// <param @="keypoints2">keypoints for im1</param>
/// <param @="matches">resulting matches found</param>
///</summary>
void ComputeSiftMatches(Mat &im0, Mat &im1, int matcher_t, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches);

///<summary>
/// This function takes two images, a matcher type defined in DescriptorMatcher, and two sets of keypoints for matching, 
/// returning the last parameter as a vector of matches found. slightly different processing is taken for FASK-FREAK.
/// <param @="im0">the first image is taken as the training image</param>
/// <param @="im1">the second image is taken as the querry or matching image</param>
/// <param @="matcher_t">this is a type that must be selected form DescriptorMatcher::</param>
/// <param @="keypoints1">these keypoints correspond to im0</param>
/// <param @="keypoints2">keypoints for im1</param>
/// <param @="matches">resulting matches found</param>
///</summary>
void ComputeFastFreakMatches(Mat &im0, Mat &im1, int matcher_t, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches);

///<summary>
/// This function does the disparity map calculation as defined by Middlebury framework as abs(x1-x2)
/// if SHOW_DEBUG_IMAGES=true then imno is the window number to draw to. Uses the DISPARITY_MAX macro
/// to filter outliers
/// <param @="im0">the first image is taken as the training image</param>
/// <param @="im1">the second image is taken as the querry or matching image</param>
/// <param @="matches">this is a vecotor of matches that has been computed already</param>
/// <param @="filt_m">this is a vecotor of matches that will be passed through from filtering @matches</param>
/// <param @="keypoints1">these keypoints correspond to the training image, im0</param>
/// <param @="keypoints2">keypoints for im1, the query image</param>
/// <param @="imno">the window number to use for debuging images</param>
///</summary>
void ComputeAndFilterDesparityMap(Mat &im0, Mat &im1, vector<DMatch> &matches, vector<DMatch> &filt_m, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, int imno);

///<summary>
/// This function filters a set of matches based on a maximum delta-y in thier coordinates, 
/// implements a vertical rejection filter specifically for stereo matching problems.
/// <param @="matches">this is a vecotor of matches that has been computed already</param>
/// <param @="filt_m">this is a vecotor of matches that will be passed through from filtering @matches</param>
/// <param @="keypoints1">these keypoints correspond to the training image</param>
/// <param @="keypoints2">keypoints for the query image</param>
/// <param @="threshold">the maximum allowed delta-y in pixels</param>
///</summary>
void filter_dy(vector<DMatch> &matches, vector<DMatch> &filt_m, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, float threshold);

///<summary>
/// This utility function creates a vector of structs under the typdef @matchData, which I use for passing around elaborated match data specific to my application
/// <param @="truth_img">this is the disparity map image, used to add "truth point" tags to each row</param>
/// <param @="trainKeys">these keypoints correspond to the training image</param>
/// <param @="compKeys">keypoints for the query image</param>
/// <param @="matches">a set of matches calculated from the given keys</param>
/// <param @="tmatches">this will be populated with @matchData objects</param>
///</summary>
void GetMatchData(Mat &truth_img, vector<KeyPoint> &trainKeys, vector<KeyPoint> &compKeys, vector<DMatch> &matches, vector<matchData> &tmatches);

///<summary>
/// This utility function creates a log file for pos-processing, its reccommended to use GetMatchData(...) first to populate @vdat
/// <param @="filename">the file name to use</param>
/// <param @="matches">a set of matches</param>d already</param>
/// <param @="trainKeys">these keypoints correspond to the training image</param>
/// <param @="querKeys">keypoints for the query image</param>
/// <param @="vdat">an aggregation of matchData corresponding to the set of matches passed in</param>
///</summary>
void print_matches(string filename, vector<DMatch> matches, vector<KeyPoint> trainKeys, vector<KeyPoint> querKeys, vector<matchData> vdat);

///<summary>
/// This is supposed to implement the RMS disparity error, but I'm still trying to corellate it with my Matlab results
/// <param @="tmatches">the final set of matches to use</param>
/// <param @="fit_param">an experimentally found best fit scale factor to account for scale error I may have made</param>
/// <returns> the RMS disparity error </returns>
///</summary>
float MiddleburyRMS(vector<matchData> &tmatches, float fit_param);

///<summary>
/// This is supposed to implement the Pecent Bad Pixels metric, but I'm still trying to corellate it with my Matlab results
/// <param @="tmatches">the final set of matches to use</param>
/// <param @="eval_bad_thresh">an experimentally found allowable threshold for detecting bad pixel, defined in the Taxonomy paper</param>
/// <param @="fit_param">an experimentally found best fit scale factor to account for scale error I may have made</param>
/// <returns> percentage of bad pixels in the image</returns>
///</summary>
float MiddleburyBadPixels(vector<matchData> &tmatches, float eval_bad_thresh, float fit_param);

///<summary>
/// utility function abstracts away details of making an image appear
/// <param @="window">any index between 0-9, use the same index twice to replace first image</param>
/// <param @="image">an image you want to display</param>
///</summary>
void window_disp(int window, Mat image);

/// collection of image titles for use with @window_disp()
string windows[10] = {
	"Middlebury cones2", // window 0
	"Middlebury cones6", // window 1
	"Keypoint Detailed Image", // window 2
	"Keypoint Detailed Transformed Image", // window 3
	"Ground Truth", // window 4
	"SIFT Match Map", // window 5
	"FAST Keypoints", // window 6
	"FAST-FREAK Match Map", // window 7
	"Sift Disparity Map", // window 8
	"FAST-FREAK Disparity Map", // window 9
	// ... add more windows as needed
};

/// useful streams for making files
ostringstream filename1, filename2;

/**
* @function do_waitKey allows application to conditionally ignore the wait key requests
*/
void do_waitKey(bool should_wait) {
	if (should_wait) waitKey();
}

/**
* @function main wrapper around do_main to handle errors and report them
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
		ofstream log;
		log.open("errors.txt", ios::out);
		log << "Oh no! An exception occurred:" << endl << e.what() << endl << endl;
		log.close();
	}

	// allow operator to view cout until key is pressed
	cout << endl << endl << "Ending Program, press any key to close...";
	waitKey();

	return 0;
}

/**
* @function do_main wrapper does all the heavy lifting
*/
void do_main() {

	// optionally initialize all immediately
	/*for (int i = 0; i < 10; i++) {
		namedWindow(windows[i], WINDOW_AUTOSIZE);
	}*/

	// Load image
	Mat im0 = imread("../inputs/cones2.png", IMREAD_GRAYSCALE);
	Mat im1 = imread("../inputs/cones6.png", IMREAD_GRAYSCALE);
	Mat gt = imread("../inputs/cones_disp2.png", IMREAD_GRAYSCALE);

	// Display incoming Image
	window_disp(0, im0);
	cout << "Hello!" << endl << "Here is image 2 from the Middlebury cones data set" << endl << endl;
	//do_waitKey(DO_ALL_WAITKEYS);

	// display stereo counterpart
	window_disp(1, im1);
	cout << "Here is image 6 from the cones data set." << endl << endl;
	//do_waitKey(DO_ALL_WAITKEYS);

	// display ground truth
	window_disp(4, gt);
	cout << "Here is ground truth for image 2 from the cones data set." << endl << "Press any key to Continue... " << endl << endl;
	do_waitKey(DO_ALL_WAITKEYS);

	// detect SIFT matches first
	vector<DMatch> sift_m;
	vector<KeyPoint> sift_key1, sift_key2;
	cout << "Here are SIFT matches." << endl << "Press any key to Continue... " << endl << endl;
	ComputeSiftMatches(im0, im1, DescriptorMatcher::BRUTEFORCE, sift_key1, sift_key2, sift_m);

	// do vertical rejection first because brute force will match on both axes
	vector<DMatch> sift_mf;
	filter_dy(sift_m, sift_mf, sift_key1, sift_key2, 0.5);

	// now detect FAST-FREAK matches
	vector<DMatch> ffk_m;
	vector<KeyPoint> ffk_key1, ffk_key2;
	cout << "Here are FAST-FREAK matches." << endl << "Press any key to Continue... " << endl << endl;
	// OpenCV community reccommended BRUTEFORCE_HAMMING for binary string descriptors
	ComputeFastFreakMatches(im0, im1, DescriptorMatcher::BRUTEFORCE_HAMMING, ffk_key1, ffk_key2, ffk_m);

	// do vertical rejection for FAST-FREAK
	vector<DMatch> ffk_mf;
	filter_dy(ffk_m, ffk_mf, ffk_key1, ffk_key2, 0.5);

	// create a disparity map for each, where no match for a point=0
	// also returns a vector<DMatch> of filtered points with best match
	// for a given point to remove redundancy. is uniqueness filtering redundant? BF matches seems to do this already...
	vector<DMatch> sift_mf2, ffk_mf2;
	cout << "Now I will compute unqiue matches for each pixel by shortest distance for SIFT." << endl << "Press any key to Continue... " << endl << endl;
	ComputeAndFilterDesparityMap(im0, im1, sift_mf, sift_mf2, sift_key1, sift_key2, 8);
	cout << "Now I will compute unqiue matches for each pixel by shortest distance for FAST-FREAK." << endl << "Press any key to Continue... " << endl << endl;
	ComputeAndFilterDesparityMap(im0, im1, ffk_mf, ffk_mf2, ffk_key1, ffk_key2, 9);

	// draw final matches
	Mat sfm_img, ffm_img;
	drawMatches(im1, sift_key2, im0, sift_key1, sift_mf2, sfm_img);
	drawMatches(im1, ffk_key2, im0, ffk_key1, ffk_mf2, ffm_img);
	window_disp(7, ffm_img);
	window_disp(5, sfm_img);
	do_waitKey(DO_ALL_WAITKEYS);

	// compare each match to the ground-truth at each related point
	vector<matchData> sft_truth, ffk_truth;
	GetMatchData(gt, sift_key1, sift_key2, sift_mf2, sft_truth);
	GetMatchData(gt, ffk_key1, ffk_key2, ffk_mf2, ffk_truth);

	// NOT WORKING... no time to fix, results still found with MATLAB for time being.
	// 10/23 2:28pm -- "bad pixels" matches matlab result but RMS is still order mag larger
	// print results to screen
	float sft_rms, ffk_rms, sft_bp, ffk_bp;
	sft_rms = MiddleburyRMS(sft_truth, SIFT_SCALEFACTOR);
	ffk_rms = MiddleburyRMS(ffk_truth, FFK_SCALEFACTOR);
	sft_bp = MiddleburyBadPixels(sft_truth, DISPARITY_ERR_THRSH, SIFT_SCALEFACTOR);
	ffk_bp = MiddleburyBadPixels(ffk_truth, DISPARITY_ERR_THRSH, FFK_SCALEFACTOR);
	// set fixed width, precision and print results
	std::cout.precision(3);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	cout << "C++ Results:\n"
		<< "------------------------------------------" << endl
		<< "|       |     SIFT     |    FASK-FREAK    " << endl
		<< "|  RMS  |    " << sft_rms << "    |  " << ffk_rms  << endl
		<< "|Bad px |     " << sft_bp << "    |   " << ffk_bp << endl << endl;

	std::cout.precision(3);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	cout << "MATLAB Results:\n"
		<< "------------------------------------------" << endl
		<< "|       |     SIFT     |    FASK-FREAK    " << endl
		<< "|  RMS  |    " << MATLAB_SIFT_RMS << "    |  " << MATLAB_FFK_RMS << endl
		<< "|Bad px |     " << MATLAB_SFT_BP << "    |   " << MATLAB_FFK_BP << endl << endl;

	cout << "Saving match data for Matlab analysis." << endl;// << "Press any key to Continue... " << endl << endl;
	print_matches("../outputs/SiftMatches.csv", sift_mf, sift_key1, sift_key2, sft_truth);
	print_matches("../outputs/FastFreakMatches.csv", ffk_mf, ffk_key1, ffk_key2, ffk_truth);

	//do_waitKey(DO_ALL_WAITKEYS);
}

void window_disp(int window, Mat image) {
	namedWindow(windows[window], WINDOW_NORMAL); // declare it
	imshow(windows[window], image); // use it

}

void ComputeSiftMatches(Mat &im0, Mat &im1, int matcher_t, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches)
{
	// store keypoint images
	Mat im0k, im1k;

	// Keypoint analysis
	Ptr<SIFT> detector = SIFT::create();

	// do detection
	detector->detect(im0, keypoints1); // left image
	detector->detect(im1, keypoints2); // right image


	if (SHOW_DEBUG_IMAGES)
	{
		// --  Draw keypoints
		Mat img_keypoints_1;
		Mat img_keypoints_2;
		drawKeypoints(im0, keypoints1, im0k, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		drawKeypoints(im1, keypoints2, im1k, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		// --  Show detected (drawn) keypoints
		window_disp(2, im0k);
		window_disp(3, im1k);
	}

	// get Descriptors
	Mat desc1, desc2;
	detector->compute(im0, keypoints1, desc1);
	detector->compute(im1, keypoints2, desc2);

	// starting with a brute force matcher for simplicity
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_t);
	// vector<DMatch> matches;
	matcher->match(desc2, desc1, matches);

	if (SHOW_DEBUG_IMAGES)
	{
		// visualize matches
		Mat im_matches;
		// order must match the descriptor order in match(), since training img is second, draw it second
		drawMatches(im1, keypoints2, im0, keypoints1, matches, im_matches);
		window_disp(5, im_matches);
		do_waitKey(DO_ALL_WAITKEYS);
	}
}

void ComputeFastFreakMatches(Mat &im0, Mat &im1, int matcher_t, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches)
{
	Mat desc1, desc2;
	Mat im_matches;

	// FAST keypoint detection
	Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();
	fast->detect(im0, keypoints1);
	fast->detect(im1, keypoints2); // right image

	// FREAK descriptors
	Ptr<FREAK> freak = FREAK::create();
	freak->compute(im0, keypoints1, desc1);
	freak->compute(im1, keypoints2, desc2);

	// use a hamming matcher as per OpenCV community reccommendation
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_t);
	matcher->match(desc2, desc1, matches);

	if (SHOW_DEBUG_IMAGES)
	{
		drawMatches(im1, keypoints2, im0, keypoints1, matches, im_matches);
		window_disp(7, im_matches);
		do_waitKey(DO_ALL_WAITKEYS);
	}
}

void ComputeAndFilterDesparityMap(Mat &im0, Mat &im1, vector<DMatch> &matches, vector<DMatch> &filt_m, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, int imno)
{
	// initialze desparity map, primarily for visualization
	Mat desp = Mat::ones(im0.rows, im0.cols, CV_32FC1) * -1; // -1 garauntees detection of unused pixel

	// go over each pixel and scale it to the smalled distance 
	// of a match whose trained keypoint is at this pixel location
	// for visualization, nearest neighbor is taken
	vector<DMatch>::iterator mIter = matches.begin();
	for (; mIter != matches.end(); mIter++) {
		DMatch cur = (*mIter); // this match
		KeyPoint tk = keypoints1[cur.trainIdx]; // trained keypoint
		KeyPoint mk = keypoints2[cur.queryIdx]; // querry kp

		// stereo disparity defined in taxonomy paper
		float disp = abs(mk.pt.x - tk.pt.x);

		// filter outliers
		if (disp > DISPARITY_MAX) continue;

		// if this is the first match at the trianed coordinates, 
		if ((desp.at<float>(tk.pt) < 0)) {
			desp.at<float>(tk.pt) = disp;
			filt_m.push_back(cur);
		}
		// need to go back and remove an existing candidate if this is not the first match here
		else if (disp < desp.at<float>(tk.pt)) {
			// remove
			for (vector<DMatch>::iterator it2 = filt_m.begin(); it2 != filt_m.end(); it2++) {
				DMatch check = (*it2);// dereference current match
				KeyPoint ck = keypoints1[check.trainIdx]; // the trained keypoit at @check, used to get its coords
				if (ck.pt.x == tk.pt.x && ck.pt.y == ck.pt.y) {
					filt_m.erase(it2); // remove it
					break; // since i will never store two matches at one point, it is safe to break here
				}
			}
			// now add new match
			filt_m.push_back(cur);
			// and color the map here
			desp.at<float>(tk.pt) = disp;
		}
	}

	if (SHOW_DEBUG_IMAGES)
	{
		// scale reletive
		normalize(desp, desp, 0, 1, NORM_MINMAX);
		window_disp(imno, desp);
		do_waitKey(DO_ALL_WAITKEYS);
	}
}

void filter_dy(vector<DMatch> &matches, vector<DMatch> &filt_m, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, float threshold)
{
	// iterate over all the matches
	for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++) {
		DMatch cur = (*it); // dereference
		KeyPoint key1 = keypoints1[cur.trainIdx]; // trained keypoint
		KeyPoint key2 = keypoints2[cur.queryIdx]; // querry keypoint

		// calculate delta
		float dy = abs(key1.pt.y - key2.pt.y);

		// only add it to output if its less than threshold
		if (dy < threshold) {
			filt_m.push_back(cur);
		}
	}
}

void print_matches(string filename, vector<DMatch> matches, vector<KeyPoint> trainKeys, vector<KeyPoint> querKeys, vector<matchData> vdat) {
	// open a file
	ofstream file;
	file.open(filename, ios::out);

	// print header`
	file << "Distance, Trained X, Trained Y, Trained angle, Trained Response, Querry X, Querry Y, Querry angle, Querry Response, dx, dy, Ground Truth" << endl;

	// for each line, format a astring with data including a new line character
	for (int i = 0; i < vdat.size(); i++) {
		DMatch cur = vdat[i].match;
		float gtruth = vdat[i].gtruth;
		KeyPoint tKey = trainKeys[cur.trainIdx];
		KeyPoint qKey = querKeys[cur.queryIdx];
		// reference:		 dist , tx  , ty  , ta  , tr  , qx  , qy  , qa  , qr  , dx  , dy  ,truth
		string lineformat = "%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f\n";
		char buf[300];
		snprintf(buf, 300, lineformat.c_str(), cur.distance, tKey.pt.x, tKey.pt.y, tKey.angle, 
			tKey.response, qKey.pt.x, qKey.pt.y, qKey.angle, qKey.response, vdat[i].dx, vdat[i].dy, gtruth);
		file << buf;
	}

	file.close();
}

void GetMatchData(Mat &truth_img, vector<KeyPoint> &trainKeys, vector<KeyPoint> &compKeys, vector<DMatch> &matches, vector<matchData> &tmatches)
{
	// iterate over all matches, basically creating "tagged" matches with the @matchData struct
	for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++) {
		matchData m; // create tagged match
		DMatch cur = (*it); // dereference
		m.match = cur; // assign it
		m.trainedPt = trainKeys[cur.trainIdx].pt; // store trained point, I use this as the match coordinates
		m.gtruth = truth_img.at<uint8_t>(m.trainedPt); // store ground-truth at this point
		m.dx = trainKeys[cur.trainIdx].pt.x - compKeys[cur.queryIdx].pt.x; // calculate delta-x
		m.dy = trainKeys[cur.trainIdx].pt.y - compKeys[cur.queryIdx].pt.y; // calculate delta-y
		tmatches.push_back(m); // push it back.
	}
}

float MiddleburyRMS(vector<matchData> &tmatches, float fit_param)
{
	double RMS = 0;// initialize
	// this is the summation operator
	for (int i = 0; i < tmatches.size(); i++) {
		matchData cur = tmatches[i]; // get local for ease
		float delta = (abs(cur.dx) - cur.gtruth)*fit_param; // this the disp error, added fit_param for scaling
		float sq = pow(delta, 2); // this is the Square part of RMS
		RMS += sq; // this is the sumation operation
	}
	RMS = RMS / tmatches.size(); //  divide by n-points, this is the Mean part of RMS
	RMS = pow(RMS, 0.5); // this is the Root part of RMS
	return RMS;
}

float MiddleburyBadPixels(vector<matchData> &tmatches, float eval_bad_thresh, float fit_param)
{
	double B = 0; // init
	// sum over all elements
	for (int i = 0; i < tmatches.size(); i++) {
		matchData cur = tmatches[i];
		float delta = (abs(cur.dx) - cur.gtruth)*fit_param; // calculate error
		if (delta > DISPARITY_ERR_THRSH) // only sum if we pass threshold
			B += 1.0;
	}
	B = B / tmatches.size(); // normalize to get into percentage
	return B;
}
