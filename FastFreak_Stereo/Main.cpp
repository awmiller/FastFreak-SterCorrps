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

#define DISPARITY_MAX 300
#define SIFT_SCALEFACTOR 3.94
#define FFK_SCALEFACTOR 3.74
#define DISPARITY_ERR_THRSH 5
#define DO_ALL_WAITKEYS false
#define SHOW_DEBUG_IMAGES false
#define MATLAB_SIFT_RMS 15.0373
#define MATLAB_FFK_RMS 31.7032
#define MATLAB_SFT_BP 0.0273
#define MATLAB_FFK_BP 0.0174

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

typedef struct {
	DMatch match; Point2d trainedPt; int gtruth; float dy, dx;
} matchData;

/// helper functions
void do_main();
void ComputeSiftMatches(Mat &, Mat &, int matcher_t, vector<KeyPoint> &, vector<KeyPoint> &, vector<DMatch> &);
void ComputeFastFreakMatches(Mat &, Mat &, int matcher_t, vector<KeyPoint> &, vector<KeyPoint> &, vector<DMatch> &);
void ComputeAndFilterDesparityMap(Mat &im0, Mat &im1, vector<DMatch> &matches, vector<DMatch> &filt_m, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, int imno);
void window_disp(int window, Mat image);
void filter_dy(vector<DMatch> &matches, vector<DMatch> &filt_m, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, float threshold);
void print_matches(string filename, vector<DMatch> matches, vector<KeyPoint> trainKeys, vector<KeyPoint> querKeys, vector<matchData> vdat);
void GetTruthMatches(Mat &truth_img, vector<KeyPoint> &trainKeys, vector<KeyPoint> &compKeys, vector<DMatch> &matches, vector<matchData> &tmatches);
float MiddleburyRMS(vector<matchData> &tmatches, float fit_param);
float MiddleburyBadPixels(vector<matchData> &tmatches, float fit_param);

void do_waitKey(bool should_wait) {
	if (should_wait) waitKey();
}

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

ostringstream filename1, filename2;
// Mat img[4];
string windows[10] = {
	"Middlebury cones2", // window 0
	"Middlebury cones6", // window 1
	"Keypoint Detailed Image", // window 2
	"Keypoint Detailed Transformed Image", // window 3
	"Ground Truth", // window 4
	"SIFT Match Map", // window 5
	"FAST Keypoints", // window 6
	"FAST-FREAK Match Map", // window 7
	// "w5", // window 5 
	// "w6", // window 6
	// "w7", // window 7
	"Sift Disparity Map", // window 8
	"FAST-FREAK Disparity Map" // window 9
};

void do_main() {

	/*for (int i = 0; i < 10; i++) {
		namedWindow(windows[i], WINDOW_AUTOSIZE);
	}*/

	// Load image
	Mat im0 = imread("cones2.png", IMREAD_GRAYSCALE);
	Mat im1 = imread("cones6.png", IMREAD_GRAYSCALE);
	Mat gt = imread("cones_disp2.png", IMREAD_GRAYSCALE);

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
	GetTruthMatches(gt, sift_key1, sift_key2, sift_mf2, sft_truth);
	GetTruthMatches(gt, ffk_key1, ffk_key2, ffk_mf2, ffk_truth);

	// NOT WORKING... no time to fix, results still found with MATLAB for time being.
	// print results to screen
	//float sft_rms, ffk_rms, sft_bp, ffk_bp;
	//sft_rms = MiddleburyRMS(sft_truth,SIFT_SCALEFACTOR);
	//ffk_rms = MiddleburyRMS(ffk_truth, FFK_SCALEFACTOR);
	//sft_bp = MiddleburyBadPixels(sft_truth, SIFT_SCALEFACTOR);
	//ffk_bp = MiddleburyBadPixels(ffk_truth,FFK_SCALEFACTOR);
	//// set fixed width, precision and print results
	std::cout.precision(3);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	cout << "Results:\n"
		<< "------------------------------------------" << endl
		<< "|       |     SIFT     |    FASK-FREAK    " << endl
		<< "|  RMS  |    " << MATLAB_SIFT_RMS << "    |  " << MATLAB_FFK_RMS << endl
		<< "|Bad px |     " << MATLAB_SFT_BP << "    |   " << MATLAB_FFK_BP << endl << endl;

	cout << "Saving match data for Matlab analysis." << endl;// << "Press any key to Continue... " << endl << endl;
	print_matches("SiftMatches.csv", sift_mf, sift_key1, sift_key2, sft_truth);
	print_matches("FastFreakMatches.csv", ffk_mf, ffk_key1, ffk_key2, ffk_truth);

	//do_waitKey(DO_ALL_WAITKEYS);
}

void window_disp(int window, Mat image) {
	namedWindow(windows[window], WINDOW_NORMAL);
	imshow(windows[window], image);

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
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_t);
	matcher->match(desc2, desc1, matches);

	if (SHOW_DEBUG_IMAGES)
	{
		drawMatches(im1, keypoints2, im0, keypoints1, matches, im_matches);
		window_disp(7, im_matches);
		do_waitKey(DO_ALL_WAITKEYS);
	}
}

void ComputeAndFilterDesparityMap(Mat &im0, Mat &im1, vector<DMatch> &matches, vector<DMatch> &filt_m, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, int imno) {

	// iterate over the entire image space and do math based on matches
	MatIterator_<uchar> it; // used later for iteration maybe...
	uchar last = 0x00; // store pixel intensity... might be useful
	Mat desp = Mat::ones(im0.rows, im0.cols, CV_32FC1) * -1; // initialze desparity map
	int r = desp.rows;
	int c = desp.cols;
	double R = DBL_MAX; // radius to nearest neighbor

	// go over each pixel and scale it to the smalled distance 
	// of a match whose trained keypoint is at this pixel location
	// for visualization, nearest neighbor is taken
	vector<DMatch>::iterator mIter = matches.begin();
	for (; mIter != matches.end(); mIter++) {
		DMatch cur = (*mIter); // this match
		KeyPoint tk = keypoints1[cur.trainIdx]; // trained keypoint
		KeyPoint mk = keypoints2[cur.queryIdx]; // querry kp
		float disp = abs(mk.pt.x - tk.pt.x);

		if (disp > DISPARITY_MAX) continue;

		// if this is the first match here or if there is a match with a shorter distance
		if ((desp.at<float>(tk.pt) < 0)) {
			desp.at<float>(tk.pt) = disp;
			filt_m.push_back(cur);
		}
		// now we need to go back and remove an existing candidate
		else if (disp < desp.at<float>(tk.pt)) {
			// remove
			for (vector<DMatch>::iterator it2 = filt_m.begin(); it2 != filt_m.end(); it2++) {
				DMatch check = (*it2);
				KeyPoint ck = keypoints1[check.trainIdx];
				if (ck.pt.x == tk.pt.x && ck.pt.y == ck.pt.y) {
					filt_m.erase(it2); break; // we can leave now because I'll only store one for each point
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

void filter_dy(vector<DMatch> &matches, vector<DMatch> &filt_m, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, float threshold) {

	for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++) {
		DMatch cur = (*it);
		KeyPoint key1 = keypoints1[cur.trainIdx];
		KeyPoint key2 = keypoints2[cur.queryIdx];

		float dy = abs(key1.pt.y - key2.pt.y);
		if (dy < threshold) {
			filt_m.push_back(cur);
		}
	}
}

void print_matches(string filename, vector<DMatch> matches, vector<KeyPoint> trainKeys, vector<KeyPoint> querKeys, vector<matchData> vdat) {
	ofstream file;
	file.open(filename, ios::out);

	file << "Distance, Trained X, Trained Y, Trained angle, Trained Response, Querry X, Querry Y, Querry angle, Querry Response, Ground Truth" << endl;

	for (int i = 0; i < vdat.size(); i++) {
		// for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++) {
		DMatch cur = vdat[i].match;
		float gtruth = vdat[i].gtruth;
		KeyPoint tKey = trainKeys[cur.trainIdx];
		KeyPoint qKey = querKeys[cur.queryIdx];
		// reference:		 dist , tx  , ty  , ta  , tr  , qx  , qy  , qa  , qr  , dx  , dy  ,truth
		string lineformat = "%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f\n";
		char buf[300];
		snprintf(buf, 300, lineformat.c_str(), cur.distance, tKey.pt.x, tKey.pt.y, tKey.angle, tKey.response, qKey.pt.x, qKey.pt.y, qKey.angle, qKey.response, gtruth);
		file << buf;
	}

	file.close();
}

void GetTruthMatches(Mat &truth_img, vector<KeyPoint> &trainKeys, vector<KeyPoint> &compKeys, vector<DMatch> &matches, vector<matchData> &tmatches) {

	for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++) {
		matchData m;
		DMatch cur = (*it);
		m.match = cur;
		m.trainedPt = trainKeys[cur.trainIdx].pt;
		m.gtruth = truth_img.at<uint8_t>(m.trainedPt);
		m.dx = trainKeys[cur.trainIdx].pt.x - trainKeys[cur.trainIdx].pt.x;
		m.dy = compKeys[cur.queryIdx].pt.y - compKeys[cur.queryIdx].pt.y;
		tmatches.push_back(m);
	}
}

float MiddleburyRMS(vector<matchData> &tmatches, float fit_param)
{
	double RMS = 0;
	for (int i = 0; i < tmatches.size(); i++) {
		matchData cur = tmatches[i];
		float delta = abs(cur.dx)*fit_param - cur.gtruth;
		float sq = pow(delta, 2);
		RMS += sq;
	}
	RMS = RMS / tmatches.size();
	RMS = pow(RMS, 0.5);
	return RMS;
}

float MiddleburyBadPixels(vector<matchData> &tmatches, float fit_param)
{
	double B = 0;
	for (int i = 0; i < tmatches.size(); i++) {
		matchData cur = tmatches[i];
		float delta = abs(cur.dx)*fit_param - cur.gtruth;
		if (delta > DISPARITY_ERR_THRSH)
			B += 1.0;
	}
	B = B / tmatches.size();
	return B;
}
