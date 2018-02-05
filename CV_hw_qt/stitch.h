#ifndef _STITCH_
#define _STITCH_
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"
#include <map>
#include <set>
#include <vector>
#include <string>
#include <queue>
#include "CImg.h"
#include "structure.h"

using namespace std;
using namespace cimg_library;
//using namespace cv;

//extern "C" {
//#include "vl/generic.h"
//#include "vl/sift.h"
//#include "vl/kdtree.h"
//}

#define RESIZE_SIZE 500.0
#define NUM_OF_PAIR 4
#define CONFIDENCE 0.99
#define INLINER_RATIO 0.5
#define RANSAC_THRESHOLD 4.0
#define MAX_STITCHING_NUM 20
#define ANGLE 15
#define PI 3.1415926

class Stitch
{
public:
	/*Stitch();
	~Stitch();*/

	//cylinderProjection
	cv::Mat cylinderProjection(const cv::Mat &src);

	//feature
	vector<cv::KeyPoint> getFeatureFromImage(const cv::Mat &src);
	cv::Mat getDescriptorFromFeature(vector<cv::KeyPoint> & feature, cv::Mat &image1);
	vector<cv::DMatch> orbpairs(cv::Mat &imageDesc1, cv::Mat &imageDesc2);

	//match
	float getXAfterWarping(float x, float y, Parameters H);
	float getYAfterWarping(float x, float y, Parameters H);
	vector<point_pair> getPointPairsDirectly(const cv::Mat &img1, const cv::Mat &img2);
	int numberOfIterations(float p, float w, int num);
	int random(int min, int max);
	vector<int> getIndexsOfInliner(const vector<point_pair> &pairs, Parameters H, set<int> seleted_indexs);
	Parameters leastSquaresSolution(const vector<point_pair> pairs, vector<int> inliner_indexs);
	Parameters getHomographyFromPoingPairs(const vector<point_pair> &pairs);
	Parameters RANSAC(const vector<point_pair> &pairs);

	//interpolation
	double sinxx(double value);
	cv::Vec3b bilinear_interpolation(const cv::Mat& image, float x, float y);
	cv::Vec3b bicubic_interpolation(const  cv::Mat& image, float x, float y);

	//warp
	four_corners_t CalcCorners(const cv::Mat& H, const cv::Mat& src, four_corners_t corners);
	float getMaxXAfterWarping(const cv::Mat &src, Parameters H);
	float getMinXAfterWarping(const cv::Mat &src, Parameters H);
	float getMaxYAfterWarping(const cv::Mat &src, Parameters H);
	float getMinYAfterWarping(const cv::Mat &src, Parameters H);
	int getWidthAfterWarping(const cv::Mat &src, Parameters H);
	int getHeightAfterWarping(const cv::Mat &src, Parameters H);
	void warpingImageByHomography(const cv::Mat &src, cv::Mat &dst, Parameters H, float offset_x, float offset_y);
	void movingImageByOffset(const cv::Mat &src, cv::Mat &dst, int offset_x, int offset_y);
	void warpingImageByHomography(cv::Mat &imageDesc1, cv::Mat &imageDesc2, cv::Mat dst);

	//blending
	bool isEmpty(const cv::Mat &img, int x, int y);
	cv::Mat blendTwoImages(const cv::Mat &a, const cv::Mat &b);

	//stitching
	int getMiddleIndex(vector<vector<int>> matching_index);
	int getMiddleIndex(vector<int> sub_matching_index);
	void updateFeaturesByHomography(vector<cv::KeyPoint>&feature, Parameters H, float offset_x, float offset_y);
	void updateFeaturesByOffset(vector<cv::KeyPoint> &feature, int offset_x, int offset_y);
	cv::Mat stitching(vector<cv::Mat> &src_imgs);
	void getRelations(vector<cv::Mat> &src_imgs);
	vector<cv::Mat> stitcher(vector<cv::Mat> &src_imgs);
	vector<vector<int> > getConnectedDomain();

private:
	int src_imgs_size;
	vector<vector<cv::KeyPoint> > features;
	vector<cv::Mat> descriptor;
	bool need_stitching[MAX_STITCHING_NUM][MAX_STITCHING_NUM];
	vector<vector<int>> matching_index;
	map<Pair,vector<cv::DMatch>> GoodMatches;
};

#endif