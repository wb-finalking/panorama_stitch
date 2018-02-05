#include "stitch.h"

//********************cylinderProjection**********************//
cv::Mat Stitch::cylinderProjection(const cv::Mat &src)
{
	int projection_width, projection_height;
	//CImg<unsigned char> res(src.width(), src.height(), 1, src.spectrum(), 0);
	cv::Mat res(src.rows, src.cols,CV_8UC3);
	float r;

	if (src.cols < src.rows) {
		projection_width = src.cols;
		projection_height = src.rows;

		r = (projection_width / 2.0) / tan(ANGLE * PI / 180.0);

		for (int i = 0; i < res.cols; i++) {
			for (int j = 0; j < res.rows; j++) {
				float dst_x = j - projection_width / 2;
				float dst_y = i - projection_height / 2;

				float k = r / sqrt(r * r + dst_x * dst_x);
				float src_x = dst_x / k;
				float src_y = dst_y / k;

				if (src_x + projection_width / 2 >= 0 && src_x + projection_width / 2 < src.cols
					&& src_y + projection_height / 2 >= 0 && src_y + projection_height / 2 < src.rows) {
					res.at<cv::Vec3b>(i, j) = bilinear_interpolation(src, src_y + projection_height / 2, src_x + projection_width / 2);
				}
			}
		}

	}
	else {
		projection_width = src.rows;
		projection_height = src.cols;

		r = (projection_width / 2.0) / tan(ANGLE * PI / 180.0);

		for (int i = 0; i < res.cols; i++) {
			for (int j = 0; j < res.rows; j++) {
				float dst_x = i - projection_width / 2;
				float dst_y = j - projection_height / 2;

				float k = r / sqrt(r * r + dst_x * dst_x);
				float src_x = dst_x / k;
				float src_y = dst_y / k;

				if (src_x + projection_width / 2 >= 0 && src_x + projection_width / 2 < src.rows
					&& src_y + projection_height / 2 >= 0 && src_y + projection_height / 2 < src.cols) {
					res.at<cv::Vec3b>(i, j) = bilinear_interpolation(src, src_x + projection_width / 2, src_y + projection_height / 2);
					
				}
			}
		}

	}

	return res;
}

//********************feature**********************//
vector<cv::DMatch> Stitch::orbpairs(cv::Mat &imageDesc1, cv::Mat &imageDesc2)
{
	cv::flann::Index flannIndex(imageDesc1, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

	vector<cv::DMatch> GoodMatchePoints;

	cv::Mat macthIndex(imageDesc2.rows, 2, CV_32SC1), matchDistance(imageDesc2.rows, 2, CV_32FC1);
	flannIndex.knnSearch(imageDesc2, macthIndex, matchDistance, 2, cv::flann::SearchParams());

	vector<point_pair> pairs;
	// Lowe's algorithm,获取优秀匹配点
	for (int i = 0; i < matchDistance.rows; i++)
	{
		if (matchDistance.at<float>(i, 0) < 0.4 * matchDistance.at<float>(i, 1))
		{
			cv::DMatch dmatches(i, macthIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
			GoodMatchePoints.push_back(dmatches);
		}
	}

	return GoodMatchePoints;
}

cv::Mat Stitch::getDescriptorFromFeature(vector<cv::KeyPoint> & feature, cv::Mat &image1)
{
	cv::OrbDescriptorExtractor  orbDescriptor;
	cv::Mat imageDesc1;
	orbDescriptor.compute(image1, feature, imageDesc1);
	return imageDesc1;
}

vector<cv::KeyPoint>  Stitch::getFeatureFromImage(const cv::Mat &input) {

	cv::Mat src(input);
	float resize_factor;
	if (input.cols < input.rows) {
		resize_factor = RESIZE_SIZE / input.cols;
	}
	else {
		resize_factor = RESIZE_SIZE / input.rows;
	}

	if (resize_factor >= 1) {
		resize_factor = 1;
	}
	else {
		cv::resize(src, src, cv::Size(src.cols* resize_factor, src.rows * resize_factor));
	}

	cv::Mat image1;
	cvtColor(src, image1, CV_RGB2GRAY);


	//提取特征点    
	std::vector<cv::KeyPoint> keypoints_1;
	cv::OrbFeatureDetector  orbDetector(3000);
	//vector<cv::KeyPoint> keyPoint1, keyPoint2;
	orbDetector.detect(image1, keypoints_1);


	/*cv::Mat descriptors_1;
	orb.compute(src, keypoints_1, descriptors_1);


	map<vector<float>, cv::KeyPoint> features;

	for (int i = 0; i < keypoints_1.size(); i++)
	{
		vector<float> des;
		des = descriptors_1.rowRange(i,i).clone();
		features.insert(pair<vector<float>, cv::KeyPoint>(des, keypoints_1[i]));
	}*/

	return keypoints_1;
}

//********************match**********************//
float Stitch::getXAfterWarping(float x, float y, Parameters H) {
	return H.c1 * x + H.c2 * y + H.c3 * x * y + H.c4;
}

float Stitch::getYAfterWarping(float x, float y, Parameters H) {
	return H.c5 * x + H.c6 * y + H.c7 * x * y + H.c8;
}

vector<point_pair> Stitch::getPointPairsDirectly(const cv::Mat &img1, const cv::Mat &img2)
{
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	cv::ORB orb;
	orb.detect(img1, keypoints_1);
	orb.detect(img2, keypoints_2);

	cv::Mat descriptors_1, descriptors_2;
	orb.compute(img1, keypoints_1, descriptors_1);
	orb.compute(img2, keypoints_2, descriptors_2);

	cv::BFMatcher matcher(cv::NORM_HAMMING);
	std::vector<cv::DMatch> mathces;
	matcher.match(descriptors_1, descriptors_2, mathces);

	vector<point_pair> res;
	for (int i = 0; i < mathces.size(); i++)
	{
		if (mathces[i].distance <40)
		{
			cv::KeyPoint left = keypoints_1[mathces[i].trainIdx];
			cv::KeyPoint right = keypoints_2[mathces[i].queryIdx];
			res.push_back(point_pair(left, right));
		}
	}

	return res;
}

Parameters Stitch::getHomographyFromPoingPairs(const vector<point_pair> &pair) {
	//assert(pair.size() == 4);

	float u0 = pair[0].a.pt.x, v0 = pair[0].a.pt.y ;
	float u1 = pair[1].a.pt.x, v1 = pair[1].a.pt.y ;
	float u2 = pair[2].a.pt.x, v2 = pair[2].a.pt.y ;
	float u3 = pair[3].a.pt.x, v3 = pair[3].a.pt.y ;

	float x0 = pair[0].b.pt.x , y0 = pair[0].b.pt.y ;
	float x1 = pair[1].b.pt.x , y1 = pair[1].b.pt.y ;
	float x2 = pair[2].b.pt.x , y2 = pair[2].b.pt.y ;
	float x3 = pair[3].b.pt.x , y3 = pair[3].b.pt.y ;

	float c1, c2, c3, c4, c5, c6, c7, c8;

	c1 = -(u0*v0*v1*x2 - u0*v0*v2*x1 - u0*v0*v1*x3 + u0*v0*v3*x1 - u1*v0*v1*x2 + u1*v1*v2*x0 + u0*v0*v2*x3 - u0*v0*v3*x2 + u1*v0*v1*x3 - u1*v1*v3*x0 + u2*v0*v2*x1 - u2*v1*v2*x0
		- u1*v1*v2*x3 + u1*v1*v3*x2 - u2*v0*v2*x3 + u2*v2*v3*x0 - u3*v0*v3*x1 + u3*v1*v3*x0 + u2*v1*v2*x3 - u2*v2*v3*x1 + u3*v0*v3*x2 - u3*v2*v3*x0 - u3*v1*v3*x2 + u3*v2*v3*x1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
		- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	c2 = (u0*u1*v0*x2 - u0*u2*v0*x1 - u0*u1*v0*x3 - u0*u1*v1*x2 + u0*u3*v0*x1 + u1*u2*v1*x0 + u0*u1*v1*x3 + u0*u2*v0*x3 + u0*u2*v2*x1 - u0*u3*v0*x2 - u1*u2*v2*x0 - u1*u3*v1*x0
		- u0*u2*v2*x3 - u0*u3*v3*x1 - u1*u2*v1*x3 + u1*u3*v1*x2 + u1*u3*v3*x0 + u2*u3*v2*x0 + u0*u3*v3*x2 + u1*u2*v2*x3 - u2*u3*v2*x1 - u2*u3*v3*x0 - u1*u3*v3*x2 + u2*u3*v3*x1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
		- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	c3 = (u0*v1*x2 - u0*v2*x1 - u1*v0*x2 + u1*v2*x0 + u2*v0*x1 - u2*v1*x0 - u0*v1*x3 + u0*v3*x1 + u1*v0*x3 - u1*v3*x0 - u3*v0*x1 + u3*v1*x0
		+ u0*v2*x3 - u0*v3*x2 - u2*v0*x3 + u2*v3*x0 + u3*v0*x2 - u3*v2*x0 - u1*v2*x3 + u1*v3*x2 + u2*v1*x3 - u2*v3*x1 - u3*v1*x2 + u3*v2*x1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
		- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	c4 = (u0*u1*v0*v2*x3 - u0*u1*v0*v3*x2 - u0*u2*v0*v1*x3 + u0*u2*v0*v3*x1 + u0*u3*v0*v1*x2 - u0*u3*v0*v2*x1 - u0*u1*v1*v2*x3 + u0*u1*v1*v3*x2 + u1*u2*v0*v1*x3 - u1*u2*v1*v3*x0 - u1*u3*v0*v1*x2 + u1*u3*v1*v2*x0
		+ u0*u2*v1*v2*x3 - u0*u2*v2*v3*x1 - u1*u2*v0*v2*x3 + u1*u2*v2*v3*x0 + u2*u3*v0*v2*x1 - u2*u3*v1*v2*x0 - u0*u3*v1*v3*x2 + u0*u3*v2*v3*x1 + u1*u3*v0*v3*x2 - u1*u3*v2*v3*x0 - u2*u3*v0*v3*x1 + u2*u3*v1*v3*x0)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
		- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	c5 = -(u0*v0*v1*y2 - u0*v0*v2*y1 - u0*v0*v1*y3 + u0*v0*v3*y1 - u1*v0*v1*y2 + u1*v1*v2*y0 + u0*v0*v2*y3 - u0*v0*v3*y2 + u1*v0*v1*y3 - u1*v1*v3*y0 + u2*v0*v2*y1 - u2*v1*v2*y0
		- u1*v1*v2*y3 + u1*v1*v3*y2 - u2*v0*v2*y3 + u2*v2*v3*y0 - u3*v0*v3*y1 + u3*v1*v3*y0 + u2*v1*v2*y3 - u2*v2*v3*y1 + u3*v0*v3*y2 - u3*v2*v3*y0 - u3*v1*v3*y2 + u3*v2*v3*y1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
		- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	c6 = (u0*u1*v0*y2 - u0*u2*v0*y1 - u0*u1*v0*y3 - u0*u1*v1*y2 + u0*u3*v0*y1 + u1*u2*v1*y0 + u0*u1*v1*y3 + u0*u2*v0*y3 + u0*u2*v2*y1 - u0*u3*v0*y2 - u1*u2*v2*y0 - u1*u3*v1*y0
		- u0*u2*v2*y3 - u0*u3*v3*y1 - u1*u2*v1*y3 + u1*u3*v1*y2 + u1*u3*v3*y0 + u2*u3*v2*y0 + u0*u3*v3*y2 + u1*u2*v2*y3 - u2*u3*v2*y1 - u2*u3*v3*y0 - u1*u3*v3*y2 + u2*u3*v3*y1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
		- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	c7 = (u0*v1*y2 - u0*v2*y1 - u1*v0*y2 + u1*v2*y0 + u2*v0*y1 - u2*v1*y0 - u0*v1*y3 + u0*v3*y1 + u1*v0*y3 - u1*v3*y0 - u3*v0*y1 + u3*v1*y0
		+ u0*v2*y3 - u0*v3*y2 - u2*v0*y3 + u2*v3*y0 + u3*v0*y2 - u3*v2*y0 - u1*v2*y3 + u1*v3*y2 + u2*v1*y3 - u2*v3*y1 - u3*v1*y2 + u3*v2*y1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
		- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	c8 = (u0*u1*v0*v2*y3 - u0*u1*v0*v3*y2 - u0*u2*v0*v1*y3 + u0*u2*v0*v3*y1 + u0*u3*v0*v1*y2 - u0*u3*v0*v2*y1 - u0*u1*v1*v2*y3 + u0*u1*v1*v3*y2 + u1*u2*v0*v1*y3 - u1*u2*v1*v3*y0 - u1*u3*v0*v1*y2 + u1*u3*v1*v2*y0
		+ u0*u2*v1*v2*y3 - u0*u2*v2*v3*y1 - u1*u2*v0*v2*y3 + u1*u2*v2*v3*y0 + u2*u3*v0*v2*y1 - u2*u3*v1*v2*y0 - u0*u3*v1*v3*y2 + u0*u3*v2*v3*y1 + u1*u3*v0*v3*y2 - u1*u3*v2*v3*y0 - u2*u3*v0*v3*y1 + u2*u3*v1*v3*y0)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
		- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	return Parameters(c1, c2, c3, c4, c5, c6, c7, c8);
	//return Parameters(c6, c5, c7, c8, c2, c1, c3, c4);
}

int Stitch::numberOfIterations(float p, float w, int num) {
	return ceil(log(1 - p) / log(1 - pow(w, num)));
}

int Stitch::random(int min, int max) {
	assert(max > min);

	return rand() % (max - min + 1) + min;
}

vector<int> Stitch::getIndexsOfInliner(const vector<point_pair> &pairs, Parameters H, set<int> seleted_indexs) {
	vector<int> inliner_indexs;

	for (int i = 0; i < pairs.size(); i++) {
		if (seleted_indexs.find(i) != seleted_indexs.end()) {
			continue;
		}

		float real_x = pairs[i].b.pt.x;
		float real_y = pairs[i].b.pt.y;

		float x = getXAfterWarping(pairs[i].a.pt.x, pairs[i].a.pt.y, H);
		float y = getYAfterWarping(pairs[i].a.pt.x, pairs[i].a.pt.y, H);

		float distance = sqrt((x - real_x) * (x - real_x) + (y - real_y) * (y - real_y));
		if (distance < RANSAC_THRESHOLD) {
			inliner_indexs.push_back(i);
		}
	}

	return inliner_indexs;
}

Parameters Stitch::leastSquaresSolution(const vector<point_pair> pairs, vector<int> inliner_indexs) {
	int calc_size = inliner_indexs.size();

	if (calc_size==0)
	{
		cout << "using full pairs" << endl;
		CImg<double> A(4, pairs.size(), 1, 1, 0);
		CImg<double> b(1, pairs.size(), 1, 1, 0);

		for (int i = 0; i < pairs.size(); i++)
		{
			A(0, i) = pairs[i].a.pt.y;
			A(1, i) = pairs[i].a.pt.x;
			A(2, i) = pairs[i].a.pt.x * pairs[i].a.pt.y;
			A(3, i) = 1;

			b(0, i) = pairs[i].b.pt.y;
		}

		CImg<double> x1 = b.get_solve(A);

		for (int i = 0; i < pairs.size(); i++) {

			b(0, i) = pairs[i].b.pt.x;
		}

		CImg<double> x2 = b.get_solve(A);

		return Parameters(x1(0, 0), x1(0, 1), x1(0, 2), x1(0, 3), x2(0, 0), x2(0, 1), x2(0, 2), x2(0, 3));
	}
	CImg<double> A(4, calc_size, 1, 1, 0);
	CImg<double> b(1, calc_size, 1, 1, 0);

	for (int i = 0; i < calc_size; i++) {
		int cur_index = inliner_indexs[i];

		A(0, i) = pairs[cur_index].a.pt.y;
		A(1, i) = pairs[cur_index].a.pt.x;
		A(2, i) = pairs[cur_index].a.pt.x * pairs[cur_index].a.pt.y;
		A(3, i) = 1;

		b(0, i) = pairs[cur_index].b.pt.y;
	}

	CImg<double> x1 = b.get_solve(A);

	for (int i = 0; i < calc_size; i++) {
		int cur_index = inliner_indexs[i];

		b(0, i) = pairs[cur_index].b.pt.x;
	}

	CImg<double> x2 = b.get_solve(A);

	return Parameters(x1(0, 0), x1(0, 1), x1(0, 2), x1(0, 3), x2(0, 0), x2(0, 1), x2(0, 2), x2(0, 3));

}

Parameters Stitch::RANSAC(const vector<point_pair> &pairs) {
	assert(pairs.size() >= NUM_OF_PAIR);

	srand(time(0));

	int iterations = numberOfIterations(CONFIDENCE, INLINER_RATIO, NUM_OF_PAIR);

	vector<int> max_inliner_indexs;

	while (iterations--) {
		vector<point_pair> random_pairs;
		set<int> seleted_indexs;

		for (int i = 0; i < NUM_OF_PAIR; i++) {
			int index = random(0, pairs.size() - 1);
			while (seleted_indexs.find(index) != seleted_indexs.end()) {
				index = random(0, pairs.size() - 1);
			}
			seleted_indexs.insert(index);

			random_pairs.push_back(pairs[index]);
		}

		Parameters H = getHomographyFromPoingPairs(random_pairs);

		vector<int> cur_inliner_indexs = getIndexsOfInliner(pairs, H, seleted_indexs);
		if (cur_inliner_indexs.size() > max_inliner_indexs.size()) {
			max_inliner_indexs = cur_inliner_indexs;
		}
	}

	Parameters t = leastSquaresSolution(pairs, max_inliner_indexs);

	return t;

}

//********************interpolation**********************//
double Stitch::sinxx(double value) {
	if (value < 0)
		value = -value;

	if (value < 1.0) {
		double temp = value * value;
		return 0.5 * temp * value - temp + 2.0 / 3.0;
	}
	else if (value < 2.0) {
		value = 2.0 - value;
		value *= value * value;
		return value / 6.0;
	}
	else {
		return 0.0;
	}
}

cv::Vec3b Stitch::bilinear_interpolation(const cv::Mat& image, float x, float y) {

	int x_pos = floor(x);
	float x_u = x - x_pos;
	int xb = (x_pos < image.rows - 1) ? x_pos + 1 : x_pos;

	int y_pos = floor(y);
	float y_v = y - y_pos;
	int yb = (y_pos < image.cols - 1) ? y_pos + 1 : y_pos;

	cv::Vec3b P1 = image.at<cv::Vec3b>(x_pos, y_pos) * (1 - x_u) + image.at<cv::Vec3b>(xb, y_pos) * x_u;
	cv::Vec3b P2 = image.at<cv::Vec3b>(x_pos, yb) * (1 - x_u) + image.at<cv::Vec3b>(xb, yb) * x_u;

	return P1 * (1 - y_v) + P2 * y_v;
}


//********************warp**********************//
four_corners_t Stitch::CalcCorners(const cv::Mat& H, const cv::Mat& src, four_corners_t corners)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	cv::Mat V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	cv::Mat V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

	return corners;
}

float Stitch::getMaxXAfterWarping(const cv::Mat &src, Parameters H) {
	float max_x = getXAfterWarping(0, 0, H);

	if (getXAfterWarping(src.rows - 1, 0, H) > max_x) {
		max_x = getXAfterWarping(src.rows - 1, 0, H);
	}
	if (getXAfterWarping(0, src.cols - 1, H) > max_x) {
		max_x = getXAfterWarping(0, src.cols - 1, H);
	}
	if (getXAfterWarping(src.rows - 1, src.cols - 1, H) > max_x) {
		max_x = getXAfterWarping(src.rows - 1, src.cols - 1, H);
	}

	return max_x;
}

float Stitch::getMinXAfterWarping(const cv::Mat &src, Parameters H) {
	float min_x = getXAfterWarping(0, 0, H);

	if (getXAfterWarping(src.rows - 1, 0, H) < min_x) {
		min_x = getXAfterWarping(src.rows - 1, 0, H);
	}
	if (getXAfterWarping(0, src.cols - 1, H) < min_x) {
		min_x = getXAfterWarping(0, src.cols - 1, H);
	}
	if (getXAfterWarping(src.rows - 1, src.cols - 1, H) < min_x) {
		min_x = getXAfterWarping(src.rows - 1, src.cols - 1, H);
	}

	return min_x;
}

float Stitch::getMaxYAfterWarping(const cv::Mat &src, Parameters H) {
	float max_y = getYAfterWarping(0, 0, H);

	if (getYAfterWarping(src.rows - 1, 0, H) > max_y) {
		max_y = getYAfterWarping(src.rows - 1, 0, H);
	}
	if (getYAfterWarping(0, src.cols - 1, H) > max_y) {
		max_y = getYAfterWarping(0, src.rows - 1, H);
	}
	if (getYAfterWarping(src.rows - 1, src.cols - 1, H) > max_y) {
		max_y = getYAfterWarping(src.rows - 1, src.cols - 1, H);
	}

	return max_y;
}

float Stitch::getMinYAfterWarping(const cv::Mat &src, Parameters H) {
	float min_y = getYAfterWarping(0, 0, H);

	if (getYAfterWarping(src.rows - 1, 0, H) < min_y) {
		min_y = getYAfterWarping(src.rows - 1, 0, H);
	}
	if (getYAfterWarping(0, src.cols - 1, H) < min_y) {
		min_y = getYAfterWarping(0, src.cols - 1, H);
	}
	if (getYAfterWarping(src.rows - 1, src.cols - 1, H) < min_y) {
		min_y = getYAfterWarping(src.rows - 1, src.cols - 1, H);
	}

	return min_y;
}

int Stitch::getWidthAfterWarping(const cv::Mat &src, Parameters H) {
	int max = getMaxXAfterWarping(src, H);
	int min = getMinXAfterWarping(src, H);

	return max - min;
}

int Stitch::getHeightAfterWarping(const cv::Mat &src, Parameters H) {
	int max = getMaxYAfterWarping(src, H);
	int min = getMinYAfterWarping(src, H);

	return max - min;
}

void Stitch::warpingImageByHomography(const cv::Mat &src, cv::Mat &dst, Parameters H, float offset_x, float offset_y) {
	for (int dst_x = 0; dst_x < dst.rows; dst_x++) {
		for (int dst_y = 0; dst_y < dst.cols; dst_y++) {
			int src_x = getXAfterWarping(dst_x + offset_x, dst_y + offset_y, H);
			int src_y = getYAfterWarping(dst_x + offset_x, dst_y + offset_y, H);

			if (src_x >= 0 && src_x < src.rows && src_y >= 0 && src_y < src.cols) {
				//cv::Vec3b tmp = bilinear_interpolation(src, src_x, src_y);
				dst.at<cv::Vec3b>(dst_x, dst_y) = bilinear_interpolation(src, src_x, src_y);
				
			}
		}
	}
}

void Stitch::movingImageByOffset(const cv::Mat &src, cv::Mat &dst, int offset_x, int offset_y) {
	for (int dst_x = 0; dst_x < dst.rows; dst_x++) {
		for (int dst_y = 0; dst_y < dst.cols; dst_y++) {
			int src_x = dst_x + offset_x;
			int src_y = dst_y + offset_y;

			if (src_x >= 0 && src_x < src.rows && src_y >= 0 && src_y < src.cols) {
					dst.at<cv::Vec3b>(dst_x, dst_y) = src.at<cv::Vec3b>(src_x, src_y);
			}
		}
	}
}

void Stitch::warpingImageByHomography(cv::Mat &imageDesc1, cv::Mat &imageDesc2,cv::Mat dst)
{
	vector<cv::Point2f> imagePoints1, imagePoints2;
	vector<cv::KeyPoint> keyPoint1, keyPoint2;
	vector<cv::DMatch> GoodMatchePoints;
	GoodMatchePoints = orbpairs(imageDesc1, imageDesc2);
	for (int i = 0; i < GoodMatchePoints.size(); i++)
	{
		imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
		imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
	}
	cv::Mat homo = findHomography(imagePoints2, imagePoints1, CV_RANSAC);
	////也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  

	//图像配准  
	cv::Mat imageTransform1, imageTransform2;
	four_corners_t corners;
	corners = CalcCorners(homo, dst, corners);
	warpPerspective(dst, imageTransform1, homo, cv::Size(MAX(corners.right_top.x, corners.right_bottom.x), dst.rows));
	//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
	/*cv::imshow("直接经过透视矩阵变换", imageTransform1);
	cv::imshow("2", src);

	cv::waitKey(0);*/
}

//********************blend**********************//
bool Stitch::isEmpty(const cv::Mat &img, int x, int y) {
	cv::Vec3b tmp(0,0,0);
	return img.at<cv::Vec3b>(x,y)==tmp;
}

cv::Mat Stitch::blendTwoImages(const cv::Mat &a, const cv::Mat &b) {
	//assert(a.width() == b.width() && a.height() == b.height() && a.spectrum() == b.spectrum());

	// Find the center point of a and overlapping part.
	double sum_a_x = 0;
	double sum_a_y = 0;
	int a_n = 0;
	//double sum_b_x = 0;
	//double sum_b_y = 0;
	//int b_n = 0;
	double sum_overlap_x = 0;
	double sum_overlap_y = 0;
	int overlap_n = 0;
	if (a.rows > a.cols) {
		for (int x = 0; x < a.rows; x++) {
			if (!isEmpty(a, x, a.cols / 2)) {
				sum_a_x += x;
				a_n++;
			}

			//if (!isEmpty(b, x, b.rows / 2)) {
			//	sum_b_x += x;
			//	b_n++;
			//}

			if (!isEmpty(a, x, a.cols / 2) && !isEmpty(b, x, a.cols / 2)) {
				sum_overlap_x += x;
				overlap_n++;
			}
		}
	}
	else {
		for (int y = 0; y < a.cols; y++) {
			if (!isEmpty(a, a.rows / 2, y)) {
				sum_a_y += y;
				a_n++;
			}

			if (!isEmpty(a, a.rows / 2, y) && !isEmpty(b, b.rows / 2, y)) {
				sum_overlap_y += y;
				overlap_n++;
			}
		}
	}

	int min_len = (a.rows < a.cols) ? a.rows : a.cols;

	int n_level = floor(log2(min_len));

	vector<cv::Mat > a_pyramid(n_level);
	vector<cv::Mat > b_pyramid(n_level);
	vector<cv::Mat > mask(n_level);

	// Initialize the base.
	//a_pyramid[0] = a.clone();
	//b_pyramid[0] = b.clone();
	a.convertTo(a_pyramid[0], CV_32FC3);
	b.convertTo(b_pyramid[0], CV_32FC3);
	mask[0] = cv::Mat(a.rows, a.cols, CV_32FC3, cv::Scalar(0, 0, 0));

	if (a.rows > a.cols) {
		if (sum_a_x / a_n < sum_overlap_x / overlap_n) {
			for (int x = 0; x < sum_overlap_x / overlap_n; x++) {
				for (int y = 0; y < a.cols; y++) {
					mask[0].at<cv::Vec3f>(x, y) = cv::Vec3f(1, 1, 1);
				}
			}
		}
		else {
			for (int x = sum_overlap_x / overlap_n + 1; x < a.rows; x++) {
				for (int y = 0; y < a.cols; y++) {
					mask[0].at<cv::Vec3f>(x, y) = cv::Vec3f(1, 1, 1);
				}
			}
		}
	}
	else {
		if (sum_a_y / a_n < sum_overlap_y / overlap_n) {
			for (int x = 0; x < a.rows; x++) {
				for (int y = 0; y < sum_overlap_y / overlap_n; y++) {
					mask[0].at<cv::Vec3f>(x, y) = cv::Vec3f(1, 1, 1);
				}
			}
		}
		else {
			for (int x = 0; x < a.rows; x++) {
				for (int y = sum_overlap_y / overlap_n; y < a.cols; y++) {
					mask[0].at<cv::Vec3f>(x, y) = cv::Vec3f(1, 1, 1);
				}
			}
		}
	}

	// Down sampling a and b, building Gaussian pyramids.
	for (int i = 1; i < n_level; i++) {
		cv::resize(a_pyramid[i-1], a_pyramid[i], cv::Size(a_pyramid[i-1].cols / 2, a_pyramid[i-1].rows / 2));
		cv::resize(b_pyramid[i-1], b_pyramid[i], cv::Size(b_pyramid[i-1].cols / 2, b_pyramid[i-1].rows / 2));

		cv::resize(mask[i-1], mask[i], cv::Size(mask[i-1].cols / 2, mask[i-1].rows / 2));
	}

	// Building Laplacian pyramids.
	for (int i = 0; i < n_level - 1; i++)
	{
		cv::Mat tmp;
		cv::resize(a_pyramid[i+1], tmp, cv::Size(a_pyramid[i].cols, a_pyramid[i].rows));
		a_pyramid[i] = a_pyramid[i] - tmp;
		cv::resize(b_pyramid[i + 1], tmp, cv::Size(b_pyramid[i].cols, b_pyramid[i].rows));
		b_pyramid[i] = b_pyramid[i] - tmp;
	}

	/*imshow("res", a_pyramid[0]);
	imshow("mask", 1-mask[0]);
	cv::waitKey(0);*/

	vector<cv::Mat > blend_pyramid(n_level);

	for (int i = 0; i < n_level; i++) 
	{
		blend_pyramid[i] = cv::Mat(a_pyramid[i].rows, a_pyramid[i].cols, CV_32FC3,cv::Scalar(0,0,0));

		for (int x = 0; x < blend_pyramid[i].rows; x++) {
			for (int y = 0; y < blend_pyramid[i].cols; y++) {
				blend_pyramid[i].at<cv::Vec3f>(x, y)[0] = a_pyramid[i].at<cv::Vec3f>(x, y)[0] * mask[i].at<cv::Vec3f>(x, y)[0] + b_pyramid[i].at<cv::Vec3f>(x, y)[0] * (1.0 - mask[i].at<cv::Vec3f>(x, y)[0]);
				blend_pyramid[i].at<cv::Vec3f>(x, y)[1] = a_pyramid[i].at<cv::Vec3f>(x, y)[1] * mask[i].at<cv::Vec3f>(x, y)[1] + b_pyramid[i].at<cv::Vec3f>(x, y)[1] * (1.0 - mask[i].at<cv::Vec3f>(x, y)[1]);
				blend_pyramid[i].at<cv::Vec3f>(x, y)[2] = a_pyramid[i].at<cv::Vec3f>(x, y)[2] * mask[i].at<cv::Vec3f>(x, y)[2] + b_pyramid[i].at<cv::Vec3f>(x, y)[2] * (1.0 - mask[i].at<cv::Vec3f>(x, y)[2]);
			}
		}
		//blend_pyramid[i] = a_pyramid[i].dot(a_pyramid[i]) + b_pyramid[i].dot(1 - mask[i]);
	}

	cv::Mat res = blend_pyramid[n_level-1];
	for (int i = n_level - 2; i >= 0; i--) 
	{
		cv::resize(res, res, cv::Size(blend_pyramid[i].cols, blend_pyramid[i].rows));
		res = res + blend_pyramid[i];
	}
	res.convertTo(res,CV_8UC3);
	return res;
}

//********************stitch**********************//
int Stitch::getMiddleIndex(vector<vector<int>> matching_index) {
	int one_side = 0;

	for (int i = 0; i < matching_index.size(); i++) {
		if (matching_index[i].size() == 1) {
			one_side = i;
			break;
		}
	}

	int middle_index = one_side;
	int pre_middle_index = -1;
	int n = matching_index.size() / 2;

	while (n--) {
		for (int i = 0; i < matching_index[middle_index].size(); i++) {
			if (matching_index[middle_index][i] != pre_middle_index) {
				pre_middle_index = middle_index;
				middle_index = matching_index[middle_index][i];

				break;
			}
		}
	}

	return middle_index;
}

int Stitch::getMiddleIndex(vector<int> sub_matching_index) {
	int one_side = sub_matching_index[0];

	for (int i = 0; i < sub_matching_index.size(); i++) {
		if (matching_index[sub_matching_index[i]].size() == 1) {
			one_side = sub_matching_index[i];
			break;
		}
	}

	int middle_index = one_side;
	int pre_middle_index = -1;
	int n = sub_matching_index.size() / 2;

	while (n--) {
		for (int i = 0; i < matching_index[middle_index].size(); i++) {
			if (matching_index[middle_index][i] != pre_middle_index) {
				pre_middle_index = middle_index;
				middle_index = matching_index[middle_index][i];

				break;
			}
		}
	}

	return middle_index;
}

void Stitch::updateFeaturesByHomography(vector<cv::KeyPoint> &feature, Parameters H, float offset_x, float offset_y) {
	for (auto iter = feature.begin(); iter != feature.end(); iter++) {
		float cur_x = iter->pt.x;
		float cur_y = iter->pt.y;
		iter->pt.y = getXAfterWarping(cur_y, cur_x, H) - offset_x;
		iter->pt.x = getYAfterWarping(cur_y, cur_x, H) - offset_y;
		//iter->second.pt.ix = int(iter->second.x);
		//iter->second.pt.iy = int(iter->second.y);
	}
}

void Stitch::updateFeaturesByOffset(vector<cv::KeyPoint> &feature, int offset_x, int offset_y) {
	for (auto iter = feature.begin(); iter != feature.end(); iter++) {
		iter->pt.x -= offset_y;
		iter->pt.y -= offset_x;
		//iter->second.ix = int(iter->second.x);
		//iter->second.iy = int(iter->second.y);
	}
}

cv::Mat Stitch::stitching(vector<cv::Mat> &src_imgs) {
	// Used to save each image's features and corresponding coordinates.
	vector<vector<cv::KeyPoint> > features(src_imgs.size());
	vector<cv::Mat> descriptor(src_imgs.size());

	for (int i = 0; i < src_imgs.size(); i++) {
		cout << "Preprocessing input image " << i << " ..." << endl;

		cout << "Cylinder projection started." << endl;
		//src_imgs[i] = cylinderProjection(src_imgs[i]);
		//cv::imshow("Cylinder projection", src_imgs[i]);
		//cv::waitKey(0);
		cout << "Cylinder projection finished." << endl;

		//CImg<unsigned char> gray = get_gray_image(src_imgs[i]);

		cout << "Extracting ORB feature started." << endl;
		features[i] = getFeatureFromImage(src_imgs[i]);
		cout << "Extracting ORB feature finished." << endl;

		cout << "Extracting ORB Descriptor started." << endl;
		descriptor[i] = getDescriptorFromFeature(features[i], src_imgs[i]);
		cout << "Extracting ORB Descriptor finished." << endl;

		cout << "Preprocessing input image " << i << " finished." << endl << endl;
	}

	// Used to record the image's adjacent images.
	bool need_stitching[MAX_STITCHING_NUM][MAX_STITCHING_NUM] = { false };

	// Used to record each image's adjacent images.
	vector<vector<int>> matching_index(src_imgs.size());

	cout << "Finding adjacent images ..." << endl;

	for (int i = 0; i < src_imgs.size(); i++) {
		for (int j = 0; j < src_imgs.size(); j++) {
			if (i == j)
				continue;

			//vector<point_pair> pairs = getPointPairsFromFeature(features[i], features[j]);
			//vector<point_pair> pairs = getPointPairsDirectly(src_imgs[i], src_imgs[j]);
			vector<cv::DMatch> GoodMatchePoints;
			//GoodMatchePoints = orbpairs(src_imgs[src_index], src_imgs[dst_index], keyPoint1, keyPoint2);
			GoodMatchePoints = orbpairs(descriptor[i], descriptor[j]);

			if (GoodMatchePoints.size() >= 20) {
				need_stitching[i][j] = true;

				cout << "Image " << i << " and " << j << " are adjacent." << endl;
				matching_index[i].push_back(j);
			}

		}
	}

	cout << endl;

	cout << "Stitching begins." << endl << endl;
	// Stitching begins.

	// Stitching from middle to have better visual effect.
	int start_index = getMiddleIndex(matching_index);
	//int start_index = 3;

	// Used to record the previous stitched image.
	int prev_dst_index = start_index;

	queue<int> unstitched_index;
	unstitched_index.push(start_index);

	cv::Mat cur_stitched_img = src_imgs[start_index];
	vector<int> stitched_idex;
	stitched_idex.push_back(start_index);

	while (!unstitched_index.empty()) {
		int src_index = unstitched_index.front();
		unstitched_index.pop();

		for (int j = matching_index[src_index].size() - 1; j >= 0; j--) {
			int dst_index = matching_index[src_index][j];

			if (need_stitching[src_index][dst_index] == false) {
				continue;
			}
			else {
				need_stitching[src_index][dst_index] = false;
				need_stitching[dst_index][src_index] = false;
				unstitched_index.push(dst_index);
			}

			// Matching features using best-bin kd-tree.
			//vector<point_pair> src_to_dst_pairs = getPointPairsFromFeature(features[src_index], features[dst_index]);
			//vector<point_pair> dst_to_src_pairs = getPointPairsFromFeature(features[dst_index], features[src_index]);
			/*vector<point_pair> src_to_dst_pairs = getPointPairsDirectly(src_imgs[src_index], src_imgs[dst_index]);
			vector<point_pair> dst_to_src_pairs = getPointPairsDirectly(src_imgs[dst_index], src_imgs[src_index]);

			if (src_to_dst_pairs.size() > dst_to_src_pairs.size()) {
				dst_to_src_pairs.clear();
				for (int i = 0; i < src_to_dst_pairs.size(); i++) {
					point_pair temp(src_to_dst_pairs[i].b, src_to_dst_pairs[i].a);
					dst_to_src_pairs.push_back(temp);
				}
			}
			else {
				src_to_dst_pairs.clear();
				for (int i = 0; i < dst_to_src_pairs.size(); i++) {
					point_pair temp(dst_to_src_pairs[i].b, dst_to_src_pairs[i].a);
					src_to_dst_pairs.push_back(temp);
				}
			}*/

			vector<cv::Point2f> imagePoints1, imagePoints2;
			vector<cv::KeyPoint> keyPoint1, keyPoint2;
			vector<cv::DMatch> GoodMatchePoints;
			//GoodMatchePoints = orbpairs(src_imgs[src_index], src_imgs[dst_index], keyPoint1, keyPoint2);
			GoodMatchePoints = orbpairs(descriptor[src_index], descriptor[dst_index]);
			vector<point_pair> src_to_dst_pairs, dst_to_src_pairs;
			src_to_dst_pairs.clear();
			dst_to_src_pairs.clear();
			for (int i = 0; i<GoodMatchePoints.size(); i++)
			{
				//imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
				//imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
				point_pair temp(features[src_index][GoodMatchePoints[i].trainIdx], features[dst_index][GoodMatchePoints[i].queryIdx]);
				src_to_dst_pairs.push_back(temp);
				point_pair temp2(features[dst_index][GoodMatchePoints[i].queryIdx], features[src_index][GoodMatchePoints[i].trainIdx]);
				dst_to_src_pairs.push_back(temp2);
			}

			// Finding homography by RANSAC.
			Parameters forward_H = RANSAC(dst_to_src_pairs);
			Parameters backward_H = RANSAC(src_to_dst_pairs);

			// Calculate the size of the image after stitching.
			float min_x = getMinXAfterWarping(src_imgs[dst_index], forward_H);
 			min_x = (min_x < 0) ? min_x : 0;
			float min_y = getMinYAfterWarping(src_imgs[dst_index], forward_H);
			min_y = (min_y < 0) ? min_y : 0;
			float max_x = getMaxXAfterWarping(src_imgs[dst_index], forward_H);
			max_x = (max_x >= cur_stitched_img.rows) ? max_x : cur_stitched_img.rows;
			float max_y = getMaxYAfterWarping(src_imgs[dst_index], forward_H);
			max_y = (max_y >= cur_stitched_img.cols) ? max_y : cur_stitched_img.cols;

			int new_height = ceil(max_x - min_x);
			int new_width= ceil(max_y - min_y);
			//int new_height = 512;
			//int new_width = 384;

			cv::Mat a(new_height, new_width, CV_8UC3, cv::Scalar(0, 0, 0));
			cv::Mat b(new_height, new_width, CV_8UC3, cv::Scalar(0, 0, 0));

			// Warping the dst image into the coordinate space of currently stitched image.
			warpingImageByHomography(src_imgs[dst_index], a, backward_H, min_x, min_y);

			// Move stitched image according to min_x and min_y as translation.
			movingImageByOffset(cur_stitched_img, b, min_x, min_y);

			// Update features coordinates according to homography.
			updateFeaturesByHomography(features[dst_index], forward_H, min_x, min_y);
			
			// Update features coordinates according to min_x and min_y.
			for (int i = 0; i < stitched_idex.size();i++)
				updateFeaturesByOffset(features[stitched_idex[i]], min_x, min_y);
			//updateFeaturesByOffset(features[prev_dst_index], min_x, min_y);
			vector<int>::iterator result = find(stitched_idex.begin(), stitched_idex.end(), dst_index);
			if (result != stitched_idex.end())
				stitched_idex.push_back(dst_index);

			/*cv::imshow("a", a);
			cv::imshow("b", b);
			cv::waitKey(0);*/

			// Blending two images.
			cur_stitched_img = blendTwoImages(a, b);
			prev_dst_index = dst_index;

			cout << "Image " << dst_index << " has stitched." << endl << endl;
		}
	}

	return cur_stitched_img;
}

void Stitch::getRelations(vector<cv::Mat> &src_imgs)
{
	// Used to save each image's features and corresponding coordinates.
	features.clear();
	descriptor.clear();
	features.resize(src_imgs.size());
	descriptor.resize(src_imgs.size());
	matching_index.clear();
	GoodMatches.clear();

	for (int i = 0; i < src_imgs.size(); i++) {
		cout << "Preprocessing input image " << i << " ..." << endl;

		cout << "Cylinder projection started." << endl;
		src_imgs[i] = cylinderProjection(src_imgs[i]);
		cout << "Cylinder projection finished." << endl;

		//CImg<unsigned char> gray = get_gray_image(src_imgs[i]);

		cout << "Extracting ORB feature started." << endl;
		features[i] = getFeatureFromImage(src_imgs[i]);
		cout << "Extracting ORB feature finished." << endl;

		cout << "Extracting ORB Descriptor started." << endl;
		descriptor[i] = getDescriptorFromFeature(features[i], src_imgs[i]);
		cout << "Extracting ORB Descriptor finished." << endl;

		cout << "Preprocessing input image " << i << " finished." << endl << endl;
	}

	// Used to record the image's adjacent images.
	for (int i = 0; i < MAX_STITCHING_NUM; i++) {
		for (int j = 0; j < MAX_STITCHING_NUM; j++) {
			need_stitching[i][j] = false ;
		}
	}

	// Used to record each image's adjacent images.
	matching_index.resize(src_imgs.size());

	cout << "Finding adjacent images ..." << endl;

	for (int i = 0; i < src_imgs.size(); i++) {
		for (int j = i+1; j < src_imgs.size(); j++) {
			if (i == j)
				continue;
			vector<cv::DMatch> GoodMatchePoints;
			GoodMatchePoints = orbpairs(descriptor[i], descriptor[j]);

			if (GoodMatchePoints.size() >= 20) {
				need_stitching[i][j] = true;
				need_stitching[j][i] = true;
				cout << "Image " << i << " and " << j << " are adjacent." << endl;
				matching_index[i].push_back(j);
				matching_index[j].push_back(i);
				Pair pair(i,j);
				GoodMatches.insert(make_pair(pair, GoodMatchePoints));
			}

		}
	}

	cout << endl;

	cout << "Stitching begins." << endl << endl;
}

vector<vector<int> > Stitch::getConnectedDomain()
{
	vector<vector<int> > connected_domain;
	vector<bool> unAssigned(src_imgs_size, TRUE);
	while (1)
	{
		vector<int> domain;
		int start_index = -1;
		for (int i = 0; i < unAssigned.size(); i++)
		{
			if (unAssigned[i] == TRUE)
			{
				start_index = i;
				break;
			}
		}
		if (start_index == -1)
			break;

		queue<int> q;
		q.push(start_index);
		while (!q.empty())
		{
			int cur_index = q.front();
			q.pop();
			unAssigned[cur_index] = FALSE;
			domain.push_back(cur_index);
			for (int i = 0; i < matching_index[cur_index].size(); i++)
			{
				if (unAssigned[matching_index[cur_index][i]])
					q.push(matching_index[cur_index][i]);
			}
		}
		connected_domain.push_back(domain);

	}
	return connected_domain;
}

vector<cv::Mat> Stitch::stitcher(vector<cv::Mat> &src_imgs)
{
	vector<cv::Mat> res;
	src_imgs_size = src_imgs.size();
	getRelations(src_imgs);

	vector<vector<int> > connected_domain=getConnectedDomain();
	cout << "Different Domain...." << endl;
	for (int i = 0; i < connected_domain.size(); i++){
		for (int j = 0; j < connected_domain[i].size(); j++)
			cout << "--" << connected_domain[i][j] << "--";
		cout << endl;
	}

	for (int i = 0; i < connected_domain.size(); i++)
	{
		cout << "Domain " <<i<< endl;
		
		if (connected_domain[i].size() == 1)
		{
			cout << "Rejected Picture " << connected_domain[i][0]<< endl;
			res.push_back(src_imgs[connected_domain[i][0]]);
			continue;
		}

		int start_index = getMiddleIndex(connected_domain[i]);
		//int start_index = 3;

		// Used to record the previous stitched image.
		int prev_dst_index = start_index;

		queue<int> unstitched_index;
		unstitched_index.push(start_index);

		cv::Mat cur_stitched_img = src_imgs[start_index];
		vector<int> stitched_idex;
		stitched_idex.push_back(start_index);

		while (!unstitched_index.empty()) {
			int src_index = unstitched_index.front();
			unstitched_index.pop();

			for (int j = matching_index[src_index].size() - 1; j >= 0; j--) {
				int dst_index = matching_index[src_index][j];

				if (need_stitching[src_index][dst_index] == false) {
					continue;
				}
				else {
					need_stitching[src_index][dst_index] = false;
					need_stitching[dst_index][src_index] = false;
					unstitched_index.push(dst_index);
				}

				// Matching features using best-bin kd-tree.
				vector<cv::Point2f> imagePoints1, imagePoints2;
				vector<cv::KeyPoint> keyPoint1, keyPoint2;
				vector<cv::DMatch> GoodMatchePoints;
				//GoodMatchePoints = orbpairs(descriptor[src_index], descriptor[dst_index]);
				std::map<Pair, vector<cv::DMatch>>::iterator it = GoodMatches.find(Pair(src_index, dst_index));
				if (it == GoodMatches.end())
					it = GoodMatches.find(Pair(dst_index, src_index));
				GoodMatchePoints = it->second;
				vector<point_pair> src_to_dst_pairs, dst_to_src_pairs;
				src_to_dst_pairs.clear();
				dst_to_src_pairs.clear();
				if (src_index<dst_index)
					for (int i = 0; i<GoodMatchePoints.size(); i++)
					{
						point_pair temp(features[src_index][GoodMatchePoints[i].trainIdx], features[dst_index][GoodMatchePoints[i].queryIdx]);
						src_to_dst_pairs.push_back(temp);
						point_pair temp2(features[dst_index][GoodMatchePoints[i].queryIdx], features[src_index][GoodMatchePoints[i].trainIdx]);
						dst_to_src_pairs.push_back(temp2);
					}
				else
					for (int i = 0; i<GoodMatchePoints.size(); i++)
					{
						point_pair temp(features[src_index][GoodMatchePoints[i].queryIdx], features[dst_index][GoodMatchePoints[i].trainIdx]);
						src_to_dst_pairs.push_back(temp);
						point_pair temp2(features[dst_index][GoodMatchePoints[i].trainIdx], features[src_index][GoodMatchePoints[i].queryIdx]);
						dst_to_src_pairs.push_back(temp2);
					}

				// Finding homography by RANSAC.
				Parameters forward_H = RANSAC(dst_to_src_pairs);
				Parameters backward_H = RANSAC(src_to_dst_pairs);

				// Calculate the size of the image after stitching.
				float min_x = getMinXAfterWarping(src_imgs[dst_index], forward_H);
				min_x = (min_x < 0) ? min_x : 0;
				float min_y = getMinYAfterWarping(src_imgs[dst_index], forward_H);
				min_y = (min_y < 0) ? min_y : 0;
				float max_x = getMaxXAfterWarping(src_imgs[dst_index], forward_H);
				max_x = (max_x >= cur_stitched_img.rows) ? max_x : cur_stitched_img.rows;
				float max_y = getMaxYAfterWarping(src_imgs[dst_index], forward_H);
				max_y = (max_y >= cur_stitched_img.cols) ? max_y : cur_stitched_img.cols;

				int new_height = ceil(max_x - min_x);
				int new_width = ceil(max_y - min_y);

				cv::Mat a(new_height, new_width, CV_8UC3, cv::Scalar(0, 0, 0));
				cv::Mat b(new_height, new_width, CV_8UC3, cv::Scalar(0, 0, 0));

				// Warping the dst image into the coordinate space of currently stitched image.
				warpingImageByHomography(src_imgs[dst_index], a, backward_H, min_x, min_y);

				// Move stitched image according to min_x and min_y as translation.
				movingImageByOffset(cur_stitched_img, b, min_x, min_y);

				// Update features coordinates according to homography.
				updateFeaturesByHomography(features[dst_index], forward_H, min_x, min_y);

				// Update features coordinates according to min_x and min_y.
				//updateFeaturesByOffset(features[prev_dst_index], min_x, min_y);			
				vector<int>::iterator result = find(stitched_idex.begin(), stitched_idex.end(), prev_dst_index);
				if (result != stitched_idex.end())
					stitched_idex.push_back(prev_dst_index);
				for (int i = 0; i < stitched_idex.size(); i++)
					updateFeaturesByOffset(features[stitched_idex[i]], min_x, min_y);

				// Blending two images.
				cur_stitched_img = blendTwoImages(a, b);
				prev_dst_index = dst_index;

				cout << "Image " << dst_index << " has stitched." << endl << endl;
			}
		}

		res.push_back(cur_stitched_img);
	}

	return res;
}