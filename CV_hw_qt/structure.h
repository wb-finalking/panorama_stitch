#ifndef _STRUCTURE_
#define _STRUCTURE_
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <map>
#include <vector>
#include <cmath>

using namespace std;

struct point_pair {
	cv::KeyPoint a;
	cv::KeyPoint b;
	point_pair(cv::KeyPoint _a, cv::KeyPoint _b) {
		a = _a;
		b = _b;
	}
};

struct point_pairs {
	vector<point_pair> pairs;
	int src;
	int dst;
	point_pairs(const vector<point_pair> &_pairs, int s, int d) {
		pairs = _pairs;
		src = s;
		dst = d;
	}
};

struct Parameters {
	float c1;
	float c2;
	float c3;
	float c4;
	float c5;
	float c6;
	float c7;
	float c8;
	Parameters(float _c1, float _c2, float _c3, float _c4, float _c5, float _c6, float _c7, float _c8) {
		c1 = _c1;
		c2 = _c2;
		c3 = _c3;
		c4 = _c4;
		c5 = _c5;
		c6 = _c6;
		c7 = _c7;
		c8 = _c8;
	}
	void print() {
		cout << c1 << " " << c2 << " " << c3 << " " << c4 << endl;
		cout << c5 << " " << c6 << " " << c7 << " " << c8 << endl;
	}
};


typedef struct
{
	cv::Point2f left_top;
	cv::Point2f left_bottom;
	cv::Point2f right_top;
	cv::Point2f right_bottom;
}four_corners_t;

struct Pair {
	int a, b;
	Pair(int _a, int _b) {
		a = _a;
		b = _b;
	}
	bool Pair::operator <(const Pair &pair)  const
	{
		return this->a < pair.a || (this->a == pair.a && this->b < pair.b);
	}
};


#endif