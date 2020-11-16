#include<iostream>
#include<opencv.hpp>
#include "WaveTransform.h"
using namespace cv;
using namespace std;
int main()
{
	char filename[] = "3.png";
	Mat src = imread(filename,0);

	int level = 2;//·Ö½â½×´Î
	double dishu = 2;
	int result = (int)pow(dishu, level);

	WaveTransform m_waveTransform;

	//double a=clock();

	// cout << src.rows << endl << src.cols << endl;

	// resize(src,src,Size((512/result)*result,(512/result)*result));

	Mat img=src;
	// cvtColor(src, img, COLOR_RGB2BGR);
	// normalize(img, img, 0, 255, NORM_MINMAX);
	imshow("img", img);
	Mat float_src;
	img.convertTo(float_src, CV_32F);

	Mat imgWave = m_waveTransform.WDT(float_src, "sym2", level);	//haar,sym2
	imgWave.convertTo(float_src, CV_32F);
	Mat imgIWave = m_waveTransform.IWDT(float_src, "sym2", level);
	imshow("imgWave", Mat_<uchar>(imgWave));
	// cout << imgWave.rows << endl << imgWave.cols << endl;
	// cout << imgIWave.rows << endl << imgIWave.cols << endl;
	normalize(imgIWave, imgIWave, 0, 255, NORM_MINMAX);
	imshow("IWDT", Mat_<uchar>(imgIWave));

	waitKey(0);
	return 0;
}