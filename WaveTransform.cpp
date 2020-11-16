#include "WaveTransform.h"


WaveTransform::WaveTransform(void)
{
}


WaveTransform::~WaveTransform(void)
{
}

Mat WaveTransform::WDT(const Mat& _src, const string _wname, const int _level)
{
    //int reValue=THID_ERR_NONE;
    Mat_<float> src = Mat_<float>(_src);
    Mat dst = Mat::zeros(src.rows, src.cols, src.type());
    int row = src.rows;
    int col = src.cols;
    //高通低通滤波器
    Mat lowFilter;
    Mat highFilter;
    wavelet_D(_wname, lowFilter, highFilter);
    //小波变换
    int t = 1;


    while (t <= _level)
    {
        //先进行 行小波变换
//#pragma omp parallel for
        for (int i = 0; i < row; i++)
        {
            //取出src中要处理的数据的一行
            Mat oneRow = Mat::zeros(1, col, src.type());

            for (int j = 0; j < col; j++)
            {
                oneRow.at<float>(0, j) = src.at<float>(i, j);
            }

            oneRow = waveletDecompose(oneRow, lowFilter, highFilter);
            for (int j = 0; j < col; j++)
            {
                dst.at<float>(i, j) = oneRow.at<float>(0, j);
            }
        }

#if 0
        //    normalize(dst,dst,0,255,NORM_MINMAX);
        IplImage dstImg1 = IplImage(dst);
        cvSaveImage("dst1.jpg", &dstImg1);
#endif

        //小波列变换
//#pragma omp parallel for
        for (int j = 0; j < col; j++)
        {
            Mat oneCol = Mat::zeros(row, 1, src.type());

            for (int i = 0; i < row; i++)
            {
                oneCol.at<float>(i, 0) = dst.at<float>(i, j);//dst,not src
            }
            oneCol = (waveletDecompose(oneCol.t(), lowFilter, highFilter)).t();

            for (int i = 0; i < row; i++)
            {
                dst.at<float>(i, j) = oneCol.at<float>(i, 0);
            }
        }
#if 0
        //    normalize(dst,dst,0,255,NORM_MINMAX);
        IplImage dstImg2 = IplImage(dst);
        cvSaveImage("dst2.jpg", &dstImg2);
#endif
        /*
        char s[10];
        itoa(t,s,10);
        imshow(s,Mat_<uchar>(dst));
        waitKey(1);
        */
        //归化各子图范围0~255
        /*
        int r_len=row/2,c_len=col/2;
        for(int i=0;i<2;i++)
        {
            for(int j=0;j<2;j++)
            {
                Point p1=Point(i*c_len,j*r_len);
                Point p2=Point((i+1)*c_len,(j+1)*r_len);
                Mat ROI=dst(Rect(p1,p2));
                //ROI=cv::abs(ROI);
                normalize(ROI,ROI,0,255,CV_MINMAX);
                //imshow("a",Mat_<uchar>(ROI));
                //waitKey(0);
            }
        }
        */

        //
        //更新 
        row /= 2;
        col /= 2;
        t++;
        src = dst;
    }
    return dst;
}

Mat WaveTransform::IWDT(const Mat& _src, const string _wname, const int _level)
{
    //int reValue=THID_ERR_NONE;
    Mat src = Mat_<float>(_src);
    Mat dst;//=Mat::zeros(src.rows,src.cols,src.type());
    src.copyTo(dst);
    int N = src.rows;
    int D = src.cols;

    //高低通滤波器
    Mat lowFilter;
    Mat highFilter;
    wavelet_R(_wname, lowFilter, highFilter);

    //小波变换
    int t = 1;
    int row = N / std::pow(2., _level - 1);
    int col = D / std::pow(2., _level - 1);

    while (row <= N && col <= D)
        //while(t<=_level)
    {
        //列逆变换
        for (int j = 0; j < col; j++)
        {
            Mat oneCol = Mat::zeros(row, 1, src.type());

            for (int i = 0; i < row; i++)
            {
                oneCol.at<float>(i, 0) = src.at<float>(i, j);
            }
            oneCol = (waveletReconstruct(oneCol.t(), lowFilter, highFilter)).t();

            for (int i = 0; i < row; i++)
            {
                dst.at<float>(i, j) = oneCol.at<float>(i, 0);
            }

        }
#if 0
        IplImage dstImg2 = IplImage(dst);
        cvSaveImage("dst.jpg", &dstImg2);
#endif

        //行逆变换
        for (int i = 0; i < row; i++)
        {
            Mat oneRow = Mat::zeros(1, col, src.type());
            for (int j = 0; j < col; j++)
            {
                oneRow.at<float>(0, j) = dst.at<float>(i, j);
            }
            oneRow = waveletReconstruct(oneRow, lowFilter, highFilter);
            for (int j = 0; j < col; j++)
            {
                dst.at<float>(i, j) = oneRow.at<float>(0, j);
            }
        }

#if 0
        IplImage dstImg1 = IplImage(dst);
        cvSaveImage("dst.jpg", &dstImg1);
#endif

        char s[10];
        _itoa_s(t, s, 10);

        //Rect rrr=Rect(Point(col-1,row-1),Point(src.cols-1,src.rows-1));
        //Rect rrr=Rect(Point(0,0),Point(col-1,row-1));
        /*
        Mat showImg;//=//dst;//(rrr);
        dst.copyTo(showImg);
        Mat showImg1;
        showImg.copyTo(showImg1);
        normalize(showImg1,showImg1,0,255,CV_MINMAX);
        imshow(s,Mat_<uchar>(showImg1));
        waitKey(1);
        */
        row *= 2;
        col *= 2;
        t++;
        src = dst;
    }

    return dst;
}


void WaveTransform::wavelet_D(const string _wname, Mat& _lowFilter, Mat& _highFilter)
{
    if (_wname == "haar" || _wname == "db1")
    {
        int N = 2;
        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        _lowFilter.at<float>(0, 0) = 1 / sqrtf(N);
        _lowFilter.at<float>(0, 1) = 1 / sqrtf(N);

        _highFilter.at<float>(0, 0) = -1 / sqrtf(N);
        _highFilter.at<float>(0, 1) = 1 / sqrtf(N);
    }
    else if (_wname == "sym2")
    {
        int N = 4;
        float h[] = { -0.4830, 0.8365, -0.2241, -0.1294 };
        float l[] = { -0.1294, 0.2241,  0.8365, 0.4830 };

        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        for (int i = 0; i < N; i++)
        {
            _lowFilter.at<float>(0, i) = l[i];
            _highFilter.at<float>(0, i) = h[i];
        }
    }
}
void WaveTransform::wavelet_R(const string _wname, Mat& _lowFilter, Mat& _highFilter)
{
    if (_wname == "haar" || _wname == "db1")
    {
        int N = 2;
        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);


        _lowFilter.at<float>(0, 0) = 1 / sqrtf(N);
        _lowFilter.at<float>(0, 1) = 1 / sqrtf(N);

        _highFilter.at<float>(0, 0) = 1 / sqrtf(N);
        _highFilter.at<float>(0, 1) = -1 / sqrtf(N);
    }
    else if (_wname == "sym2")
    {
        int N = 4;
        float h[] = { -0.1294,-0.2241,0.8365,-0.4830 };
        float l[] = { 0.4830, 0.8365, 0.2241, -0.1294 };

        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        for (int i = 0; i < N; i++)
        {
            _lowFilter.at<float>(0, i) = l[i];
            _highFilter.at<float>(0, i) = h[i];
        }
    }
}


Mat WaveTransform::waveletDecompose(const Mat& _src, const Mat& _lowFilter, const Mat& _highFilter)
{
    assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
    assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
    // Mat& src = Mat_<float>(_src);
    Mat src = Mat_<float>(_src);


    int D = src.cols;

    // Mat& lowFilter = Mat_<float>(_lowFilter);
    // Mat& highFilter = Mat_<float>(_highFilter);
    Mat lowFilter = Mat_<float>(_lowFilter);
    Mat highFilter = Mat_<float>(_highFilter);

    //频域滤波或时域卷积；ifft( fft(x) * fft(filter)) = cov(x,filter) 
    Mat dst1 = Mat::zeros(1, D, src.type());
    Mat dst2 = Mat::zeros(1, D, src.type());

    filter2D(src, dst1, -1, lowFilter);
    filter2D(src, dst2, -1, highFilter);

    //下采样
    //数据拼接
    for (int i = 0, j = 1; i < D / 2; i++, j += 2)
    {
        src.at<float>(0, i) = dst1.at<float>(0, j);//lowFilter
        src.at<float>(0, i + D / 2) = dst2.at<float>(0, j);//highFilter
    }
    return src;
}


Mat WaveTransform::waveletReconstruct(const Mat& _src, const Mat& _lowFilter, const Mat& _highFilter)
{
    assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
    assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
    // Mat& src = Mat_<float>(_src);
    Mat src = Mat_<float>(_src);

    int D = src.cols;

    // Mat& lowFilter = Mat<float>(_lowFilter);
    // Mat& highFilter = Mat_<float>(_highFilter);
    Mat lowFilter = Mat_<float>(_lowFilter);
    Mat highFilter = Mat_<float>(_highFilter);

    /// 插值;
    Mat Up1 = Mat::zeros(1, D, src.type());
    Mat Up2 = Mat::zeros(1, D, src.type());

    //Mat roi1(src, Rect(0, 0, D / 2, 1));
    //Mat roi2(src, Rect(D / 2, 0, D / 2, 1));
    /// 插值为0

    for (int i = 0, cnt = 0; i < D / 2; i++, cnt += 2)
    {
        Up1.at<float>(0, cnt) = src.at<float>(0, i);     ///< 前一半
        Up2.at<float>(0, cnt) = src.at<float>(0, i + D / 2); ///< 后一半
    }
    //std::cout<<Up1<<std::endl;

    /// 线性插值

    //resize(roi1, Up1, Up1.size(), 0, 0, INTER_CUBIC);
    //resize(roi2, Up2, Up2.size(), 0, 0, INTER_CUBIC);


    /// 前一半低通，后一半高通
    Mat dst1 = Mat::zeros(1, D, src.type());
    Mat dst2 = Mat::zeros(1, D, src.type());
    filter2D(Up1, dst1, -1, lowFilter);
    filter2D(Up2, dst2, -1, highFilter);

    /// 结果相加
    dst1 = dst1 + dst2;
    return dst1;

}