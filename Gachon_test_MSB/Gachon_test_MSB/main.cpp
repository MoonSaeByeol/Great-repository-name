#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>


#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


//1. �������� �����϶�.
/*
int main()
{
	for (int i = 1; i < 10; i++)
	{
		int x = i % 3;
		cout << i << "��\t";
		if (x == 0)
		{
			cout << "\n";
			for (int j = 1; j < 10; j++)
			{
				for (int i_2 = i - 2; i_2 <= i; i_2++)
				{
					cout << i_2 << "x" << j << "=" << i_2 * j << "\t";
				}
				cout << endl;
			}
			cout << endl;
		}

	}
	return 0;
}
//*/


//2. GLCM Matrix �����϶�.
// �׷��� ���� �̹����� �־����� ��
// ���� �߻� ����� �̹������� Ư�� ���� �������� �ִ� �ȼ� ���� 
// �󸶳� ���� �߻��ϴ��� ���
// ���� GLCM�� ũ��� �ȼ����� �ִ��� �� ��.
/*

Mat clac_GLCM(Mat m)
{
	double minVal, maxVal;
	minMaxLoc(m, &minVal, &maxVal);
	int size_GLCM = maxVal - minVal + 1;

	Mat m_GLCM = Mat::zeros(size_GLCM, size_GLCM, CV_8SC1);

	for (int y = 0; y < m.rows; y++)
	{
		uchar* p = m.ptr<uchar>(y);
		for (int x = 0; x < m.cols; x++)
		{
			if (x + 1 < m.cols)
				m_GLCM.at<uchar>(p[x], p[x + 1])++;
		}
	}
	return m_GLCM;
}

int main()
{
	int rows = 3;
	int cols = 3;
	char data[] = { 0,1,1, 1,1,2, 0,1,0 };
	Mat m_input(rows, cols, CV_8SC1, data);
	Mat m_output = clac_GLCM(m_input);

	cout << m_input << endl << m_output << endl;
	return 0;
}

//*/

//3. ���׶� ������ ���϶�.
/*
Mat getHistImg(const Mat& hist)		//������׷� �̹���
{
	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat histImg(100, hist.cols, CV_8UC1, Scalar(255));

	for (int i = 0; i < hist.cols; i++)
	{
		line(histImg,
			Point(i, 100),
			Point(i, 100 - cvRound((hist.at<float>(0, i) / histMax) * 100)),
			Scalar(0));
	}

	return histImg;
}


int main()
{
	Mat input = imread("q3.jpg");
	if (input.empty())	cout << "������ �о�� �� �����ϴ�" << endl;

	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	Mat Gblur_gray;
	GaussianBlur(gray, Gblur_gray, Size(3, 3), 0, 0);

	Scalar mean_bright = mean(Gblur_gray);		//��ϵ� ��հ����� ����ȭ �ϱ�����
	Mat binaryImg;
	threshold(Gblur_gray, binaryImg, mean_bright[0], 255, THRESH_BINARY_INV);

	Mat ptrX(1, binaryImg.cols, CV_32FC1, Scalar(0));	//x�� ���������ϱ� ���� ���
	for (int y = 0; y < binaryImg.rows; y++)	//�˻�����ؼ�
	{
		uchar* p = binaryImg.ptr<uchar>(y);
		for (int x = 0; x < binaryImg.cols; x++)
		{
			if (p[x] == 255)	ptrX.at<float>(0, x)++;		//��ü�� ������ �� ++
		}
	}
	Mat histImgptrX = getHistImg(ptrX);		//x�� �������� �׷���

	Point leftPoint[2], rightPoint[2];	//��ü 2���� �������, �����ϴ� ��ǥ
	int count1 = 0, count2 = 0;			//��ü ����
	for (int X = 0; X < binaryImg.cols; X++)
		if ((ptrX.at<float>(0, X) > 0) && (ptrX.at<float>(0, X - 1) == 0))
		{
			leftPoint[count1].x = X;	//���� ��� x��ǥ
			count1++;
		}
		else if((ptrX.at<float>(0, X) > 0) && (ptrX.at<float>(0, X + 1) == 0))
		{
			rightPoint[count2].x = X;	//���� ��� x��ǥ
			count2++;
		}
	count1--; count2--;		//���� ��ü���� �ϳ��� ī���� �Ǿ����Ƿ�

	Mat ptrY(1, binaryImg.rows, CV_32FC1, Scalar(0));	//y�� ���������ϱ� ���� ���
	for (int y = 0; y < binaryImg.rows; y++)
	{
		uchar* p = binaryImg.ptr<uchar>(y);
		for (int x = leftPoint[count1].x; x <= rightPoint[count2].x; x++)
		{
			if (p[x] == 255)	ptrY.at<float>(0, y)++;		//��ü�� ������ �� ++
		}
	}

	for (int Y = 0; Y < binaryImg.rows; Y++)
		if ((ptrY.at<float>(0, Y) > 0) && (ptrY.at<float>(0, Y - 1) == 0))
			leftPoint[count1].y = Y;	//y�� �������� �׷����� �˾Ƴ� ��ü�� �ֻ�� ��ǥ
		else if ((ptrY.at<float>(0, Y) > 0) && (ptrY.at<float>(0, Y + 1) == 0))
			rightPoint[count2].y = Y;	//��ü�� ���ϴ� ��ǥ

	rectangle(input, leftPoint[count1], rightPoint[count2], Scalar(0, 0, 255), 1);	//�ĺ�����
	int lengthX = rightPoint[count2].x - leftPoint[count1].x;	//x�� ����
	int lengthY = rightPoint[count2].y - leftPoint[count1].y;	//y�� ����
	int radius = (lengthX + lengthY) / 4;
	Point center((rightPoint[count2].x + leftPoint[count1].x) / 2, (rightPoint[count2].y + leftPoint[count1].y) / 2);
//	circle(input, center, radius, Scalar(0, 0, 255), 1);	//��
	circle(input, center, 2, Scalar(0, 0, 255), 2);			//�� �߽�

	int sizeMask = radius * 2 * 1.3;	//���ϱ� ���� �� ����ũ ������ (���� 1.3��)
	Mat circleMask = Mat::zeros(sizeMask, sizeMask, CV_8UC1);
	circle(circleMask, Point(sizeMask/2, sizeMask/2), radius / 2, Scalar(255), radius);

	//���� �̹������� �ĺ� ���� ����
	Mat resizeImg = binaryImg(Range(center.y - sizeMask / 2, center.y + sizeMask / 2+1),
		Range(center.x - sizeMask / 2, center.x + sizeMask / 2+1));

	//��
	//��ü ���� �߿��� �ձ� ���� ��ġ�� ���� => �ձ� ����
	Mat andImg, orImg;
	bitwise_and(circleMask, resizeImg, andImg);		// �ձ� ���� ��ġ�� ����
	bitwise_or(circleMask, resizeImg, orImg);		// ��ü ����

	double andCount = 0, orCount = 0;
	for (int y = 0; y < andImg.rows; y++)
	{
		uchar* pAnd = andImg.ptr<uchar>(y);
		uchar* pOr = orImg.ptr<uchar>(y);
		for (int x = 0; x < andImg.cols; x++)
		{
			if (pAnd[x] != 0)
				andCount++;
			if (pOr[x] != 0)
				orCount++;
		}
	}

	cout << "������ ��ü�� " << andCount / orCount * 100 << " % ������ ���̴�." << endl;

	imshow("input", input);
//	imshow("img", binaryImg);			//��ϵ� ��հ� ���� ����ȭ
//	imshow("histImgptrX", histImgptrX);	//x�� �������� �׷���
	imshow("circleMask", circleMask);	//���ϱ� ���� �� ����ũ		-(1)
	imshow("inputMask", resizeImg);		//������ ��ü ����ũ			-(2)
//	imshow("andImg", andImg);			//(1)�� (2) and ����
//	imshow("orImg", orImg);				//(1)�� (2) or  ����


	waitKey(0);
	return 0;
}
//*/

//4. LungCancer���� Lung Nodule�� ã�ƶ�.
/*

//���� ä���(ä�� ����, ��� ����, x1, x2, y1, y2)
Mat filling(Mat input, Mat output, int x1, int x2, int y1, int y2)
{
	Mat m = Mat::zeros(input.size(), CV_32FC1);

	for (int y = y1; y < y2; y++)
	{
		uchar* p = input.ptr<uchar>(y);

		for (int x = x1 + 1; x < x2; x++)			//���� ȭ�ҿ� ���� ȭ���� ���� �̿��Ͽ� ����ȭ ������ ��踦 ����.
			m.at<float>(y, x) = p[x] - p[x - 1];

		if (p[0] == 255)
			m.at<float>(y, 0) = 255;
	}
	for (int y = y1; y < y2; y++)
	{
		float* p = m.ptr<float>(y);
		for (int x = x1; x < x2; x++)
			if (p[x] > 0)
				for (int n = x + 1; n < x2; n++)
					if (p[n] > 0)
						for (int m = x; m <= n; m++)
							output.at<uchar>(y, m) =  255;
	}

	return output;
}


int main()
{
	Mat input = imread("q4.jpg");
	if (input.empty())	cout << "������ �о�� �� �����ϴ�" << endl;

	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	Mat binaryImg, binaryImgINV;
	threshold(gray, binaryImg, 10, 255, THRESH_BINARY);
	threshold(gray, binaryImgINV, 10, 255, THRESH_BINARY_INV);

	//�󿵿��� �����ؾ���.
	//�󿵿� ������ ���� binaryImg�� ä��� ����.
	//�� ���� ������ ������ �ּ�ȭ �ϱ� ���� ħ�� ��â ����
	Mat openImg;
	morphologyEx(binaryImg, openImg, MORPH_OPEN, Mat::ones(Size(5, 5), CV_8UC1));

	//x�� ä��� ����
	Mat fillImg;
	openImg.copyTo(fillImg);
	filling(openImg, fillImg, 0, binaryImg.cols, 0, binaryImg.rows);

	//fillImg�� binaryImgINV�� and�����Ͽ� �󿵿� ����
	Mat andImg;
	bitwise_and(fillImg, binaryImgINV, andImg);

	//����� �󿵿����� ���� ���� �����Ƿ�
	//�󿵿� �̹����� ����ó���Ͽ�
	//����� �ĺ������� ����
	Mat invImg;
	invImg = 255 - andImg;

	//�̹��� ������ 2��
	int multi = 2;
	Mat resizeImg;
	resize(invImg, resizeImg, Size(input.cols* multi, input.rows* multi));
	Mat dst;	//��� ����
	resize(input, dst, Size(input.cols * multi, input.rows * multi));

	//������ȯ�� ���� ��ó��
	Mat blurImg;
	blur(resizeImg, blurImg, Size(3, 3));

	Mat houghImg;
	blurImg.copyTo(houghImg);

	//������ȯ �� ����
	vector<Vec3f> circles;
	HoughCircles(houghImg, circles, HOUGH_GRADIENT, 1, 50, 100, 20, 0, 0);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center(c[0], c[1]);
		int radius = c[2];

		//�������� ũ�Ⱑ ���� �� �̻��� ��쿡�� ������ �Ǿ��ٰ� �Ǵ�.
		if (radius < 20)
		{
		//	cout << radius << endl;
		//	circle(dst, center, radius, Scalar(0, 255, 0), 1);
			circle(dst, center, 2, Scalar(0, 0, 255), 3);
			line(dst, Point(center.x - 10, center.y), Point(center.x + 10, center.y), Scalar(0, 0, 255), 3);
			line(dst, Point(center.x, center.y - 10), Point(center.x, center.y + 10), Scalar(0, 0, 255), 3);
		}
	}

//	imshow("binaryImg", binaryImg);			//����ȭ
//	imshow("binaryImgINV", binaryImgINV);	//����ȭ ���� ����
//	imshow("openImg", openImg);				//����ȭ ���� ä��� �� open����
//	imshow("fillImg", fillImg);				//ä���
//	imshow("andImg", andImg);				//�󿵿�
//	imshow("invImg", invImg);				//��� �ĺ�����
//	imshow("houghImg", houghImg);			//������ȯ�� �ϱ����� �̹���
	imshow("dst", dst);						//������ 2���Ͽ� ���� ��ġ ǥ���� ��� �̹���

	waitKey(0);
	return 0;
}
//*/

//5. ����� N�� ������ ã�ƶ�.
/*
int main()
{
	Mat input = imread("q5.jpg");
	if (input.empty())	cout << "������ �о�� �� �����ϴ�" << endl;

	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	Mat equalizeImg;
	equalizeHist(gray, equalizeImg);

	Mat sobelImg, absSobelImg;
	int scale_sobel = 1, delta_sobel = 0;
	Sobel(equalizeImg, sobelImg, CV_16S, 0, 1, 3, scale_sobel, delta_sobel, BORDER_DEFAULT);
	convertScaleAbs(sobelImg, absSobelImg);

	Mat addImg;
	add(absSobelImg, gray, addImg);

	Mat binaryAdaptiveImg;
	int blockSize = 21;		// �̿� ũ��
	int thre = - 15;	//ȭ�Ҹ� (���-��� ��)�� ��
	adaptiveThreshold(addImg, // �Է¿���
		binaryAdaptiveImg,	// ����ȭ ��� ����
		255,				// �ִ� ȭ�� ��
		ADAPTIVE_THRESH_MEAN_C,		// Adaptive �Լ�
		THRESH_BINARY,		// ����ȭ Ÿ��
		blockSize,			// �̿�ũ��
		thre);			// threshold used
	imshow("binaryAdaptiveImg", binaryAdaptiveImg);

	Mat threSobelImg;
	threshold(absSobelImg, threSobelImg, 40, 255, THRESH_BINARY);

//	Mat openImg;
//	morphologyEx(threSobelImg, openImg, MORPH_OPEN, Mat::ones(Size(3, 3), CV_8UC1));

	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(cannyImg, img_labels, stats, centroids, 8, CV_32S);

	// ���̺� ����� �簢�� �׸���, �ѹ� ǥ���ϱ�
	for (int j = 1; j < numOfLables; j++) {
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);


	//	rectangle(input, Point(left, top), Point(left + width, top + height),
	//		Scalar(0, 0, 255), 1);

	//	putText(input, to_string(j), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1);
	}

	imshow("input", input);
	imshow("equalizeImg", equalizeImg);
	imshow("sobelImg", absSobelImg);
	imshow("binaryAdaptiveImg", binaryAdaptiveImg);
	imshow("addImg", addImg);
	imshow("cannyImg", cannyImg);

	imshow("threSobelImg", threSobelImg);
	waitKey(0);
	return 0;
}
//*/

//6. Detecting Bifurcations
/*
Mat getHistImg(const Mat& hist)		//������׷� �̹���
{
	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat histImg(230, hist.cols * 20, CV_8UC3, Scalar(255, 255, 255));

	int n = 0;
	for (int i = 0; i < hist.cols * 20; i++)
	{
		if ((i % 20 == 0) && (i != 0))
			n++;
		if (i % 20 != 19)
			line(histImg,
				Point(i, 200),
				Point(i, 200 - cvRound((hist.at<int>(0, n) / histMax) * 200)),
				Scalar(220, 179, 113));
	}
	for (int i = 1; i < hist.cols + 1; i++)
		putText(histImg, to_string(i), Point((i * 20) - 20, 215), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0), 1);


	return histImg;
}

int main()
{
	Mat input = imread("q6.jpg");
	if (input.empty())	cout << "������ �о�� �� �����ϴ�" << endl;

	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	//������ threshold�� ����ȭ ���� ��� ���� ���� ��Ⱚ ���̷� ���� �߸��� ������ �����.
	//���� ��������ȭ�� ������.
	Mat binaryAdaptiveImg;
	int blockSize = 21;		// �̿� ũ�� 
	int threshold = -10;	//ȭ�Ҹ� (���-��� ��)�� �� 
	adaptiveThreshold(gray, // �Է¿��� 
		binaryAdaptiveImg,	// ����ȭ ��� ���� 
		255,				// �ִ� ȭ�� �� 
		ADAPTIVE_THRESH_MEAN_C,		// Adaptive �Լ�
		THRESH_BINARY,		// ����ȭ Ÿ�� 
		blockSize,			// �̿�ũ�� 
		threshold);			// threshold used

	Mat BAImg;
	binaryAdaptiveImg.copyTo(BAImg);

	//����ȭ ������ �ݺ������� ħ���Ͽ� ���ȭ ������ ����.
	int rows = 3;
	int cols = 3;
	char data[] = { 0,1,0, 1,1,1, 0,1,0 };
	Mat element(rows, cols, CV_8UC1, data);
	Mat erodedImg, tempImg;
	Mat skelImg(input.size(), CV_8UC1, Scalar(0));
	do
	{
		erode(binaryAdaptiveImg, erodedImg, element);
		dilate(erodedImg, tempImg, element);
		subtract(binaryAdaptiveImg, tempImg, tempImg);
		bitwise_or(skelImg, tempImg, skelImg);
		erodedImg.copyTo(binaryAdaptiveImg);
	} while ((countNonZero(binaryAdaptiveImg) != 0));


	//�ڳ� ����⸦ �̿��� Bifurcations ���� 
	Mat reverseImg = ~skelImg.clone();
	Mat circleImg;
	cvtColor(reverseImg, circleImg, COLOR_GRAY2BGR);

	//GFTT�� �̿��� �ڳ� ���� 
	vector<Point2f> corners;
	goodFeaturesToTrack(skelImg, corners, 1000, 0.05, 8);

	//���� ���� ī���� �ϱ� ���� ��
	Mat bifurcationsCountingMap = Mat::zeros(Size(input.size()), CV_32SC1);


	Mat temp2Img, andImg;
	for (int i = 0; i < corners.size(); i++)
	{
		//�ڳʷ� ����� ���� ������ 3�� ���� �׸�
		//������ ������ũ�� ���� ������ and ���� ��. => ������ ���� ������ ã��
		temp2Img = Mat::zeros(Size(input.size()), CV_8UC1);
		circle(temp2Img, corners[i], 3, Scalar(255));
		bitwise_and(temp2Img, skelImg, andImg);

		//���� �ϳ��� ���� �����ӿ��� �ȼ��� 2���� ����Ǵ� ��쵵 ����
		//���� �ȼ������� ������ ī�������� �ʰ�, �󺧸����� ī������. 
		Mat img_labels, stats, centroids;
		int numOfLables = connectedComponentsWithStats(andImg, img_labels, stats, centroids, 8, CV_32S);
		bifurcationsCountingMap.at<int>(corners[i]) = numOfLables;
		if (numOfLables > 2)
			circle(circleImg, corners[i], 4, Scalar(0, 0, 255));
	}

	//������׷� �׸���
	Mat bifurcationsHist = Mat::zeros(Size(10, 1), CV_32SC1);
	for (int y = 0; y < input.rows; y++)
	{
		int* p = bifurcationsCountingMap.ptr<int>(y);
		for (int x = 0; x < input.cols; x++)
		{
			if (p[x] != 0 && p[x] != 1)
				bifurcationsHist.at<int>(0, p[x])++;	//����
		}
	}

	//������׷� �׸���
	cout << bifurcationsHist << endl;
	Mat histImg = getHistImg(bifurcationsHist);

	imshow("input", input);					//�Է� ����
	imshow("binaryAdaptiveImg", BAImg);		//��������ȭ 
	imshow("skelImg", skelImg);				//���̷���			--(1)
	imshow("circleImg", circleImg);			//����� �ڳ� 
//	imshow("temp2Img", temp2Img);			//������ ������ũ	--(2)
//	imshow("andImg", andImg);				//(1)�� (2) and ����
	imshow("histImg", histImg);				//������׷� 


	waitKey(0);
	return 0;
}
//*/

//7. Hemorrhage�� ã�ƶ�.
/*
int main()
{
	Mat input = imread("q7.jpg");
	if (input.empty())	cout << "������ �о�� �� �����ϴ�" << endl;

	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);


	Mat blurImg;
	blur(gray, blurImg, Size(3, 3));

	Mat thresholdImg;
	threshold(blurImg, thresholdImg, 130, 255, THRESH_BINARY);

	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(thresholdImg, img_labels, stats, centroids, 8, CV_32S);


	int memoryLeft[2], memoryTop[2], memoryWidth[2], memoryHeight[2];
	int count = 0;
	// ���̺� ����� �簢�� �׸���, �ѹ� ǥ���ϱ�
	for (int j = 1; j < numOfLables; j++)
	{
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);

	//	cout << j <<" : " << area << endl;	//������ �� �κ��� ���� �ϱ� ���� ���� Ȯ��
		
		if (area > 400)
		{
			memoryLeft[count] = left;
			memoryTop[count] = top;
			memoryWidth[count] = width;
			memoryHeight[count] = height;
			count++;
		}
		else
		{
			rectangle(input, Point(left, top), Point(left + width, top + height),
				Scalar(0, 0, 255), 1);
			putText(input, to_string(j), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1);

		}

	}


	for (int n = 0; n < count; n++)
		for (int y = memoryTop[n]; y < memoryTop[n] + memoryHeight[n]; y++)
		{
			for (int x = memoryLeft[n]; x < memoryLeft[n] + memoryWidth[n]; x++)
				thresholdImg.at<uchar>(y, x) = 0;
		}

	imshow("input", input);
	imshow("rectangleImg", gray);
	imshow("output", thresholdImg);

	waitKey(0);
	return 0;
}

//*/
