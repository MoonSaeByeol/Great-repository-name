#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>


#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


//1. 구구단을 구현하라.
/*
int main()
{
	for (int i = 1; i < 10; i++)
	{
		int x = i % 3;
		cout << i << "단\t";
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


//2. GLCM Matrix 구현하라.
// 그레이 레벨 이미지가 주어졌을 때
// 동시 발생 행렬은 이미지에서 특정 값과 오프셋이 있는 픽셀 쌍이 
// 얼마나 자주 발생하는지 계산
// 따라서 GLCM의 크기는 픽셀값의 최댓값이 될 것.
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

//3. 동그란 정도를 평가하라.
/*
Mat getHistImg(const Mat& hist)		//히스토그램 이미지
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
	if (input.empty())	cout << "영상을 읽어올 수 없습니다" << endl;

	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	Mat Gblur_gray;
	GaussianBlur(gray, Gblur_gray, Size(3, 3), 0, 0);

	Scalar mean_bright = mean(Gblur_gray);		//명암도 평균값으로 이진화 하기위해
	Mat binaryImg;
	threshold(Gblur_gray, binaryImg, mean_bright[0], 255, THRESH_BINARY_INV);

	Mat ptrX(1, binaryImg.cols, CV_32FC1, Scalar(0));	//x축 프로젝션하기 위한 행렬
	for (int y = 0; y < binaryImg.rows; y++)	//검사시작해서
	{
		uchar* p = binaryImg.ptr<uchar>(y);
		for (int x = 0; x < binaryImg.cols; x++)
		{
			if (p[x] == 255)	ptrX.at<float>(0, x)++;		//객체가 존재할 때 ++
		}
	}
	Mat histImgptrX = getHistImg(ptrX);		//x축 프로젝션 그래프

	Point leftPoint[2], rightPoint[2];	//객체 2개의 좌측상단, 우측하단 좌표
	int count1 = 0, count2 = 0;			//객체 개수
	for (int X = 0; X < binaryImg.cols; X++)
		if ((ptrX.at<float>(0, X) > 0) && (ptrX.at<float>(0, X - 1) == 0))
		{
			leftPoint[count1].x = X;	//좌측 경계 x좌표
			count1++;
		}
		else if((ptrX.at<float>(0, X) > 0) && (ptrX.at<float>(0, X + 1) == 0))
		{
			rightPoint[count2].x = X;	//우측 경계 x좌표
			count2++;
		}
	count1--; count2--;		//실제 객체보다 하나더 카운팅 되었으므로

	Mat ptrY(1, binaryImg.rows, CV_32FC1, Scalar(0));	//y축 프로젝션하기 위한 행렬
	for (int y = 0; y < binaryImg.rows; y++)
	{
		uchar* p = binaryImg.ptr<uchar>(y);
		for (int x = leftPoint[count1].x; x <= rightPoint[count2].x; x++)
		{
			if (p[x] == 255)	ptrY.at<float>(0, y)++;		//객체가 존재할 때 ++
		}
	}

	for (int Y = 0; Y < binaryImg.rows; Y++)
		if ((ptrY.at<float>(0, Y) > 0) && (ptrY.at<float>(0, Y - 1) == 0))
			leftPoint[count1].y = Y;	//y축 프로젝션 그래프로 알아낸 객체의 최상단 좌표
		else if ((ptrY.at<float>(0, Y) > 0) && (ptrY.at<float>(0, Y + 1) == 0))
			rightPoint[count2].y = Y;	//객체의 최하단 좌표

	rectangle(input, leftPoint[count1], rightPoint[count2], Scalar(0, 0, 255), 1);	//후보영역
	int lengthX = rightPoint[count2].x - leftPoint[count1].x;	//x축 길이
	int lengthY = rightPoint[count2].y - leftPoint[count1].y;	//y축 길이
	int radius = (lengthX + lengthY) / 4;
	Point center((rightPoint[count2].x + leftPoint[count1].x) / 2, (rightPoint[count2].y + leftPoint[count1].y) / 2);
//	circle(input, center, radius, Scalar(0, 0, 255), 1);	//원
	circle(input, center, 2, Scalar(0, 0, 255), 2);			//원 중심

	int sizeMask = radius * 2 * 1.3;	//비교하기 위한 원 마스크 사이즈 (원의 1.3배)
	Mat circleMask = Mat::zeros(sizeMask, sizeMask, CV_8UC1);
	circle(circleMask, Point(sizeMask/2, sizeMask/2), radius / 2, Scalar(255), radius);

	//원본 이미지에서 후보 영역 추출
	Mat resizeImg = binaryImg(Range(center.y - sizeMask / 2, center.y + sizeMask / 2+1),
		Range(center.x - sizeMask / 2, center.x + sizeMask / 2+1));

	//비교
	//전체 영역 중에서 둥근 원과 겹치는 영역 => 둥근 정도
	Mat andImg, orImg;
	bitwise_and(circleMask, resizeImg, andImg);		// 둥근 원과 겹치는 영역
	bitwise_or(circleMask, resizeImg, orImg);		// 전체 영역

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

	cout << "오른쪽 객체는 " << andCount / orCount * 100 << " % 원형일 것이다." << endl;

	imshow("input", input);
//	imshow("img", binaryImg);			//명암도 평균값 기준 이진화
//	imshow("histImgptrX", histImgptrX);	//x축 프로젝션 그래프
	imshow("circleMask", circleMask);	//비교하기 위한 원 마스크		-(1)
	imshow("inputMask", resizeImg);		//오른쪽 객체 마스크			-(2)
//	imshow("andImg", andImg);			//(1)과 (2) and 연산
//	imshow("orImg", orImg);				//(1)과 (2) or  연산


	waitKey(0);
	return 0;
}
//*/

//4. LungCancer에서 Lung Nodule를 찾아라.
/*

//영상 채우기(채울 영상, 결과 영상, x1, x2, y1, y2)
Mat filling(Mat input, Mat output, int x1, int x2, int y1, int y2)
{
	Mat m = Mat::zeros(input.size(), CV_32FC1);

	for (int y = y1; y < y2; y++)
	{
		uchar* p = input.ptr<uchar>(y);

		for (int x = x1 + 1; x < x2; x++)			//이전 화소와 현재 화소의 차를 이용하여 이진화 영상의 경계를 구함.
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
	if (input.empty())	cout << "영상을 읽어올 수 없습니다" << endl;

	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	Mat binaryImg, binaryImgINV;
	threshold(gray, binaryImg, 10, 255, THRESH_BINARY);
	threshold(gray, binaryImgINV, 10, 255, THRESH_BINARY_INV);

	//폐영역을 검출해야함.
	//폐영역 검출을 위해 binaryImg의 채우기 진행.
	//그 전에 잡음의 영향을 최소화 하기 위해 침식 팽창 연산
	Mat openImg;
	morphologyEx(binaryImg, openImg, MORPH_OPEN, Mat::ones(Size(5, 5), CV_8UC1));

	//x축 채우기 진행
	Mat fillImg;
	openImg.copyTo(fillImg);
	filling(openImg, fillImg, 0, binaryImg.cols, 0, binaryImg.rows);

	//fillImg와 binaryImgINV의 and연산하여 폐영역 추출
	Mat andImg;
	bitwise_and(fillImg, binaryImgINV, andImg);

	//폐암은 폐영역보다 밝은 값을 가지므로
	//폐영역 이미지를 반전처리하여
	//폐암의 후보영역을 설정
	Mat invImg;
	invImg = 255 - andImg;

	//이미지 사이즈 2배
	int multi = 2;
	Mat resizeImg;
	resize(invImg, resizeImg, Size(input.cols* multi, input.rows* multi));
	Mat dst;	//결과 영상
	resize(input, dst, Size(input.cols * multi, input.rows * multi));

	//허프변환을 위한 블러처리
	Mat blurImg;
	blur(resizeImg, blurImg, Size(3, 3));

	Mat houghImg;
	blurImg.copyTo(houghImg);

	//허프변환 원 검출
	vector<Vec3f> circles;
	HoughCircles(houghImg, circles, HOUGH_GRADIENT, 1, 50, 100, 20, 0, 0);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center(c[0], c[1]);
		int radius = c[2];

		//반지름의 크기가 일정 값 이상일 경우에는 오검출 되었다고 판단.
		if (radius < 20)
		{
		//	cout << radius << endl;
		//	circle(dst, center, radius, Scalar(0, 255, 0), 1);
			circle(dst, center, 2, Scalar(0, 0, 255), 3);
			line(dst, Point(center.x - 10, center.y), Point(center.x + 10, center.y), Scalar(0, 0, 255), 3);
			line(dst, Point(center.x, center.y - 10), Point(center.x, center.y + 10), Scalar(0, 0, 255), 3);
		}
	}

//	imshow("binaryImg", binaryImg);			//이진화
//	imshow("binaryImgINV", binaryImgINV);	//이진화 반전 영상
//	imshow("openImg", openImg);				//이진화 영상 채우기 전 open연산
//	imshow("fillImg", fillImg);				//채우기
//	imshow("andImg", andImg);				//폐영역
//	imshow("invImg", invImg);				//폐암 후보영역
//	imshow("houghImg", houghImg);			//허프변환을 하기위한 이미지
	imshow("dst", dst);						//사이즈 2배하여 암을 위치 표시할 결과 이미지

	waitKey(0);
	return 0;
}
//*/

//5. 갈비뼈 N개 정도를 찾아라.
/*
int main()
{
	Mat input = imread("q5.jpg");
	if (input.empty())	cout << "영상을 읽어올 수 없습니다" << endl;

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
	int blockSize = 21;		// 이웃 크기
	int thre = - 15;	//화소를 (평균-경계 값)과 비교
	adaptiveThreshold(addImg, // 입력영상
		binaryAdaptiveImg,	// 이진화 결과 영상
		255,				// 최대 화소 값
		ADAPTIVE_THRESH_MEAN_C,		// Adaptive 함수
		THRESH_BINARY,		// 이진화 타입
		blockSize,			// 이웃크기
		thre);			// threshold used
	imshow("binaryAdaptiveImg", binaryAdaptiveImg);

	Mat threSobelImg;
	threshold(absSobelImg, threSobelImg, 40, 255, THRESH_BINARY);

//	Mat openImg;
//	morphologyEx(threSobelImg, openImg, MORPH_OPEN, Mat::ones(Size(3, 3), CV_8UC1));

	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(cannyImg, img_labels, stats, centroids, 8, CV_32S);

	// 레이블링 결과에 사각형 그리고, 넘버 표시하기
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
Mat getHistImg(const Mat& hist)		//히스토그램 이미지
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
	if (input.empty())	cout << "영상을 읽어올 수 없습니다" << endl;

	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	//고정된 threshold로 이진화 했을 경우 영상 내의 밝기값 차이로 인해 잘못된 영상을 출력함.
	//따라서 지역이진화를 실행함.
	Mat binaryAdaptiveImg;
	int blockSize = 21;		// 이웃 크기 
	int threshold = -10;	//화소를 (평균-경계 값)과 비교 
	adaptiveThreshold(gray, // 입력영상 
		binaryAdaptiveImg,	// 이진화 결과 영상 
		255,				// 최대 화소 값 
		ADAPTIVE_THRESH_MEAN_C,		// Adaptive 함수
		THRESH_BINARY,		// 이진화 타입 
		blockSize,			// 이웃크기 
		threshold);			// threshold used

	Mat BAImg;
	binaryAdaptiveImg.copyTo(BAImg);

	//이진화 영상을 반복적으로 침식하여 골격화 영상을 만듦.
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


	//코너 검출기를 이용한 Bifurcations 검출 
	Mat reverseImg = ~skelImg.clone();
	Mat circleImg;
	cvtColor(reverseImg, circleImg, COLOR_GRAY2BGR);

	//GFTT를 이용한 코너 검출 
	vector<Point2f> corners;
	goodFeaturesToTrack(skelImg, corners, 1000, 0.05, 8);

	//가지 개수 카운팅 하기 위한 맵
	Mat bifurcationsCountingMap = Mat::zeros(Size(input.size()), CV_32SC1);


	Mat temp2Img, andImg;
	for (int i = 0; i < corners.size(); i++)
	{
		//코너로 검출된 점에 반지름 3의 원을 그림
		//각각의 원마스크와 가지 영상을 and 연산 함. => 가지와 원의 교점을 찾음
		temp2Img = Mat::zeros(Size(input.size()), CV_8UC1);
		circle(temp2Img, corners[i], 3, Scalar(255));
		bitwise_and(temp2Img, skelImg, andImg);

		//라인 하나와 원의 교점임에도 픽셀은 2개씩 검출되는 경우도 존재
		//따라서 픽셀단위로 교점을 카운팅하지 않고, 라벨링으로 카운팅함. 
		Mat img_labels, stats, centroids;
		int numOfLables = connectedComponentsWithStats(andImg, img_labels, stats, centroids, 8, CV_32S);
		bifurcationsCountingMap.at<int>(corners[i]) = numOfLables;
		if (numOfLables > 2)
			circle(circleImg, corners[i], 4, Scalar(0, 0, 255));
	}

	//히스토그램 그리기
	Mat bifurcationsHist = Mat::zeros(Size(10, 1), CV_32SC1);
	for (int y = 0; y < input.rows; y++)
	{
		int* p = bifurcationsCountingMap.ptr<int>(y);
		for (int x = 0; x < input.cols; x++)
		{
			if (p[x] != 0 && p[x] != 1)
				bifurcationsHist.at<int>(0, p[x])++;	//누적
		}
	}

	//히스토그램 그리기
	cout << bifurcationsHist << endl;
	Mat histImg = getHistImg(bifurcationsHist);

	imshow("input", input);					//입력 영상
	imshow("binaryAdaptiveImg", BAImg);		//지역이진화 
	imshow("skelImg", skelImg);				//스켈레톤			--(1)
	imshow("circleImg", circleImg);			//검출된 코너 
//	imshow("temp2Img", temp2Img);			//각각의 원마스크	--(2)
//	imshow("andImg", andImg);				//(1)과 (2) and 연산
	imshow("histImg", histImg);				//히스토그램 


	waitKey(0);
	return 0;
}
//*/

//7. Hemorrhage를 찾아라.
/*
int main()
{
	Mat input = imread("q7.jpg");
	if (input.empty())	cout << "영상을 읽어올 수 없습니다" << endl;

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
	// 레이블링 결과에 사각형 그리고, 넘버 표시하기
	for (int j = 1; j < numOfLables; j++)
	{
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);

	//	cout << j <<" : " << area << endl;	//오검출 된 부분을 제거 하기 위해 넓이 확인
		
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
