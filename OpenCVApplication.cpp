// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
using namespace cv;

Mat RGB2GrayScale(Mat_<Vec3b> img)
{
	int width = img.cols;
	int height = img.rows;
	Mat_<uchar> dst1(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b channel = img(i, j);
			dst1(i, j) = float(channel[2] + channel[1] + channel[0]) / 3.;
		}
	}
	//imshow("Src", img);
	//imshow("Dst", dst1);
	//waitKey();

	return dst1;
}

bool isInside(Mat img, int i, int j)
{
	int width = img.cols;
	int height = img.rows;

	if (i < height && j < width && i >= 0 && j >= 0)
		return true;
	else
		return false;
}

Mat_<float> convolution(Mat_<uchar> img, Mat_<float> kernel) {

	Mat_<float> result(img.rows, img.cols);
	float sum = 0;
	float sum_neg = 0;

	for (int u = 0; u < kernel.rows; u++) {
		for (int v = 0; v < kernel.cols; v++) {
			if (kernel(u, v) > 0)
			{
				sum += kernel(u, v);
			}
			else
			{
				sum_neg += kernel(u, v);
			}
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			float temp = 0;
			for (int u = 0; u < kernel.rows; u++) {
				for (int v = 0; v < kernel.cols; v++) {
					int new_i = i + u - (kernel.rows / 2);
					int new_j = j + v - (kernel.cols / 2);
					int new_val;
					if (!isInside(img, new_i, new_j))
					{
						new_val = 0;
					}
					else
					{
						new_val = img(new_i, new_j);
					}

					temp += new_val * kernel(u, v);
				}
			}

			//temp = abs(temp);
			float res_val = temp;

			result(i, j) = (float)res_val;
		}
	}

	return result;
}

Mat_<float> gradientX(Mat_<uchar> img)
{
	Mat_<float> Gx(3, 3);
	Gx(0, 0) = -1.;
	Gx(0, 1) = 0.;
	Gx(0, 2) = 1.;
	Gx(1, 0) = -2.;
	Gx(1, 1) = 0.;
	Gx(1, 2) = 2.;
	Gx(2, 0) = -1.;
	Gx(2, 1) = 0.;
	Gx(2, 2) = 1.;

	return convolution(img, Gx);

}

Mat_<float> gradientY(Mat_<uchar> img)
{
	Mat_<float> Gy(3, 3);
	Gy(0, 0) = 1.;
	Gy(0, 1) = 2.;
	Gy(0, 2) = 1.;
	Gy(1, 0) = 0.;
	Gy(1, 1) = 0.;
	Gy(1, 2) = 0.;
	Gy(2, 0) = -1.;
	Gy(2, 1) = -2.;
	Gy(2, 2) = -1.;

	return convolution(img, Gy);
}

Mat_<float> energyMap(Mat_<Vec3b> img)
{
	Mat_<uchar> src = RGB2GrayScale(img);
	return (abs(gradientX(src)) + abs(gradientY(src))) / 255;
}

enum direction { vertical, horizontal };
Mat cumulativeEnergyMap(Mat_<float> energyMap, direction dir)
{
	int height = energyMap.rows;
	int width = energyMap.cols;

	Mat_<float> cumulativeEnergy(height, width);

	if (dir == vertical) //copy first line
	{
		for (int j = 0; j < width; j++)
		{
			cumulativeEnergy(0, j) = energyMap(0, j);
		}
	}

	if (dir == horizontal) //copy first column
	{
		for (int i = 0; i < height; i++)
		{
			cumulativeEnergy(i, 0) = energyMap(i, 0);
		}
	}

	if (dir == vertical)
	{
		for (int i = 1; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float upperFirst = cumulativeEnergy(i - 1, max(j - 1, 0)); // max - in case of out of bounds
				float upperSecond = cumulativeEnergy(i - 1, j);
				float upperThird = cumulativeEnergy(i - 1, min(j + 1, width - 1)); //min - in case of out of bounds

				//Cumulative minimum energy M, calculation
				cumulativeEnergy(i, j) = energyMap(i, j) + min(upperFirst, min(upperSecond, upperThird));
			}
		}
	}

	if (dir == horizontal)
	{
		for (int j = 1; j < width; j++)
		{
			for (int i = 0; i < height; i++)
			{
				float firstLeft = cumulativeEnergy(max(i - 1, 0), j - 1); // max - in case of out of bounds
				float secondLeft = cumulativeEnergy(i, j - 1);
				float thirdLeft = cumulativeEnergy(min(i + 1, height - 1), j - 1); //min - in case of out of bounds

				//Cumulative minimum energy M, calculation
				cumulativeEnergy(i, j) = energyMap(i, j) + min(firstLeft, min(secondLeft, thirdLeft));
			}
		}
	}

	/*for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << energyMap(i, j) << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
	std::cout << std::endl;

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << cumulativeEnergy(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/

	return cumulativeEnergy / 255;
}

std::vector<std::pair<int, int>> findSeam(Mat_<float> cumul, direction dir)
{
	std::pair<int, int> current;
	std::vector<std::pair<int, int>> path;
	int height = cumul.rows;
	int width = cumul.cols;

	if (dir == vertical)
	{
		float min = cumul(height - 1, 0);
		std::pair<int, int> minLoc;
		minLoc.first = height - 1;
		minLoc.second = 0;
		
		for (int j = 1; j < width; j++) //finding minimum on last row and its location
		{
			if (cumul(height - 1, j) < min)
			{
				min = cumul(height - 1, j);
				minLoc.first = height - 1;
				minLoc.second = j;
			}
		}
		path.push_back(minLoc);

		for (int i = height - 2; i >= 0; i--) //finding the minimum of the above 3 pixels and adding to path
		{
			float first = cumul(i, max(minLoc.second - 1, 0));
			float second = cumul(i, minLoc.second);
			float third = cumul(i, min(minLoc.second + 1, width - 1));

			min = min(first, min(second, third));
			minLoc.first = i;
			if (min == first) minLoc.second = max(minLoc.second - 1, 0);
			else if (min == second) minLoc.second = minLoc.second;
			else minLoc.second = min(minLoc.second + 1, width - 1);

			path.push_back(minLoc);
		}
	}
	
	if (dir == horizontal)
	{
		float min = cumul(0, width - 1);
		std::pair<int, int> minLoc;
		minLoc.first = 0;
		minLoc.second = width - 1;

		for (int i = 1; i < height; i++) //finding minimum on last column and its location
		{
			if (cumul(i, width - 1) < min)
			{
				min = cumul(i, width - 1);
				minLoc.first = i;
				minLoc.second = width - 1;
			}
		}
		path.push_back(minLoc);

		for (int i = width - 2; i >= 0; i--) //finding the minimum of the 3 pixels to the left of the current one and add it to path
		{
			float first = cumul(max(minLoc.first - 1, 0), i);
			float second = cumul(minLoc.first, i);
			float third = cumul(min(minLoc.first + 1, height - 1), i);

			min = min(first, min(second, third));
			minLoc.second = i;
			if (min == first) minLoc.first = max(minLoc.first - 1, 0);
			else if (min == second) minLoc.first = minLoc.first;
			else minLoc.first = min(minLoc.first + 1, height - 1);

			path.push_back(minLoc);
		}
	}

	/*for (int i = 0; i < path.size(); i++)
	{
	std::cout << path[i].first << ","<< path[i].second << std::endl;
	}*/

	return path; //return path of pixel locations as a vector
}

Mat deleteSeamPath(std::vector<std::pair<int, int>> path, Mat_<Vec3b> source, direction dir)
{
	if (dir == vertical) 
	{
		Mat_<Vec3b> result(source.rows, source.cols - 1);

		for (int i = 0; i < source.rows; i++)
		{
			for (int j = path[i].second; j < source.cols - 1; j++)
			{
				source(i, j) = source(i, j + 1);

			}
		}

		for (int i = 0; i < source.rows; i++)
		{
			for (int j = 0; j < source.cols - 1; j++)
			{
				result(i, j) = source(i, j);
			}
		}

		return result;
	}

	if (dir == horizontal)
	{
		Mat_<Vec3b> result(source.rows - 1, source.cols);

		for (int i = 0; i < source.cols; i++)
		{
			for (int j = path[i].first; j < source.rows - 1; j++)
			{
				source(j, i) = source(j + 1, i);
			}
		}

		for (int i = 0; i < source.rows - 1; i++)
		{
			for (int j = 0; j < source.cols; j++)
			{
				result(i, j) = source(i, j);
			}
		}

		return result;
	}
	
}

Mat drawSeam(std::vector<std::pair<int, int>> path, Mat_<Vec3b> source)
{
	Mat_<Vec3b> result = source.clone();

	for (int i = path.size() - 1; i >= 0; i--)
	{
		int sourceLoci = path[i].first;
		int sourceLocj = path[i].second;

		result(sourceLoci, sourceLocj) = Vec3b(255, 255, 255); //RED IS NOT GOOD BECAUSE OF CONVERSION TO GRAYSCALE 
	}

	return result;
}

int main()
{
	char fname[MAX_PATH];
	int iterations;
	int option;
	direction dir;

	std::cout << "1 -> Vertical\n2 -> Horizontal\n";
	label1: std::cin >> option;
	if (option == 1) dir = vertical;
	else if (option == 2) dir = horizontal;
	else 
	{
		std::cout << "Choose 1 or 2!\n"; goto label1;
	}
	std::cout << "No. of iterations ? \n";
	std::cin >> iterations;

	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src;
		src = imread(fname);
		Mat_<Vec3b> copy = src.clone();

		for (int i = 0; i < iterations; i++)
		{
			Mat_<float> energy = abs(energyMap(copy)); 
			Mat_<float> cumul = cumulativeEnergyMap(energy, dir);

			//copy = deleteSeamPath(findSeam(cumul, dir), copy, dir);
			copy = drawSeam(findSeam(cumul, dir), copy); //draws multiple seams ok because of high energy calculated of white seam
		}

		imshow("source", src);
		imshow("seamify", copy);
		imwrite("result.jpg", copy);
		waitKey(0);
	}
}