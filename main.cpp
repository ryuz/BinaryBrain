#include <iostream>
#include "NeuralNet.h"
#include "NeuralNetAffine.h"
#include "NeuralNetSigmoid.h"
#include "NeuralNetSoftmax.h"


int main()
{
	NeuralNet<> net;
	NeuralNetAffine<> affine0(28*28, 50);
	NeuralNetSigmoid<> sigmoid0(50);
	NeuralNetAffine<> affine1(50, 10);
	NeuralNetSoftmax<> softmax1(10);

	net.AddLayer(&affine0);
	net.AddLayer(&sigmoid0);
	net.AddLayer(&affine1);
	net.AddLayer(&softmax1);
	net.SetBatchSize(100);



	return 0;
}


#if 0

#include <windows.h>
#include <tchar.h>
#pragma comment(lib, "winmm.lib")

#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <utility>
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
#include "mnist_read.h"
#include "ShuffleSet.h"


int main()
{
#if 0
	Eigen::MatrixXf a(2, 1);
	Eigen::MatrixXf b(1, 2);
	a(0, 0) = 1;
//	a(0, 1) = 2;
	a(1, 0) = 3;
//	a(1, 1) = 4;
	b(0, 0) = 10;
	b(0, 1) = 20;
	Eigen::MatrixXf c = a * b;

	std::cout << "A = \n" << a << std::endl;
	std::cout << "B = \n" << b << std::endl;
	std::cout << "C = \n" << c << std::endl;
#endif

#if 1
	float a[4] = { 1,2,3,4 };
	float b[4] = { 10,20,30,40 };
	float c[4] = { 0, 0, 0, 0 };
	float d[2] = { 1000, 2000 };

	Eigen::Matrix<float, -1, -1, Eigen::RowMajor> mat;
//	Eigen::Matrix<float, -1, -1> mat;


	Eigen::Map< Eigen::Matrix<float, -1, -1, Eigen::RowMajor> > mat_a(a, 2, 2);
	Eigen::Map<Eigen::MatrixXf> mat_b(b, 2, 2);
	Eigen::Map<Eigen::MatrixXf> mat_c(c, 2, 2);
	Eigen::Map<Eigen::VectorXf> mat_d(d, 2);

	mat_c = mat_a * mat_b;
	std::cout << "A = \n" << mat_a << std::endl;
	std::cout << "B = \n" << mat_b << std::endl;
	std::cout << "C = \n" << mat_c << std::endl;
	std::cout << "D = \n" << mat_d << std::endl;
	mat_c.colwise() += mat_d;
	std::cout << "C = \n" << mat_c << std::endl;

	std::cout << "c =" << std::endl;
	std::cout << c[0] << std::endl;
	std::cout << c[1] << std::endl;
	std::cout << c[2] << std::endl;
	std::cout << c[3] << std::endl;
#endif

	getchar();

	return 0;
}

#endif