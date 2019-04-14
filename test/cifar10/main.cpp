// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <omp.h>
#include <string.h>




void Cifar10LutMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void Cifar10LutCnn(int epoch_size, size_t mini_batch_size, bool binary_mode);
void Cifar10DenseCnn(int epoch_size, size_t mini_batch_size, bool binary_mode);
void Cifar10DenseMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void Cifar10StochasticLut6Cnn(int epoch_size, size_t mini_batch_size, bool binary_mode);

void DataTest(void);


// メイン関数
int main(int argc, char *argv[])
{
//  DataTest();

    std::string netname = "All";
    int         epoch_size      = 16;
    int         mini_batch_size = 32;
    bool        binary_mode = true;

	if ( argc < 2 ) {
        std::cout << "usage:" << std::endl;
        std::cout << argv[0] << " [options] <netname>" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "options" << std::endl;
        std::cout << "  -epoch      <epoch size>        set epoch size" << std::endl;
        std::cout << "  -mini_batch <mini_batch size>   set mini batch size" << std::endl;
        std::cout << "  -binary     <0|1>               set binary mode" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "netname" << std::endl;
        std::cout << "  LutMlp       LUT-Network Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  LutCnn       LUT-Network Simple CNN" << std::endl;
        std::cout << "  All          run all" << std::endl;
		return 1;
	}

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-num_threads") == 0 && i + 1 < argc) {
            ++i;
            int num_threads = (int)strtoul(argv[i], NULL, 0);
            omp_set_num_threads(num_threads);
        }
        else if (strcmp(argv[i], "-epoch") == 0 && i + 1 < argc) {
            ++i;
            epoch_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-mini_batch") == 0 && i + 1 < argc) {
            ++i;
            mini_batch_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-binary_mode") == 0 && i + 1 < argc) {
            ++i;
            binary_mode = (strtoul(argv[i], NULL, 0) != 0);
        }
        else {
            netname = argv[i];
        }
    }


	if ( netname == "All" || netname == "LutMlp" ) {
		Cifar10LutMlp(epoch_size, mini_batch_size, true);
	}

	if ( netname == "All" || netname == "LutCnn" ) {
    	Cifar10StochasticLut6Cnn(epoch_size, mini_batch_size, true);
	}

	if ( netname == "All" || netname == "DenseMlp" ) {
    	Cifar10DenseMlp(epoch_size, mini_batch_size, true);
	}

	if ( netname == "All" || netname == "DenseCnn" ) {
    	Cifar10DenseCnn(epoch_size, mini_batch_size, true);
	}

	return 0;
}



#if 0

#include <opencv2/opencv.hpp>
#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/DenseAffine.h"
#include "bb/LoweringConvolution.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/MaxPooling.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadCifar10.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"
#include "bb/ExportVerilog.h"

//#include <Windows.h>


void DataTest(void)
{
    auto td = bb::LoadCifar10<>::Load();

#if 0
    {
        for (int i = 0; i < td.x_train.size(); ++i) {
            cv::Mat img(32, 32, CV_32FC3);
		    for (int y = 0; y < 32; ++y) {
   		        for (int x = 0; x < 32; ++x) {
   	    	        img.at<cv::Vec3f>(y, x)[0] = td.x_train[i][(2*32+y)*32+x];
   	    	        img.at<cv::Vec3f>(y, x)[1] = td.x_train[i][(1*32+y)*32+x];
   	    	        img.at<cv::Vec3f>(y, x)[2] = td.x_train[i][(0*32+y)*32+x];
                }
            }
            int label = bb::argmax(td.t_train[i]);

            std::stringstream ss;
            ss << "image/train/" << label;
            ::CreateDirectoryA(ss.str().c_str(), NULL);
            ss << "/" << i << ".png";
            cv::imwrite(ss.str(), img * 255.0);
        }
        for (int i = 0; i < td.x_test.size(); ++i) {
            cv::Mat img(32, 32, CV_32FC3);
		    for (int y = 0; y < 32; ++y) {
   		        for (int x = 0; x < 32; ++x) {
   	    	        img.at<cv::Vec3f>(y, x)[0] = td.x_test[i][(2*32+y)*32+x];
   	    	        img.at<cv::Vec3f>(y, x)[1] = td.x_test[i][(1*32+y)*32+x];
   	    	        img.at<cv::Vec3f>(y, x)[2] = td.x_test[i][(0*32+y)*32+x];
                }
            }
            int label = bb::argmax(td.t_test[i]);

            std::stringstream ss;
            ss << "image/test/" << label;
            ::CreateDirectoryA(ss.str().c_str(), NULL);
            ss << "/" << i << ".png";
            cv::imwrite(ss.str(), img * 255.0);
        }
    }
#endif


    bb::FrameBuffer x_buf;
    x_buf.Resize(BB_TYPE_FP32, 16, td.x_shape);
    x_buf.SetVector(td.x_train, 0);

    auto im2col = bb::ConvolutionIm2Col<>::Create(3, 3);
    auto y_buf = im2col->Forward(x_buf);

    for ( int i = 0; i < 10; ++i ) {
        cv::Mat img0(30, 30, CV_32FC3);
        for (int y = 0; y < 30; ++y) {
            for (int x = 0; x < 30; ++x) {
                img0.at<cv::Vec3f>(y, x)[0] = y_buf.GetFP32(30*30*i + y*30+x, 2*9+0);
                img0.at<cv::Vec3f>(y, x)[1] = y_buf.GetFP32(30*30*i + y*30+x, 1*9+0);
                img0.at<cv::Vec3f>(y, x)[2] = y_buf.GetFP32(30*30*i + y*30+x, 0*9+0);
            }
        }

        cv::Mat img5(30, 30, CV_32FC3);
        for (int y = 0; y < 30; ++y) {
            for (int x = 0; x < 30; ++x) {
                img5.at<cv::Vec3f>(y, x)[0] = y_buf.GetFP32(30*30*i + y*30+x, 2*9+5);
                img5.at<cv::Vec3f>(y, x)[1] = y_buf.GetFP32(30*30*i + y*30+x, 1*9+5);
                img5.at<cv::Vec3f>(y, x)[2] = y_buf.GetFP32(30*30*i + y*30+x, 0*9+5);
            }
        }

        cv::Mat img8(30, 30, CV_32FC3);
        for (int y = 0; y < 30; ++y) {
            for (int x = 0; x < 30; ++x) {
                img8.at<cv::Vec3f>(y, x)[0] = y_buf.GetFP32(30*30*i + y*30+x, 2*9+8);
                img8.at<cv::Vec3f>(y, x)[1] = y_buf.GetFP32(30*30*i + y*30+x, 1*9+8);
                img8.at<cv::Vec3f>(y, x)[2] = y_buf.GetFP32(30*30*i + y*30+x, 0*9+8);
            }
        }

        cv::imshow("img0", img0);
        cv::imshow("img5", img5);
        cv::imshow("img8", img8);
        cv::waitKey();
    }
}

#endif
