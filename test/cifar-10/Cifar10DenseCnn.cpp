// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/DenseAffine.h"
#include "bb/LoweringConvolution.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/MaxPooling.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/AccuracyCategoricalClassification.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadCifar10.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"
#include "bb/ExportVerilog.h"

#include <Windows.h>

// MNIST CNN with LUT networks
void Cifar10DenseCnn(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    std::string net_name = "Cifar10SimpleLutCnn";
    int const   frame_mux_size = 7;

  // load cifar-10 data
#ifdef _DEBUG
	auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

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

    // create network
    auto net = bb::Sequential::Create();
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(32), 3, 3));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(32), 3, 3));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::DenseAffine<>::Create(512));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::DenseAffine<>::Create(td.t_shape));
    net->SetInputShape(td.x_shape);

    // print model information
    net->PrintInfo(2);

    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name      = net_name;
    runner_create.net       = net;
    runner_create.lossFunc  = bb::LossSoftmaxCrossEntropy<>::Create();
    runner_create.accFunc   = bb::AccuracyCategoricalClassification<>::Create();
    runner_create.optimizer = bb::OptimizerAdam<>::Create();
    runner_create.file_read  = false;       // 前の計算結果があれば読み込んで再開するか
    runner_create.file_write = true;        // 計算結果をファイルに保存するか
    runner_create.write_serial = true; 
    runner_create.print_progress = true;    // 途中結果を表示
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}



// end of file
