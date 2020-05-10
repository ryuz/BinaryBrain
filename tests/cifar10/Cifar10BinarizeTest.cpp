// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LoweringConvolution.h"
#include "bb/MaxPooling.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadCifar10.h"



template<typename T>
void Cifar10BinarizeTest_(int epoch_size, int mini_batch_size, int depth_modulation_size, int frame_modulation_size, bool binary_mode)
{
    std::string net_name = "Cifar10BinarizeTest";
    if ( binary_mode ) {
        net_name += "_depth=" + std::to_string(depth_modulation_size) + "_frame=" + std::to_string(frame_modulation_size);
    }
    else {
        net_name += "_fp32";
    }

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    float momentam = 0.9f;
//  if ( binary_mode ) {
//      momentam = 0.1f;
//  }

    {
        std::cout << "\n<Training>" << std::endl;

        // create network
        auto cnv0_net = bb::Sequential::Create();
        cnv0_net->Add(bb::DenseAffine<>::Create(32));
        cnv0_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv0_net->Add(bb::ReLU<T>::Create());

        auto cnv1_net = bb::Sequential::Create();
        cnv1_net->Add(bb::DenseAffine<>::Create(64));
        cnv1_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv1_net->Add(bb::ReLU<T>::Create());

        auto cnv2_net = bb::Sequential::Create();
        cnv2_net->Add(bb::DenseAffine<>::Create(64));
        cnv2_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv2_net->Add(bb::ReLU<T>::Create());

        auto cnv3_net = bb::Sequential::Create();
        cnv3_net->Add(bb::DenseAffine<>::Create(128));
        cnv3_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv3_net->Add(bb::ReLU<T>::Create());

        auto cnv4_net = bb::Sequential::Create();
        cnv4_net->Add(bb::DenseAffine<>::Create(128));
        cnv4_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv4_net->Add(bb::ReLU<T>::Create());

        auto cnv5_net = bb::Sequential::Create();
        cnv5_net->Add(bb::DenseAffine<>::Create(256));
        cnv5_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv5_net->Add(bb::ReLU<T>::Create());

        auto main_net = bb::Sequential::Create();

#if 0
        if ( depth_modulation_size > 1 ) {
            auto cnv_net = bb::Sequential::Create();
            cnv_net->Add(bb::DenseAffine<>::Create(32));
            cnv_net->Add(bb::BatchNormalization<>::Create(momentam));
            cnv_net->Add(bb::ReLU<T>::Create());
            main_net->Add(bb::LoweringConvolution<T>::Create(cnv_net, 1, 1));// 32x32
        }
#endif

        main_net->Add(bb::LoweringConvolution<T>::Create(cnv0_net, 3, 3));   // 30x30
        main_net->Add(bb::LoweringConvolution<T>::Create(cnv1_net, 3, 3));   // 28x28
        main_net->Add(bb::MaxPooling<T>::Create(2, 2));                      // 14x14
        main_net->Add(bb::LoweringConvolution<T>::Create(cnv2_net, 3, 3));   // 12x12
        main_net->Add(bb::LoweringConvolution<T>::Create(cnv3_net, 3, 3));   // 10x10
        main_net->Add(bb::MaxPooling<T>::Create(2, 2));                      // 5x5
        main_net->Add(bb::LoweringConvolution<T>::Create(cnv4_net, 3, 3));   // 3x3
        main_net->Add(bb::LoweringConvolution<T>::Create(cnv5_net, 3, 3));   // 1x1
        
        main_net->Add(bb::DenseAffine<>::Create(512));
        main_net->Add(bb::BatchNormalization<>::Create(momentam));
        main_net->Add(bb::ReLU<T>::Create());

        main_net->Add(bb::DenseAffine<>::Create(td.t_shape));
        if ( binary_mode ) {
            main_net->Add(bb::BatchNormalization<>::Create());
            main_net->Add(bb::ReLU<T>::Create());
        }

        // modulation wrapper
        auto net = bb::Sequential::Create();
        if ( binary_mode ) {
            net->Add(bb::BinaryModulation<T>::Create(main_net, frame_modulation_size, frame_modulation_size, depth_modulation_size));
//          net->Add(bb::RealToBinary<T>::Create(frame_modulation_size, depth_modulation_size));
//          net->Add(main_net);
//          net->Add(bb::BinaryToReal<T>::Create(frame_modulation_size));
        }
        else {
            net->Add(main_net);
        }

        // set input shape
        net->SetInputShape(td.x_shape);

        // set binary mode
        if ( binary_mode ) {
            net->SendCommand("binary true");
        }
        else {
            net->SendCommand("binary false");
        }

        // print model information
        net->PrintInfo();

        std::cout << "-----------------------------------" << std::endl;
        std::cout << "epoch_size            : " << epoch_size            << std::endl;
        std::cout << "mini_batch_size       : " << mini_batch_size       << std::endl;
        if ( binary_mode ) {
        std::cout << "depth_modulation_size : " << depth_modulation_size << std::endl;
        std::cout << "frame_modulation_size : " << frame_modulation_size << std::endl;
        }
        std::cout << "binary_mode           : " << binary_mode           << std::endl;

        // run fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name               = net_name;
        runner_create.net                = net;
        runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = false;           // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = true;            // ファイルを読んだ場合は最初に評価しておく 
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }
}


void Cifar10BinarizeTest(void)
{
    int epoch_size      = 256;
    int mini_batch_size = 32*4;

    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  4, 4, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size, 32, 1, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size, 64, 1, true);

/*
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size, 16, 1, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  8, 1, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  4, 1, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  2, 1, true);

    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  1,  1, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  1,  2, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  1,  4, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  1,  8, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  1, 16, true);
    Cifar10BinarizeTest_<bb::Bit>(epoch_size, mini_batch_size,  1, 32, true);

    Cifar10BinarizeTest_<float>(epoch_size, mini_batch_size, 1, 1, false);
*/
}



// end of file
