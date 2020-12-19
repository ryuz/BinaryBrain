// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/Shuffle.h"
#include "bb/Convolution2d.h"
#include "bb/MaxPooling.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"


template <typename T>
void MnistDenseCnn_(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    // Network name
    std::string net_name = "MnistDenseCnn";
    if ( binary_mode ) {
        net_name += "Bit";
    }

    // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    {
        std::cout << "\n<Training>" << std::endl;

        // create network
        auto cnv0_net = bb::Sequential::Create();
        cnv0_net->Add(bb::DenseAffine<>::Create(32));
        cnv0_net->Add(bb::BatchNormalization<>::Create());
        cnv0_net->Add(bb::ReLU<T>::Create());

        auto cnv1_net = bb::Sequential::Create();
        cnv1_net->Add(bb::DenseAffine<>::Create(32));
        cnv1_net->Add(bb::BatchNormalization<>::Create());
        cnv1_net->Add(bb::ReLU<T>::Create());

        auto cnv2_net = bb::Sequential::Create();
        cnv2_net->Add(bb::DenseAffine<>::Create(64));
        cnv2_net->Add(bb::BatchNormalization<>::Create());
        cnv2_net->Add(bb::ReLU<T>::Create());

        auto cnv3_net = bb::Sequential::Create();
        cnv3_net->Add(bb::DenseAffine<>::Create(64));
        cnv3_net->Add(bb::BatchNormalization<>::Create());
        cnv3_net->Add(bb::ReLU<T>::Create());

        auto main_net = bb::Sequential::Create();
        main_net->Add(bb::Convolution2d<T>::Create(cnv0_net, 3, 3));   // Conv3x3 x 32 [26x26]
        main_net->Add(bb::Convolution2d<T>::Create(cnv1_net, 3, 3));   // Conv3x3 x 32 [24x24]
        main_net->Add(bb::MaxPooling<T>::Create(2, 2));                      //              [12x12]
        main_net->Add(bb::Convolution2d<T>::Create(cnv2_net, 3, 3));   // Conv3x3 x 64 [10x10]
        main_net->Add(bb::Convolution2d<T>::Create(cnv3_net, 3, 3));   // Conv3x3 x 64 [8x8]
        main_net->Add(bb::MaxPooling<T>::Create(2, 2));                      // Conv3x3 x 64 [4x4]
        main_net->Add(bb::DenseAffine<>::Create(256));
        main_net->Add(bb::ReLU<T>::Create());
        main_net->Add(bb::DenseAffine<>::Create(td.t_shape));
        if ( binary_mode ) {
            main_net->Add(bb::BatchNormalization<>::Create());
            main_net->Add(bb::ReLU<T>::Create());
        }

        // modulation wrapper
        auto net = bb::BinaryModulation<T>::Create(main_net, train_modulation_size, test_modulation_size);

        // set input shape
        (void)net->SetInputShape(td.x_shape);

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
        std::cout << "train_modulation_size : " << train_modulation_size << std::endl;
        std::cout << "test_modulation_size  : " << test_modulation_size  << std::endl;
        }
        std::cout << "binary_mode           : " << binary_mode           << std::endl;
        std::cout << "file_read             : " << file_read             << std::endl;

        // run fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name               = net_name;
        runner_create.net                = net;
        runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく 
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }
}


void MnistDenseCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    if ( binary_mode ) {
        MnistDenseCnn_<bb::Bit>(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    else {
        MnistDenseCnn_<float>(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
}



// end of file
