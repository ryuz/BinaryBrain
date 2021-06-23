// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/BinaryModulation.h"
#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/OptimizerAdam.h"
#include "bb/Sigmoid.h"
#include "bb/LossBinaryCrossEntropy.h"
#include "bb/MetricsBinaryCategoricalAccuracy.h"
#include "bb/LossSigmoidCrossEntropy.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"


std::vector<float> conv_binary_class(std::vector<float> t10)
{
    std::vector<float>  t2(1);
    t2[0] = 0;
    for (size_t i = 0; i < t10.size(); ++i) {
        t2[0] += t10[i] * (i%2);
    }
    return t2;
}


void MnistDenseBinary(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "MnistDenseBinary";

    // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    td.t_shape[0] = 1;
    for (auto& t : td.t_train) { t = conv_binary_class(t); }
    for (auto& t : td.t_test)  { t = conv_binary_class(t); }

    {
        std::cout << "\n<Training>" << std::endl;
        
        auto sigmoid = bb::Sigmoid<float>::Create();

        // create network (ReLU acts as a Binarizer when in binary mode)
        auto main_net = bb::Sequential::Create();
        main_net->Add(bb::DenseAffine<float>::Create(1024));
        main_net->Add(bb::BatchNormalization<float>::Create());
        main_net->Add(bb::ReLU<float>::Create());
        main_net->Add(bb::DenseAffine<float>::Create(512));
        main_net->Add(bb::BatchNormalization<float>::Create());
        main_net->Add(bb::ReLU<float>::Create());
        main_net->Add(bb::DenseAffine<float>::Create(1));
//      main_net->Add(sigmoid);

        // modulation wrapper
        auto net = bb::BinaryModulation<float>::Create(main_net, train_modulation_size, test_modulation_size);

        // set input shape
        net->SetInputShape(td.x_shape);

        // set binary mode
        if ( binary_mode ) {
            net->SendCommand("binary true");
        }
        else {
            net->SendCommand("binary false");
        }
        sigmoid->SendCommand("binary false");

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
        std::cout << "-----------------------------------" << std::endl;

    
        // run fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name               = net_name;
        runner_create.net                = net;
//      runner_create.lossFunc           = bb::LossBinaryCrossEntropy<float>::Create();
        runner_create.lossFunc           = bb::LossSigmoidCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsBinaryCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }
}

// end of file
