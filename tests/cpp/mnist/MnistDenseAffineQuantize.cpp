// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/BinaryModulation.h"
#include "bb/Binarize.h"
#include "bb/MaxPooling.h"
#include "bb/DifferentiableLutN.h"
#include "bb/DenseAffineQuantize.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"


void MnistDenseAffineQuantize(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "MnistDenseAffineQuantize";

    // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    const int wb = 6;
    const int wq = 4;
    const int ob = 16;
    const int oq = wq;
    const int ib = ob;
    const int iq = oq;

    {
        std::cout << "\n<Training>" << std::endl;

        auto main_net = bb::Sequential::Create();

        main_net->Add(bb::RealToBinary<float>::Create());
//      main_net->Add(bb::MaxPooling<float>::Create(2, 2));

        main_net->Add(bb::DenseAffineQuantize<float>::Create(256, true, wb, ob, ib, 1.0f/(1<<wq), 1.0f/(1<<oq), 1.0/(1 << iq)));
        main_net->Add(bb::ReLU<float>::Create());

        main_net->Add(bb::DenseAffineQuantize<float>::Create(256, true, wb, ob, ib, 1.0f/(1<<wq), 1.0f/(1<<oq), 1.0/(1 << iq)));
        main_net->Add(bb::ReLU<float>::Create());

        main_net->Add(bb::DenseAffineQuantize<float>::Create(td.t_shape, true, wb, ob, ib, 1.0f/(1<<wq), 1.0f/(1<<oq), 1.0/(1 << iq)));

        // modulation wrapper
        auto net = main_net;

        // set input shape
        net->SetInputShape(td.x_shape);

        // set binary mode
        net->SendCommand("binary false");
        net->SendCommand("quantize true");

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
        runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);

//        net->SendCommand("quantize true");
        runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner->Fitting(td, epoch_size, mini_batch_size);

    }
}


#if 0

void MnistDenseAffineQuantize(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "MnistDenseAffineQuantize";

    // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    const int wb = 8;
    const int wq = 8;
    const int ob = 16;
    const int oq = 5;
    const int ib = ob;
    const int iq = oq;

    {
        std::cout << "\n<Training>" << std::endl;

        // create network (ReLU acts as a Binarizer when in binary mode)
        auto main_net = bb::Sequential::Create();

        main_net->Add(bb::RealToBinary<float>::Create());

//        main_net->Add(bb::Binarize<float>::Create());
        main_net->Add(bb::MaxPooling<float>::Create(2, 2));
//        main_net->Add(bb::Binarize<float>::Create());

        main_net->Add(bb::DifferentiableLutN<6, float>::Create(6*128));
        main_net->Add(bb::DifferentiableLutN<6, float>::Create(128));

//      main_net->Add(bb::DenseAffineQuantize<float>::Create(256, true, 4, ob, ib, 1.0f/(1<<4), 1.0f/(1<<oq), 1.0/(1 << iq)));
//      main_net->Add(bb::DifferentiableLutN<6, float>::Create(6*6*64));
//      main_net->Add(bb::Binarize<float>::Create());
        main_net->Add(bb::DifferentiableLutN<6, float>::Create(6*64));
//      main_net->Add(bb::Binarize<float>::Create());
//      main_net->Add(bb::BatchNormalization<float>::Create());
        main_net->Add(bb::DifferentiableLutN<6, float>::Create(64));
//      main_net->Add(bb::Binarize<float>::Create());
//      main_net->Add(bb::BatchNormalization<float>::Create());
//      main_net->Add(bb::ReLU<float>::Create());


        main_net->Add(bb::DifferentiableLutN<6, float>::Create(6*64));
        main_net->Add(bb::DifferentiableLutN<6, float>::Create(64));

        main_net->Add(bb::DifferentiableLutN<6, float>::Create(6*32));
        main_net->Add(bb::DifferentiableLutN<6, float>::Create(32));

//        main_net->Add(bb::Binarize<float>::Create());
//        main_net->Add(bb::DifferentiableLutN<6, float>::Create(128));
//        main_net->Add(bb::Binarize<float>::Create());

//        main_net->Add(bb::DenseAffineQuantize<float>::Create(32, true, 8, ob, ib, 1.0f/(1<<8), 1.0f/(1<<oq), 1.0/(1 << iq)));
//      main_net->Add(bb::BatchNormalization<float>::Create());
//        main_net->Add(bb::ReLU<float>::Create());

        main_net->Add(bb::DenseAffineQuantize<float>::Create(td.t_shape, true, wb, ob, ib, 1.0f/(1<<wq), 1.0f/(1<<oq), 1.0/(1 << iq)));
        if ( binary_mode ) {
//            main_net->Add(bb::BatchNormalization<float>::Create());
            main_net->Add(bb::ReLU<float>::Create());
        }

        // modulation wrapper
//      auto net = bb::BinaryModulation<float>::Create(main_net, train_modulation_size, test_modulation_size);
        auto net = main_net;

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

#endif

// end of file
