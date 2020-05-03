// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/DepthwiseDenseAffine.h"
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



void Cifar10MobileNet(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10MobileNet";


  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    float bn_momentum = 0.01f;

    {
        std::cout << "\n<Training>" << std::endl;

        // create network
        auto cnv0_net = bb::Sequential::Create();
        cnv0_net->Add(bb::DenseAffine<>::Create(32));
        cnv0_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv0_net->Add(bb::ReLU<float>::Create());
        
        auto cnv10_net = bb::Sequential::Create();
        cnv10_net->Add(bb::DepthwiseDenseAffine<>::Create(32));
        cnv10_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv10_net->Add(bb::ReLU<float>::Create());
        auto cnv11_net = bb::Sequential::Create();
        cnv11_net->Add(bb::DenseAffine<>::Create(32));
        cnv11_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv11_net->Add(bb::ReLU<float>::Create());

        auto cnv20_net = bb::Sequential::Create();
        cnv20_net->Add(bb::DepthwiseDenseAffine<>::Create(32));
        cnv20_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv20_net->Add(bb::ReLU<float>::Create());
        auto cnv21_net = bb::Sequential::Create();
        cnv21_net->Add(bb::DenseAffine<>::Create(64));
        cnv21_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv21_net->Add(bb::ReLU<float>::Create());

        auto cnv30_net = bb::Sequential::Create();
        cnv30_net->Add(bb::DepthwiseDenseAffine<>::Create(64));
        cnv30_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv30_net->Add(bb::ReLU<float>::Create());
        auto cnv31_net = bb::Sequential::Create();
        cnv31_net->Add(bb::DenseAffine<>::Create(64));
        cnv31_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv31_net->Add(bb::ReLU<float>::Create());

        auto cnv40_net = bb::Sequential::Create();
        cnv40_net->Add(bb::DepthwiseDenseAffine<>::Create(64));
        cnv40_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv40_net->Add(bb::ReLU<float>::Create());
        auto cnv41_net = bb::Sequential::Create();
        cnv41_net->Add(bb::DenseAffine<>::Create(128));
        cnv41_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv41_net->Add(bb::ReLU<float>::Create());

        auto cnv50_net = bb::Sequential::Create();
        cnv50_net->Add(bb::DepthwiseDenseAffine<>::Create(128));
        cnv50_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv50_net->Add(bb::ReLU<float>::Create());
        auto cnv51_net = bb::Sequential::Create();
        cnv51_net->Add(bb::DenseAffine<>::Create(128));
        cnv51_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv51_net->Add(bb::ReLU<float>::Create());

        auto cnv60_net = bb::Sequential::Create();
        cnv60_net->Add(bb::DepthwiseDenseAffine<>::Create(128));
        cnv60_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv60_net->Add(bb::ReLU<float>::Create());
        auto cnv61_net = bb::Sequential::Create();
        cnv61_net->Add(bb::DenseAffine<>::Create(256));
        cnv61_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv61_net->Add(bb::ReLU<float>::Create());

        auto cnv70_net = bb::Sequential::Create();
        cnv70_net->Add(bb::DepthwiseDenseAffine<>::Create(256));
        cnv70_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv70_net->Add(bb::ReLU<float>::Create());
        auto cnv71_net = bb::Sequential::Create();
        cnv71_net->Add(bb::DenseAffine<>::Create(256));
        cnv71_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv71_net->Add(bb::ReLU<float>::Create());

        auto cnv80_net = bb::Sequential::Create();
        cnv80_net->Add(bb::DepthwiseDenseAffine<>::Create(256));
        cnv80_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv80_net->Add(bb::ReLU<float>::Create());
        auto cnv81_net = bb::Sequential::Create();
        cnv81_net->Add(bb::DenseAffine<>::Create(256));
        cnv81_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv81_net->Add(bb::ReLU<float>::Create());

        auto cnv90_net = bb::Sequential::Create();
        cnv90_net->Add(bb::DepthwiseDenseAffine<>::Create(256));
        cnv90_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv90_net->Add(bb::ReLU<float>::Create());
        auto cnv91_net = bb::Sequential::Create();
        cnv91_net->Add(bb::DenseAffine<>::Create(256));
        cnv91_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv91_net->Add(bb::ReLU<float>::Create());

        auto cnv100_net = bb::Sequential::Create();
        cnv100_net->Add(bb::DepthwiseDenseAffine<>::Create(256));
        cnv100_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv100_net->Add(bb::ReLU<float>::Create());
        auto cnv101_net = bb::Sequential::Create();
        cnv101_net->Add(bb::DenseAffine<>::Create(10));
        cnv101_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv101_net->Add(bb::ReLU<float>::Create());

        auto main_net = bb::Sequential::Create();
        main_net->Add(bb::LoweringConvolution<>::Create(cnv0_net,  3, 3, 1, 1, "same"));    // Conv3x3 x 32            [32x32]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv10_net, 3, 3, 1, 1, "same"));    // depthwise Conv3x3 x 32
        main_net->Add(bb::LoweringConvolution<>::Create(cnv11_net, 1, 1));                  // pointwise Conv1x1 x 32
        main_net->Add(bb::MaxPooling<>::Create(2, 2));                                      //                         [16x16]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv20_net, 3, 3, 1, 1, "same"));    // depthwise Conv3x3 x 64  [16x16]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv21_net, 1, 1));                  // pointwise Conv1x1 x 64
        main_net->Add(bb::LoweringConvolution<>::Create(cnv30_net, 3, 3, 1, 1, "same"));    // depthwise Conv3x3 x 64  [16x16]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv31_net, 1, 1));                  // pointwise Conv1x1 x 64
        main_net->Add(bb::MaxPooling<>::Create(2, 2));                                      //                         [8x8]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv40_net, 3, 3, 1, 1, "same"));    // depthwise Conv3x3 x 128 [8x8]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv41_net, 1, 1));                  // pointwise Conv1x1 x 128
        main_net->Add(bb::LoweringConvolution<>::Create(cnv50_net, 3, 3, 1, 1, "same"));    // depthwise Conv3x3 x 128 [8x8]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv51_net, 1, 1));                  // pointwise Conv1x1 x 128
        main_net->Add(bb::MaxPooling<>::Create(2, 2));                                      //                         [4x4]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv60_net, 3, 3, 1, 1, "same"));    // depthwise Conv3x3 x 256 [4x4]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv61_net, 1, 1));                  // pointwise Conv1x1 x 256
        main_net->Add(bb::LoweringConvolution<>::Create(cnv70_net, 3, 3, 1, 1, "same"));    // depthwise Conv3x3 x 256 [4x4]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv71_net, 1, 1));                  // pointwise Conv1x1 x 256
        main_net->Add(bb::MaxPooling<>::Create(2, 2));                                      //                         [2x2]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv80_net, 3, 3, 1, 1, "same"));    // depthwise Conv3x3 x 256 [2x2]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv81_net, 1, 1));                  // pointwise Conv1x1 x 256
        main_net->Add(bb::LoweringConvolution<>::Create(cnv90_net, 3, 3, 1, 1, "same"));    // depthwise Conv3x3 x 256 [2x2]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv91_net, 1, 1));                  // pointwise Conv1x1 x 256
        main_net->Add(bb::LoweringConvolution<>::Create(cnv100_net, 2, 2, 1, 1));           // depthwise Conv3x3 x 256 [2x2]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv101_net, 1, 1));                 // pointwise Conv1x1 x 10


//        main_net->Add(bb::DenseAffine<float>::Create(512));
//        main_net->Add(bb::BatchNormalization<float>::Create());
//        main_net->Add(bb::ReLU<float>::Create());
//        main_net->Add(bb::DenseAffine<float>::Create(td.t_shape));
//        if ( binary_mode ) {
//            main_net->Add(bb::BatchNormalization<float>::Create());
//            main_net->Add(bb::ReLU<float>::Create());
 //       }

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
        runner_create.initial_evaluation = false;//file_read;       // ファイルを読んだ場合は最初に評価しておく 
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }
}


#if 0
void Cifar10MobileNet(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10MobileNet";


  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    float bn_momentum = 0.01f;

    {
        std::cout << "\n<Training>" << std::endl;

        // create network
        auto cnv0_net = bb::Sequential::Create();
        cnv0_net->Add(bb::DenseAffine<>::Create(32));
        cnv0_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv0_net->Add(bb::ReLU<float>::Create());
        
        auto cnv10_net = bb::Sequential::Create();
        cnv10_net->Add(bb::DepthwiseDenseAffine<>::Create(32));
        cnv10_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv10_net->Add(bb::ReLU<float>::Create());
        auto cnv11_net = bb::Sequential::Create();
        cnv11_net->Add(bb::DenseAffine<>::Create(32));
        cnv11_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv11_net->Add(bb::ReLU<float>::Create());

        auto cnv20_net = bb::Sequential::Create();
        cnv20_net->Add(bb::DepthwiseDenseAffine<>::Create(32));
        cnv20_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv20_net->Add(bb::ReLU<float>::Create());
        auto cnv21_net = bb::Sequential::Create();
        cnv21_net->Add(bb::DenseAffine<>::Create(64));
        cnv21_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv21_net->Add(bb::ReLU<float>::Create());

        auto cnv30_net = bb::Sequential::Create();
        cnv30_net->Add(bb::DepthwiseDenseAffine<>::Create(64));
        cnv30_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv30_net->Add(bb::ReLU<float>::Create());
        auto cnv31_net = bb::Sequential::Create();
        cnv31_net->Add(bb::DenseAffine<>::Create(64));
        cnv31_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv31_net->Add(bb::ReLU<float>::Create());

        auto cnv40_net = bb::Sequential::Create();
        cnv40_net->Add(bb::DepthwiseDenseAffine<>::Create(64));
        cnv40_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv40_net->Add(bb::ReLU<float>::Create());
        auto cnv41_net = bb::Sequential::Create();
        cnv41_net->Add(bb::DenseAffine<>::Create(128));
        cnv41_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv41_net->Add(bb::ReLU<float>::Create());

        auto cnv50_net = bb::Sequential::Create();
        cnv50_net->Add(bb::DepthwiseDenseAffine<>::Create(128));
        cnv50_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv50_net->Add(bb::ReLU<float>::Create());
        auto cnv51_net = bb::Sequential::Create();
        cnv51_net->Add(bb::DenseAffine<>::Create(10));
        cnv51_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        cnv51_net->Add(bb::ReLU<float>::Create());

        auto main_net = bb::Sequential::Create();
        main_net->Add(bb::LoweringConvolution<>::Create(cnv0_net,  3, 3));   // Conv3x3 x 32            [30x30]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv10_net, 3, 3));   // depthwise Conv3x3 x 32  [28x28]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv11_net, 1, 1));   // pointwise Conv1x1 x 32
        main_net->Add(bb::MaxPooling<>::Create(2, 2));                       //                         [14x14]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv20_net, 3, 3));   // depthwise Conv3x3 x 64  [12x12]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv21_net, 1, 1));   // pointwise Conv1x1 x 64
        main_net->Add(bb::LoweringConvolution<>::Create(cnv30_net, 3, 3));   // depthwise Conv3x3 x 64  [10x10]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv31_net, 1, 1));   // pointwise Conv1x1 x 64
        main_net->Add(bb::MaxPooling<>::Create(2, 2));                       //                         [5x5]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv40_net, 3, 3));   // depthwise Conv3x3 x 64  [3x3]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv41_net, 1, 1));   // pointwise Conv1x1 x 64
        main_net->Add(bb::LoweringConvolution<>::Create(cnv50_net, 3, 3));   // depthwise Conv3x3 x 64  [1x1]
        main_net->Add(bb::LoweringConvolution<>::Create(cnv51_net, 1, 1));   // pointwise Conv1x1 x 64

//        main_net->Add(bb::DenseAffine<float>::Create(512));
//        main_net->Add(bb::BatchNormalization<float>::Create());
//        main_net->Add(bb::ReLU<float>::Create());
//        main_net->Add(bb::DenseAffine<float>::Create(td.t_shape));
//        if ( binary_mode ) {
//            main_net->Add(bb::BatchNormalization<float>::Create());
//            main_net->Add(bb::ReLU<float>::Create());
 //       }

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
#endif


// end of file
