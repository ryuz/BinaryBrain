// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>

#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/DenseAffine.h"
#include "bb/LoweringConvolution.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/Sigmoid.h"
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
#include "bb/Reduce.h"
#include "bb/UniformDistributionGenerator.h"

#if 0

// Dense CNN
void Cifar10DenseCnnX(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, bool binary_mode, bool file_read,
            bool gen_rand, bool framewise, bool log_append = true)
{
    std::string net_name = "Cifar10DenseCnn";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    float bn_momentum = 0.9f;
    if (binary_mode) {
        bn_momentum = 0.1f;
    }
    
    // create network
    auto net = bb::Sequential::Create();
    if ( binary_mode ) {
        if ( gen_rand ) {
            net->Add(bb::RealToBinary<>::Create(frame_mux_size, bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1), framewise));
        }
        else {
            net->Add(bb::RealToBinary<>::Create(frame_mux_size));
        }
    }
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3));
    net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3));
    net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(128), 3, 3));
    net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(128), 3, 3));
    net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::DenseAffine<>::Create(1024));
    net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::DenseAffine<>::Create(310));
    if ( binary_mode ) {
    net->Add(bb::BatchNormalization<>::Create());
        net->Add(bb::Binarize<>::Create());
        net->Add(bb::Reduce<>::Create(td.t_shape));
        net->Add(bb::BinaryToReal<>::Create(td.t_shape, frame_mux_size));
    }
    else {
        net->Add(bb::Reduce<>::Create(td.t_shape));
    }
    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        std::cout << "binary true" << std::endl;
        net->SendCommand("binary true");
    }
    else {
        std::cout << "binary false" << std::endl;
        net->SendCommand("binary false");
    }

    // print model information
//  net->PrintInfo();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "file_read          : " << file_read          << std::endl;
    std::cout << "log_append         : " << log_append         << std::endl;
    std::cout << "epoch_size         : " << epoch_size         << std::endl;
    std::cout << "mini_batch_size    : " << mini_batch_size    << std::endl;
    std::cout << "max_run_size       : " << max_run_size       << std::endl;
    std::cout << "frame_mux_size     : " << frame_mux_size     << std::endl;
    std::cout << "binary_mode        : " << binary_mode        << std::endl;
    std::cout << "gen_rand           : " << gen_rand           << std::endl;
    std::cout << "framewise          : " << framewise          << std::endl;

    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name               = net_name;
    runner_create.net                = net;
    runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<>::Create();
    runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<>::Create();
    runner_create.optimizer          = bb::OptimizerAdam<>::Create();
    runner_create.max_run_size       = max_run_size;    // 実際の1回の実行サイズ
    runner_create.log_append         = log_append;
    runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
    runner_create.file_write         = true;            // 計算結果をファイルに保存するか
    runner_create.print_progress     = true;            // 途中結果を表示
    runner_create.initial_evaluation = true;            // ファイルを読んだ場合は最初に評価しておく
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}

#else
// Dense CNN
void Cifar10DenseCnnX(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, bool binary_mode, bool file_read,
            bool gen_rand, bool framewise, bool log_append = true)
{
    std::string net_name = "Cifar10DenseCnn_Lowering";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    float bn_momentum = 0.9f;
    if (binary_mode) {
        bn_momentum = 0.1f;
    }
    
    auto cnv0_sub = bb::Sequential::Create();
    cnv0_sub->Add(bb::DenseAffine<>::Create(64));
    cnv0_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv0_sub->Add(bb::ReLU<>::Create());

    auto cnv1_sub = bb::Sequential::Create();
    cnv1_sub->Add(bb::DenseAffine<>::Create(64));
    cnv1_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv1_sub->Add(bb::ReLU<>::Create());

    auto cnv2_sub = bb::Sequential::Create();
    cnv2_sub->Add(bb::DenseAffine<>::Create(128));
    cnv2_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv2_sub->Add(bb::ReLU<>::Create());

    auto cnv3_sub = bb::Sequential::Create();
    cnv3_sub->Add(bb::DenseAffine<>::Create(128));
    cnv3_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv3_sub->Add(bb::ReLU<>::Create());


    // create network
    auto net = bb::Sequential::Create();
    if ( binary_mode ) {
        if ( gen_rand ) {
            net->Add(bb::RealToBinary<>::Create(frame_mux_size, bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1), framewise));
        }
        else {
            net->Add(bb::RealToBinary<>::Create(frame_mux_size));
        }
    }
    net->Add(bb::LoweringConvolution<>::Create(cnv0_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv1_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::LoweringConvolution<>::Create(cnv2_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv3_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::DenseAffine<>::Create(1024));
    net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    net->Add(bb::ReLU<>::Create());
    net->Add(bb::DenseAffine<>::Create(310));
    if ( binary_mode ) {
        net->Add(bb::BatchNormalization<>::Create(bn_momentum));
        net->Add(bb::Binarize<>::Create());
        net->Add(bb::Reduce<>::Create(td.t_shape));
        net->Add(bb::BinaryToReal<>::Create(td.t_shape, frame_mux_size));
    }
    else {
        net->Add(bb::Reduce<>::Create(td.t_shape));
    }
    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        std::cout << "binary true" << std::endl;
        net->SendCommand("binary true");
    }
    else {
        std::cout << "binary false" << std::endl;
        net->SendCommand("binary false");
    }

    // print model information
//  net->PrintInfo();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "file_read          : " << file_read          << std::endl;
    std::cout << "log_append         : " << log_append         << std::endl;
    std::cout << "epoch_size         : " << epoch_size         << std::endl;
    std::cout << "mini_batch_size    : " << mini_batch_size    << std::endl;
    std::cout << "max_run_size       : " << max_run_size       << std::endl;
    std::cout << "frame_mux_size     : " << frame_mux_size     << std::endl;
    std::cout << "binary_mode        : " << binary_mode        << std::endl;
    std::cout << "gen_rand           : " << gen_rand           << std::endl;
    std::cout << "framewise          : " << framewise          << std::endl;

    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name               = net_name;
    runner_create.net                = net;
    runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<>::Create();
    runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<>::Create();
    runner_create.optimizer          = bb::OptimizerAdam<>::Create();
    runner_create.max_run_size       = max_run_size;    // 実際の1回の実行サイズ
    runner_create.log_append         = log_append;
    runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
    runner_create.file_write         = true;            // 計算結果をファイルに保存するか
    runner_create.print_progress     = true;            // 途中結果を表示
    runner_create.initial_evaluation = true;            // ファイルを読んだ場合は最初に評価しておく
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}

#endif



void Cifar10DenseCnn(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, int lut_frame_mux_size, bool binary_mode, bool file_read)
{
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 1,  binary_mode, file_read, false, false, false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 3,  binary_mode, file_read, false, false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 7,  binary_mode, file_read, false, false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 15, binary_mode, file_read, false, false);
    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 31, binary_mode, file_read, false, false);

    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 1,  binary_mode, file_read, true,  false);
    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 3,  binary_mode, file_read, true,  false);
    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 7,  binary_mode, file_read, true,  false);
    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 15, binary_mode, file_read, true,  false);
    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 31, binary_mode, file_read, true,  false);

    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 1,  binary_mode, file_read, true,  true);
    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 3,  binary_mode, file_read, true,  true);
    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 7,  binary_mode, file_read, true,  true);
    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 15, binary_mode, file_read, true,  true);
    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 31, binary_mode, file_read, true,  true);
}


// end of file
