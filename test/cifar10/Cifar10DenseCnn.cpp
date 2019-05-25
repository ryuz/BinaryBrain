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

#include "bb/SparseLutN.h"
#include "bb/MicroMlp.h"
#include "bb/SparseBinaryLutN.h"
#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/DenseAffine.h"
#include "bb/BinaryModulation.h"
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
void Cifar10DenseCnn(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, int lut_frame_mux_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10DenseCnn";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
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
    net->Add(bb::ReLU<>::Create());
    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        net->SendCommand("binary true");
    }
    else {
        net->SendCommand("binary false");
    }

    // print model information
    net->PrintInfo();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "file_read          : " << file_read          << std::endl;
    std::cout << "epoch_size         : " << epoch_size         << std::endl;
    std::cout << "mini_batch_size    : " << mini_batch_size    << std::endl;
    std::cout << "max_run_size       : " << max_run_size       << std::endl;
    std::cout << "frame_mux_size     : " << frame_mux_size     << std::endl;
    std::cout << "binary_mode        : " << binary_mode        << std::endl;

    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name        = net_name;
    runner_create.net         = net;
    runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<>::Create();
    runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<>::Create();
    runner_create.optimizer   = bb::OptimizerAdam<>::Create();
    runner_create.max_run_size       = max_run_size;    // 実際の1回の実行サイズ
    runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
    runner_create.file_write         = true;            // 計算結果をファイルに保存するか
    runner_create.print_progress     = true;            // 途中結果を表示
    runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}

#endif


#if 0

void Cifar10DenseCnn(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, int lut_frame_mux_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10DenseCnn_full_cnn";

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
    auto main_net = bb::Sequential::Create();

    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3, 1, 1, "same"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3, 1, 1, "same"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::MaxPooling<>::Create(2, 2));

    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(128), 3, 3, 1, 1, "same"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(128), 3, 3, 1, 1, "same"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::MaxPooling<>::Create(2, 2));

    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(256), 3, 3, 1, 1, "same"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(256), 3, 3, 1, 1, "same"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::MaxPooling<>::Create(2, 2));

#if 1
    main_net->Add(bb::RealToBinary<bb::Bit>::Create());

#if 0
    auto sub6_net = bb::Sequential::Create();
    sub6_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 16, 512}, "random"));
    sub6_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 6, 512}, "depthwise"));
    sub6_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 1, 512}, "depthwise"));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(sub6_net, 3, 3, 1, 1, "same"));
    auto sub7_net = bb::Sequential::Create();
    sub7_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 16, 512}, "random"));
    sub7_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 6, 512}, "depthwise"));
    sub7_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 1, 512}, "depthwise"));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(sub7_net, 3, 3, 1, 1, "same"));
    main_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));
#else
    auto sub6_net = bb::Sequential::Create();
    sub6_net->Add(bb::SparseLutN<6, bb::Bit>::Create(18432));
    sub6_net->Add(bb::SparseLutN<6, bb::Bit>::Create(3072));
    sub6_net->Add(bb::SparseLutN<6, bb::Bit>::Create(512));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(sub6_net, 3, 3, 1, 1, "same"));
    auto sub7_net = bb::Sequential::Create();
    sub7_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 16, 512}, "depthwise"));
    sub7_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 16, 512}, "pointwise"));
    sub7_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  6, 512}, "depthwise"));
    sub7_net->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1, 512}, "random"));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(sub7_net, 3, 3, 1, 1, "same"));
    main_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));
#endif

//  main_net->Add(bb::BinaryToReal<bb::Bit, float>::Create());
#else
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(512), 3, 3, 1, 1, "same"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(512), 3, 3, 1, 1, "same"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::MaxPooling<>::Create(2, 2));
#endif

#if 1
//  main_net->Add(bb::RealToBinary<bb::Bit>::Create());

    main_net->Add(bb::SparseLutN<6, bb::Bit>::Create(9216));
    main_net->Add(bb::SparseLutN<6, bb::Bit>::Create(1536));
    main_net->Add(bb::SparseLutN<6, bb::Bit>::Create(256));

    main_net->Add(bb::SparseLutN<6, bb::Bit>::Create(2160));
    main_net->Add(bb::SparseLutN<6, bb::Bit>::Create(360));
    main_net->Add(bb::SparseLutN<6, bb::Bit>::Create(60));
    main_net->Add(bb::SparseLutN<6, bb::Bit>::Create(10));

    main_net->Add(bb::BinaryToReal<bb::Bit, float>::Create());

#else
    main_net->Add(bb::DenseAffine<>::Create(512));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::DenseAffine<>::Create(10));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
#endif

    auto net = bb::Sequential::Create();
#if 1
    bb::BinaryModulation<float, float>::create_t mod_create;
    mod_create.layer                     = main_net;
    mod_create.output_shape              = td.t_shape;
#if 1
    mod_create.training_modulation_size  = frame_mux_size;
    mod_create.training_value_generator  = nullptr;
#else
    mod_create.training_modulation_size  = 1;
//  mod_create.training_value_generator  = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 12345);
//  mod_create.training_value_generator  = bb::NormalDistributionGenerator<float>::Create(0.5f, 0.3f, 777);
#endif
    mod_create.inference_modulation_size = frame_mux_size;
    mod_create.inference_value_generator = nullptr;
    net->Add(bb::BinaryModulation<float, float>::Create(mod_create));
#else
    net->Add(main_net);

#endif
//  net->Add(bb::Reduce<>::Create(td.t_shape));

    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        net->SendCommand("binary true");
    }
    else {
        net->SendCommand("binary false");
    }

    // print model information
    net->PrintInfo();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "file_read          : " << file_read          << std::endl;
    std::cout << "epoch_size         : " << epoch_size         << std::endl;
    std::cout << "mini_batch_size    : " << mini_batch_size    << std::endl;
    std::cout << "max_run_size       : " << max_run_size       << std::endl;
    std::cout << "frame_mux_size     : " << frame_mux_size     << std::endl;
    std::cout << "binary_mode        : " << binary_mode        << std::endl;

    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name        = net_name;
    runner_create.net         = net;
    runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<>::Create();
    runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<>::Create();
    runner_create.optimizer   = bb::OptimizerAdam<>::Create();
    runner_create.max_run_size       = max_run_size;    // 実際の1回の実行サイズ
    runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
    runner_create.file_write         = true;            // 計算結果をファイルに保存するか
    runner_create.print_progress     = true;            // 途中結果を表示
    runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}

#endif


#if 0

void Cifar10DenseCnn(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, int lut_frame_mux_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10DenseCnn_full_cnn_mini";

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
    auto main_net = bb::Sequential::Create();

#if 0
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(32), 3, 3, 1, 1, "valid"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(32), 3, 3, 1, 1, "valid"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::MaxPooling<>::Create(2, 2));
#else
    main_net->Add(bb::RealToBinary<bb::Bit>::Create());

    auto sub0_net = bb::Sequential::Create();
    sub0_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(1152));
    sub0_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(192));
    sub0_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(32));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(sub0_net, 3, 3, 1, 1, "valid"));

    auto sub1_net = bb::Sequential::Create();
    sub1_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(1152));
    sub1_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(192));
    sub1_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(32));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(sub1_net, 3, 3, 1, 1, "valid"));
#endif

#if 0
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3, 1, 1, "valid"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3, 1, 1, "valid"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::MaxPooling<>::Create(2, 2));
#else
    auto sub2_net = bb::Sequential::Create();
    sub2_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(2304));
    sub2_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(384));
    sub2_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(64));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(sub2_net, 3, 3, 1, 1, "valid"));

    auto sub3_net = bb::Sequential::Create();
    sub3_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(2304));
    sub3_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(384));
    sub3_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(64));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(sub3_net, 3, 3, 1, 1, "valid"));
#endif

#if 0
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(128), 3, 3, 1, 1, "valid"));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
#else
//  main_net->Add(bb::RealToBinary<bb::Bit>::Create());

    auto sub4_net = bb::Sequential::Create();
    sub4_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(4608));
    sub4_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(768));
    sub4_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(128));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(sub4_net, 3, 3, 1, 1, "valid"));
#endif

#if 0

    main_net->Add(bb::DenseAffine<>::Create(128));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());
    main_net->Add(bb::DenseAffine<>::Create(10));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<>::Create());

#else

//  main_net->Add(bb::RealToBinary<bb::Bit>::Create());
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(4608));
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(768));
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(128));

    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(360));
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(60));
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(10));

    main_net->Add(bb::BinaryToReal<bb::Bit, float>::Create());
#endif

    auto net = bb::Sequential::Create();
#if 1
    bb::BinaryModulation<float, float>::create_t mod_create;
    mod_create.layer                     = main_net;
    mod_create.output_shape              = td.t_shape;
#if 1
    mod_create.training_modulation_size  = frame_mux_size;
    mod_create.training_value_generator  = nullptr;
#else
    mod_create.training_modulation_size  = 1;
//  mod_create.training_value_generator  = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 12345);
//  mod_create.training_value_generator  = bb::NormalDistributionGenerator<float>::Create(0.5f, 0.3f, 777);
#endif
    mod_create.inference_modulation_size = frame_mux_size;
    mod_create.inference_value_generator = nullptr;
    net->Add(bb::BinaryModulation<float, float>::Create(mod_create));
#else
    net->Add(main_net);

#endif
//  net->Add(bb::Reduce<>::Create(td.t_shape));

    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        net->SendCommand("binary true");
    }
    else {
        net->SendCommand("binary false");
    }

    // print model information
    net->PrintInfo();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "file_read          : " << file_read          << std::endl;
    std::cout << "epoch_size         : " << epoch_size         << std::endl;
    std::cout << "mini_batch_size    : " << mini_batch_size    << std::endl;
    std::cout << "max_run_size       : " << max_run_size       << std::endl;
    std::cout << "frame_mux_size     : " << frame_mux_size     << std::endl;
    std::cout << "binary_mode        : " << binary_mode        << std::endl;

    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name        = net_name;
    runner_create.net         = net;
    runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<>::Create();
    runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<>::Create();
    runner_create.optimizer   = bb::OptimizerAdam<>::Create();
    runner_create.max_run_size       = max_run_size;    // 実際の1回の実行サイズ
    runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
    runner_create.file_write         = true;            // 計算結果をファイルに保存するか
    runner_create.print_progress     = true;            // 途中結果を表示
    runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}

#endif



/////////////////



#if 1


// Dense CNN
void Cifar10DenseCnn(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10DenseCnn";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    // BatchNorm momentum
    float bn_momentum = 0.9f;
    if (binary_mode) {
        bn_momentum = 0.1f;
    }
    
    // create network
    auto main_net = bb::Sequential::Create();
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::ReLU<>::Create());
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::ReLU<>::Create());
    main_net->Add(bb::MaxPooling<>::Create(2, 2));
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(128), 3, 3));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::ReLU<>::Create());
    main_net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(128), 3, 3));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::ReLU<>::Create());
    main_net->Add(bb::MaxPooling<>::Create(2, 2));
    main_net->Add(bb::DenseAffine<>::Create(512));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::ReLU<>::Create());
    main_net->Add(bb::DenseAffine<>::Create(10));
    if ( binary_mode ) {
        main_net->Add(bb::BatchNormalization<>::Create());
        main_net->Add(bb::Binarize<>::Create());
    }

    auto net = bb::Sequential::Create();
    if ( binary_mode ) {
        bb::BinaryModulation<float, float>::create_t mod_create;
        mod_create.layer                     = main_net;
        mod_create.output_shape              = td.t_shape;
        mod_create.training_modulation_size  = frame_mux_size;
        mod_create.training_value_generator  = nullptr;
        mod_create.inference_modulation_size = frame_mux_size;
        mod_create.inference_value_generator = nullptr;
        net->Add(bb::BinaryModulation<float, float>::Create(mod_create));
    }
    else {
        net->Add(main_net);
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

    net->PrintInfo();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "file_read          : " << file_read          << std::endl;
    std::cout << "epoch_size         : " << epoch_size         << std::endl;
    std::cout << "mini_batch_size    : " << mini_batch_size    << std::endl;
    std::cout << "max_run_size       : " << max_run_size       << std::endl;
    std::cout << "frame_mux_size     : " << frame_mux_size     << std::endl;
    std::cout << "binary_mode        : " << binary_mode        << std::endl;

    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name               = net_name;
    runner_create.net                = net;
    runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<>::Create();
    runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<>::Create();
    runner_create.optimizer          = bb::OptimizerAdam<>::Create();
    runner_create.max_run_size       = max_run_size;    // 実際の1回の実行サイズ
    runner_create.log_append         = true;
    runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
    runner_create.file_write         = true;            // 計算結果をファイルに保存するか
    runner_create.print_progress     = true;            // 途中結果を表示
    runner_create.initial_evaluation = true;            // ファイルを読んだ場合は最初に評価しておく
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}


void Cifar10DenseCnnTest(int epoch_size, int mini_batch_size, int max_run_size)
{
    Cifar10DenseCnn(epoch_size, mini_batch_size, max_run_size, 1,  false, false);
    Cifar10DenseCnn(epoch_size, mini_batch_size, max_run_size, 1,  true,  false);
    Cifar10DenseCnn(epoch_size, mini_batch_size, max_run_size, 3,  true,  false);
    Cifar10DenseCnn(epoch_size, mini_batch_size, max_run_size, 7,  true,  false);
    Cifar10DenseCnn(epoch_size, mini_batch_size, max_run_size, 15, true,  false);
    Cifar10DenseCnn(epoch_size, mini_batch_size, max_run_size, 31, true,  false);
}



#else

#if 0
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


#if 0

// Dense CNN
void Cifar10DenseCnn(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, int lut_frame_mux_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10DenseCnn_Lowering_mix";

    bool gen_rand = false;
    bool framewise= false;
    bool log_append = true;


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
#if 0
    cnv0_sub->Add(bb::MicroMlp<>::Create(384));
    cnv0_sub->Add(bb::MicroMlp<>::Create(64));
#else
    cnv0_sub->Add(bb::DenseAffine<>::Create(64));
    cnv0_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv0_sub->Add(bb::ReLU<>::Create());
#endif

    auto cnv1_sub = bb::Sequential::Create();
#if 0
    cnv1_sub->Add(bb::MicroMlp<>::Create(384));
    cnv1_sub->Add(bb::MicroMlp<>::Create(64));
#else
    cnv1_sub->Add(bb::DenseAffine<>::Create(64));
    cnv1_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv1_sub->Add(bb::ReLU<>::Create());
#endif

    auto cnv2_sub = bb::Sequential::Create();
#if 0
    cnv2_sub->Add(bb::MicroMlp<>::Create(768));
    cnv2_sub->Add(bb::MicroMlp<>::Create(128));
#else
    cnv2_sub->Add(bb::DenseAffine<>::Create(128));
    cnv2_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv2_sub->Add(bb::ReLU<>::Create());
#endif

    auto cnv3_sub = bb::Sequential::Create();
#if 0
    cnv3_sub->Add(bb::MicroMlp<>::Create(768));
    cnv3_sub->Add(bb::MicroMlp<>::Create(128));
#else
    cnv3_sub->Add(bb::DenseAffine<>::Create(128));
    cnv3_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv3_sub->Add(bb::ReLU<>::Create());
#endif

    // create network
    auto main_net = bb::Sequential::Create();
    if ( binary_mode ) {
        main_net->Add(bb::RealToBinary<>::Create(frame_mux_size));
    }
    main_net->Add(bb::LoweringConvolution<>::Create(cnv0_sub, 3, 3));
    main_net->Add(bb::LoweringConvolution<>::Create(cnv1_sub, 3, 3));
    main_net->Add(bb::MaxPooling<>::Create(2, 2));
    main_net->Add(bb::LoweringConvolution<>::Create(cnv2_sub, 3, 3));
    main_net->Add(bb::LoweringConvolution<>::Create(cnv3_sub, 3, 3));
    main_net->Add(bb::MaxPooling<>::Create(2, 2));
    
#if 0
    main_net->Add(bb::MicroMlp<>::Create(1860));
    main_net->Add(bb::MicroMlp<>::Create(310));
#else
    main_net->Add(bb::DenseAffine<>::Create(1024));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::ReLU<>::Create());
    main_net->Add(bb::DenseAffine<>::Create(30));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::ReLU<>::Create());
#endif

#if 0
    bb::BinaryModulation<float, float>::create_t mod_create;
    mod_create.layer                     = main_net;
    mod_create.output_shape              = td.t_shape;
    mod_create.training_modulation_size  = 1;
    mod_create.training_value_generator  = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 12345);
//  mod_create.training_value_generator  = bb::NormalDistributionGenerator<float>::Create(0.5f, 0.3f, 777);
    mod_create.inference_modulation_size = frame_mux_size;
    mod_create.inference_value_generator = nullptr;
    auto net = bb::BinaryModulation<float, float>::Create(mod_create);
#else
    main_net->Add(bb::Reduce<>::Create(td.t_shape));
    auto net = main_net;
#endif

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
    net->PrintInfo();

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
    runner_create.initial_evaluation = false;//true;            // ファイルを読んだ場合は最初に評価しておく
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}

#endif


#endif



void Cifar10DenseCnnX(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, int lut_frame_mux_size, bool binary_mode, bool file_read)
{
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 1,  binary_mode, file_read, false, false, false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 3,  binary_mode, file_read, false, false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 7,  binary_mode, file_read, false, false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 15, binary_mode, file_read, false, false);
//   Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 31, binary_mode, file_read, false, false);

//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 1,  binary_mode, file_read, true,  false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 3,  binary_mode, file_read, true,  false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 7,  binary_mode, file_read, true,  false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 15, binary_mode, file_read, true,  false);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 31, binary_mode, file_read, true,  false);

//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 1,  binary_mode, file_read, true,  true);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 3,  binary_mode, file_read, true,  true);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 7,  binary_mode, file_read, true,  true);
//    Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 15, binary_mode, file_read, true,  true);
//      Cifar10DenseCnnX(epoch_size, mini_batch_size, max_run_size, 31, binary_mode, file_read, true,  true);
}


// end of file
