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

#include "bb/BinaryModulation.h"
#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/SparseBinaryLutN.h"
#include "bb/StochasticLut6.h"
#include "bb/StochasticLutBn.h"
#include "bb/SparseLutN.h"
#include "bb/BinaryLutN.h"
#include "bb/DenseAffine.h"
#include "bb/LoweringConvolution.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/HardTanh.h"
#include "bb/MaxPooling.h"
#include "bb/Reduce.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerAdaGrad.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadCifar10.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"
#include "bb/ExportVerilog.h"
#include "bb/UniformDistributionGenerator.h"


#if 0

void Cifar10Sparse6Cnn(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10Sparse6Cnn";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    float bn_momentum = 0.0f;
    
#if 0
    auto cnv0_sub = bb::Sequential::Create();
    cnv0_sub->Add(bb::BinaryToReal<bb::Bit>::Create());
    cnv0_sub->Add(bb::DenseAffine<>::Create(32));
    cnv0_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv0_sub->Add(bb::Binarize<bb::Bit>::Create());
#else
    auto bin_cnv0_sub0 = bb::Sequential::Create();
    bin_cnv0_sub0->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create({1,  6, 8}));
    bin_cnv0_sub0->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create({1,  1, 8}));

    auto bin_cnv0_sub1 = bb::Sequential::Create();
    bin_cnv0_sub1->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create({1,  6, 16}));
    bin_cnv0_sub1->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create({1,  1, 16}));

    auto bin_cnv0_sub2p = bb::Sequential::Create();
    bin_cnv0_sub2p->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create({1,  6, 32}));
    bin_cnv0_sub2p->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create({1,  1, 32}));

    auto bin_cnv0_sub2d = bb::Sequential::Create();
    bin_cnv0_sub2d->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create({1,  6, 32}, "depthwise"));
    bin_cnv0_sub2d->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create({1,  1, 32}, "depthwise"));
#endif

    auto cnv1_sub = bb::Sequential::Create();
    cnv1_sub->Add(bb::BinaryToReal<bb::Bit>::Create());
    cnv1_sub->Add(bb::DenseAffine<>::Create(32));
    cnv1_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv1_sub->Add(bb::Binarize<bb::Bit>::Create());

    auto cnv2_sub = bb::Sequential::Create();
    cnv2_sub->Add(bb::BinaryToReal<bb::Bit>::Create());
    cnv2_sub->Add(bb::DenseAffine<>::Create(64));
    cnv2_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv2_sub->Add(bb::Binarize<bb::Bit>::Create());

    auto cnv3_sub = bb::Sequential::Create();
    cnv3_sub->Add(bb::BinaryToReal<bb::Bit>::Create());
    cnv3_sub->Add(bb::DenseAffine<>::Create(64));
    cnv3_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv3_sub->Add(bb::Binarize<bb::Bit>::Create());

    // create network
    auto main_net = bb::Sequential::Create();
#if 0
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(cnv0_sub, 3, 3));
#else
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(bin_cnv0_sub0,  3, 3, 1, 1, "valid"));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(bin_cnv0_sub1,  3, 3, 1, 1, "same"));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(bin_cnv0_sub2p, 1, 1, 1, 1, "same"));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(bin_cnv0_sub2d, 3, 3, 1, 1, "same"));
#endif
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(cnv1_sub, 3, 3));
    main_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(cnv2_sub, 3, 3));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(cnv3_sub, 3, 3));
    main_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));

    main_net->Add(bb::BinaryToReal<bb::Bit>::Create());
    main_net->Add(bb::DenseAffine<>::Create(512));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<bb::Bit>::Create());

    main_net->Add(bb::BinaryToReal<bb::Bit>::Create());
    main_net->Add(bb::DenseAffine<>::Create(10));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<bb::Bit>::Create());

    bb::BinaryModulation<bb::Bit>::create_t mod_create;
    mod_create.layer                     = main_net;
    mod_create.output_shape              = td.t_shape;
    mod_create.training_modulation_size  = frame_mux_size;
    mod_create.training_value_generator  = nullptr;
    mod_create.inference_modulation_size = frame_mux_size;
    mod_create.inference_value_generator = nullptr;
    auto net = bb::BinaryModulation<bb::Bit>::Create(mod_create);

    net->SetInputShape(td.x_shape);

    std::cout << "binary true" << std::endl;
    net->SendCommand("binary true");

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
    runner_create.name               = net_name;
    runner_create.net                = net;
    runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<>::Create();
    runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<>::Create();
    runner_create.optimizer          = bb::OptimizerAdam<>::Create();
    runner_create.max_run_size       = max_run_size;    // 実際の1回の実行サイズ
    runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
    runner_create.file_write         = true;            // 計算結果をファイルに保存するか
    runner_create.print_progress     = true;            // 途中結果を表示
    runner_create.initial_evaluation = false;//true;            // ファイルを読んだ場合は最初に評価しておく
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}


#else


// Dense CNN
void Cifar10Sparse6Cnn(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10Sparse6Cnn_dense_mix";
    
  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    float bn_momentum = 0.9f;
    if (binary_mode) {
        bn_momentum = 0.0f;
    }
    
    auto cnv0_sub = bb::Sequential::Create();
#if 1
    cnv0_sub->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(192));
    cnv0_sub->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(32));
#else
    cnv0_sub->Add(bb::BinaryToReal<bb::Bit>::Create());
    cnv0_sub->Add(bb::DenseAffine<>::Create(32));
    cnv0_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv0_sub->Add(bb::Binarize<bb::Bit>::Create());
#endif

    auto cnv1_sub = bb::Sequential::Create();
#if 1
    cnv1_sub->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(1152));
    cnv1_sub->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(192));
    cnv1_sub->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(32));
#else
    cnv1_sub->Add(bb::BinaryToReal<bb::Bit>::Create());
    cnv1_sub->Add(bb::DenseAffine<>::Create(32));
    cnv1_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv1_sub->Add(bb::Binarize<bb::Bit>::Create());
#endif

    auto cnv2_sub = bb::Sequential::Create();
#if 0
    cnv2_sub->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(768));
    cnv2_sub->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(128));
#else
    cnv2_sub->Add(bb::BinaryToReal<bb::Bit>::Create());
    cnv2_sub->Add(bb::DenseAffine<>::Create(64));
    cnv2_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv2_sub->Add(bb::Binarize<bb::Bit>::Create());
#endif

    auto cnv3_sub = bb::Sequential::Create();
#if 0
    cnv3_sub->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(768));
    cnv3_sub->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(128));
#else
    cnv3_sub->Add(bb::BinaryToReal<bb::Bit>::Create());
    cnv3_sub->Add(bb::DenseAffine<>::Create(64));
    cnv3_sub->Add(bb::BatchNormalization<>::Create(bn_momentum));
    cnv3_sub->Add(bb::Binarize<bb::Bit>::Create());
#endif

    // create network
    auto main_net = bb::Sequential::Create();
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(cnv0_sub, 3, 3));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(cnv1_sub, 3, 3));
    main_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));

    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(cnv2_sub, 3, 3));
    main_net->Add(bb::LoweringConvolution<bb::Bit>::Create(cnv3_sub, 3, 3));
    main_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));
    
#if 0
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(18432));
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(3072));
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(512));

    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(2160));
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(360));
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(60));
    main_net->Add(bb::SparseBinaryLutN<6, bb::Bit>::Create(10));
#else
    main_net->Add(bb::BinaryToReal<bb::Bit>::Create());
    main_net->Add(bb::DenseAffine<>::Create(512));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<bb::Bit>::Create());

    main_net->Add(bb::BinaryToReal<bb::Bit>::Create());
    main_net->Add(bb::DenseAffine<>::Create(10));
    main_net->Add(bb::BatchNormalization<>::Create(bn_momentum));
    main_net->Add(bb::Binarize<bb::Bit>::Create());
#endif

    // PWM
#if 1
    bb::BinaryModulation<bb::Bit, float>::create_t mod_create;
    mod_create.layer                     = main_net;
    mod_create.output_shape              = td.t_shape;
    mod_create.training_modulation_size  = frame_mux_size;
    mod_create.training_value_generator  = nullptr;
    mod_create.inference_modulation_size = frame_mux_size;
    mod_create.inference_value_generator = nullptr;
    auto net = bb::BinaryModulation<bb::Bit, float>::Create(mod_create);
#else
    auto net = main_net;
#endif

    net->SetInputShape(td.x_shape);

    std::cout << "binary true" << std::endl;
    net->SendCommand("binary true");


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
    runner_create.name               = net_name;
    runner_create.net                = net;
    runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<>::Create();
    runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<>::Create();
    runner_create.optimizer          = bb::OptimizerAdam<>::Create();
    runner_create.max_run_size       = max_run_size;    // 実際の1回の実行サイズ
    runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
    runner_create.file_write         = true;            // 計算結果をファイルに保存するか
    runner_create.print_progress     = true;            // 途中結果を表示
    runner_create.initial_evaluation = false;//true;            // ファイルを読んだ場合は最初に評価しておく
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}

#endif



// end of file
