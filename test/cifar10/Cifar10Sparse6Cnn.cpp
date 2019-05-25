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


#if 1

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

#endif



#if 0

void Cifar10Sparse6Cnn(int epoch_size, int mini_batch_size, int max_run_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10Sparse6Cnn_X";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif


    std::cout << "binary mode : " << binary_mode << std::endl;

    int frame_mux_size = 1;

    {
        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 6, 8}));
        cnv0_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 1, 8}));

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 6, 8}));
        cnv1_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 1, 8}));

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 6, 8}));
        cnv2_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 1, 8}));

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 6, 8}));
        cnv3_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1, 1, 8}));

        auto cnv4p_sub = bb::Sequential::Create();
        cnv4p_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1, 96}));
        cnv4p_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1, 16}));
        auto cnv4d_sub = bb::Sequential::Create();
        cnv4d_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  6, 16}, "depthwise"));
        cnv4d_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1, 16}, "depthwise"));

        auto cnv5p_sub = bb::Sequential::Create();
        cnv5p_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1, 96}));
        cnv5p_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1, 16}));
        auto cnv5d_sub = bb::Sequential::Create();
        cnv5d_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  6, 16}, "depthwise"));
        cnv5d_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1, 16}, "depthwise"));

        auto cnv6p_sub = bb::Sequential::Create();
        cnv6p_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1, 192}));
        cnv6p_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1,  32}));
        auto cnv6d_sub = bb::Sequential::Create();
        cnv6d_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  6, 32}, "depthwise"));
        cnv6d_sub->Add(bb::SparseLutN<6, bb::Bit>::Create({1,  1, 32}, "depthwise"));

        auto cnv4_sub = bb::Sequential::Create();
        cnv4_sub->Add(bb::SparseLutN<6>::Create({1, 36, 16}));
        cnv4_sub->Add(bb::SparseLutN<6>::Create({1,  6, 16}));
        cnv4_sub->Add(bb::SparseLutN<6>::Create({1,  1, 16}));

        auto cnv5_sub = bb::Sequential::Create();
        cnv5_sub->Add(bb::SparseLutN<6>::Create({1, 36, 16}));
        cnv5_sub->Add(bb::SparseLutN<6>::Create({1,  6, 16}));
        cnv5_sub->Add(bb::SparseLutN<6>::Create({1,  1, 16}));


        
        auto net = bb::Sequential::Create();
        net->Add(bb::RealToBinary<>::Create(frame_mux_size, bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1)));
        net->Add(bb::LoweringConvolution<>::Create(cnv0_sub, 3, 3));        // 30
        net->Add(bb::LoweringConvolution<>::Create(cnv1_sub, 3, 3));        // 28
        net->Add(bb::LoweringConvolution<>::Create(cnv2_sub, 3, 3));        // 26
        net->Add(bb::MaxPooling<>::Create(2, 2));

        net->Add(bb::LoweringConvolution<>::Create(cnv3p_sub, 1, 1));       // 13
        net->Add(bb::LoweringConvolution<>::Create(cnv3d_sub, 3, 3, 1));    // 11
        net->Add(bb::LoweringConvolution<>::Create(cnv4p_sub, 1, 1));       // 1
        net->Add(bb::LoweringConvolution<>::Create(cnv4d_sub, 3, 3, 1));    // 22
        net->Add(bb::LoweringConvolution<>::Create(cnv5p_sub, 1, 1));       // 22
        net->Add(bb::LoweringConvolution<>::Create(cnv5d_sub, 3, 3, 1));    // 20
        
        net->Add(bb::MaxPooling<>::Create(2, 2));
        net->Add(bb::SparseLutN<>::Create(512));
        net->Add(bb::SparseLutN<>::Create(512));
        net->Add(bb::SparseLutN<>::Create(150));
        net->Add(bb::Reduce<>::Create(td.t_shape));
        net->Add(bb::BinaryToReal<>::Create(frame_mux_size, td.t_shape));

        net->SetInputShape(td.x_shape);

        net->SendCommand("lut_binarize false");

        if (binary_mode) {
            net->SendCommand("binary true");
        }
        else {
            net->SendCommand("binary false");
        }

        // print model information
        net->PrintInfo();

        // run fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name               = net_name;
        runner_create.net                = net;
        runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
//      runner_create.optimizer          = bb::OptimizerAdaGrad<float>::Create();
        runner_create.max_run_size       = max_run_size;    // ・ｽ・ｽ・ｽﾛゑｿｽ1・ｽ・ｽﾌ趣ｿｽ・ｽs・ｽT・ｽC・ｽY
        runner_create.file_read          = file_read;       // ・ｽO・ｽﾌ計・ｽZ・ｽ・ｽ・ｽﾊゑｿｽ・ｽ・ｽ・ｽ・ｽﾎ読み搾ｿｽ・ｽ・ｽﾅ再開・ｽ・ｽ・ｽ驍ｩ
        runner_create.file_write         = true;            // ・ｽv・ｽZ・ｽ・ｽ・ｽﾊゑｿｽ・ｽt・ｽ@・ｽC・ｽ・ｽ・ｽﾉ保托ｿｽ・ｽ・ｽ・ｽ驍ｩ
        runner_create.print_progress     = true;            // ・ｽr・ｽ・ｽ・ｽ・ｽ・ｽﾊゑｿｽ\・ｽ・ｽ
        runner_create.initial_evaluation = false; // file_read;       // ・ｽt・ｽ@・ｽC・ｽ・ｽ・ｽ・ｽﾇんだ場合・ｽﾍ最擾ｿｽ・ｽﾉ評・ｽ・ｽ・ｽ・ｽ・ｽﾄゑｿｽ・ｽ・ｽ
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }
}

#endif



// end of file
