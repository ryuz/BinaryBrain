// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>

#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/DenseAffine.h"
#include "bb/MicroMlp.h"
#include "bb/LoweringConvolution.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/MaxPooling.h"
#include "bb/LossCrossEntropyWithSoftmax.h"
#include "bb/AccuracyCategoricalClassification.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"


#if 1

// MNIST CNN with LUT networks
void MnistSimpleCnnMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto cvn0_mm0 = bb::MicroMlp<>::Create(192);
    auto cvn0_mm1 = bb::MicroMlp<>::Create(32);

    auto cvn1_mm0 = bb::MicroMlp<>::Create(192);
    auto cvn1_mm1 = bb::MicroMlp<>::Create(32);

    auto cvn2_mm0 = bb::MicroMlp<>::Create(192);
    auto cvn2_mm1 = bb::MicroMlp<>::Create(32);

    auto cvn3_mm0 = bb::MicroMlp<>::Create(192);
    auto cvn3_mm1 = bb::MicroMlp<>::Create(32);

    auto cnv0_sub = bb::Sequential::Create();
    cnv0_sub->Add(cvn0_mm0);
    cnv0_sub->Add(cvn0_mm1);

    auto cnv1_sub = bb::Sequential::Create();
    cnv1_sub->Add(cvn1_mm0);
    cnv1_sub->Add(cvn1_mm1);

    auto cnv2_sub = bb::Sequential::Create();
    cnv2_sub->Add(cvn2_mm0);
    cnv2_sub->Add(cvn2_mm1);

    auto cnv3_sub = bb::Sequential::Create();
    cnv3_sub->Add(cvn3_mm0);
    cnv3_sub->Add(cvn3_mm1);

    auto net = bb::Sequential::Create();
    net->Add(bb::RealToBinary<>::Create(1));
    net->Add(bb::LoweringConvolution<>::Create(cnv0_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv1_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::LoweringConvolution<>::Create(cnv2_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv3_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::MicroMlp<>::Create({480}));
    net->Add(bb::MicroMlp<>::Create({80}));
    net->Add(bb::BinaryToReal<>::Create({ 10 }, 1));
    net->SetInputShape({28, 28, 1});

    std::cout << "binary mode" << std::endl;
    net->SendCommand("binary true");

//  net->SendCommand("host_only true");
//  net->SendCommand("host_only true", "BatchNormalization");

//  net->PrintInfo(2);
    net->PrintInfo();

    bb::Runner<float>::create_t runner_create;
    runner_create.name      = "MnistSimpleCnnMlp";
    runner_create.net       = net;
    runner_create.lossFunc  = bb::LossCrossEntropyWithSoftmax<float>::Create();
    runner_create.accFunc   = bb::AccuracyCategoricalClassification<float>::Create(10);
    runner_create.optimizer = bb::OptimizerAdam<float>::Create();
    runner_create.print_progress = true;
    runner_create.file_write = true;
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);

}

#else

// MNIST CNN with LUT networks
void MnistSimpleCnnMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto cnv0_sub = bb::Sequential::Create();
    cnv0_sub->Add(bb::MicroMlp<>::Create(192));
    cnv0_sub->Add(bb::MicroMlp<>::Create(22));

    auto cnv1_sub = bb::Sequential::Create();
    cnv1_sub->Add(bb::MicroMlp<>::Create(192));
    cnv1_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv2_sub = bb::Sequential::Create();
    cnv2_sub->Add(bb::MicroMlp<>::Create(192));
    cnv2_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv3_sub = bb::Sequential::Create();
    cnv3_sub->Add(bb::MicroMlp<>::Create(192));
    cnv3_sub->Add(bb::MicroMlp<>::Create(32));

    auto net = bb::Sequential::Create();
    net->Add(bb::RealToBinary<>::Create(1));
    net->Add(bb::LoweringConvolution<>::Create(cnv0_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv1_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv2_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv3_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(3, 3));
    net->Add(bb::MicroMlp<>::Create({480}));
    net->Add(bb::MicroMlp<>::Create({80}));
    net->Add(bb::BinaryToReal<>::Create({ 10 }, 1));
    net->SetInputShape({28, 28, 1});

    std::cout << "binary mode" << std::endl;
    net->SendCommand("binary true");

    bb::Runner<float>::create_t runner_create;
    runner_create.name      = "MnistSimpleCnnMlp";
    runner_create.net       = net;
    runner_create.lossFunc  = bb::LossCrossEntropyWithSoftmax<float>::Create();
    runner_create.accFunc   = bb::AccuracyCategoricalClassification<float>::Create(10);
    runner_create.optimizer = bb::OptimizerAdam<float>::Create();
    runner_create.print_progress = true;
    runner_create.file_write = true;
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);

}
#endif

// end of file
