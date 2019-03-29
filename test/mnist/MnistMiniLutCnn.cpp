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
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"



// MNIST CNN with LUT networks
void MnistMiniLutCnn(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    std::string net_name = "MnistMiniLutCnn";
    int const mux_size = 7;

    // load MNIST data
#ifdef _DEBUG
	  auto td = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto cnv0_sub = bb::Sequential::Create();
    cnv0_sub->Add(bb::MicroMlp<>::Create(128));
    cnv0_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv1_sub = bb::Sequential::Create();
    cnv1_sub->Add(bb::MicroMlp<>::Create(128));
    cnv1_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv2_sub = bb::Sequential::Create();
    cnv2_sub->Add(bb::MicroMlp<>::Create(128));
    cnv2_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv3_sub = bb::Sequential::Create();
    cnv3_sub->Add(bb::MicroMlp<>::Create(128));
    cnv3_sub->Add(bb::MicroMlp<>::Create(32));

    auto net = bb::Sequential::Create();
    net->Add(bb::RealToBinary<>::Create(mux_size));
    net->Add(bb::LoweringConvolution<>::Create(cnv0_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv1_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::LoweringConvolution<>::Create(cnv2_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv3_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::MicroMlp<>::Create({512}));
    net->Add(bb::MicroMlp<>::Create({420}));
    net->Add(bb::MicroMlp<>::Create({70}));
    net->Add(bb::BinaryToReal<>::Create({ 10 }, mux_size));
    net->SetInputShape({28, 28, 1});

    std::cout << "binary mode" << std::endl;
    net->SendCommand("binary true");

//  net->PrintInfo(2);
    net->PrintInfo();

    bb::Runner<float>::create_t runner_create;
    runner_create.name        = net_name;
    runner_create.net         = net;
    runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<float>::Create();
    runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<float>::Create();
    runner_create.optimizer   = bb::OptimizerAdam<float>::Create();
    runner_create.print_progress = true;
    runner_create.print_progress_accuracy = false;
    runner_create.print_progress_loss = false;
    runner_create.file_write = true;
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);

}


// end of file
