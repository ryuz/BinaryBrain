﻿// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
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
#include "bb/MicroMlp.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LossCrossEntropyWithSoftmax.h"
#include "bb/AccuracyCategoricalClassification.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"



// MNIST CNN with LUT networks
void MnistSimpleLutMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto net = bb::Sequential::Create();
    net->Add(bb::RealToBinary<>::Create(7));
    net->Add(bb::MicroMlp<>::Create({1024}));
    net->Add(bb::MicroMlp<>::Create({360}));
    net->Add(bb::MicroMlp<>::Create({60}));
    net->Add(bb::BinaryToReal<float, float>::Create({10}, 7));
    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        net->SendCommand("binary true");
        std::cout << "binary mode" << std::endl;
    }

//  net->SendCommand("host_only true", "BatchNormalization");

    net->PrintInfo();

    // fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name      = "MnistSimpleLutMlp";
    runner_create.net       = net;
    runner_create.lossFunc  = bb::LossCrossEntropyWithSoftmax<float>::Create();
    runner_create.accFunc   = bb::AccuracyCategoricalClassification<float>::Create(10);
    runner_create.optimizer = bb::OptimizerAdam<float>::Create();
    runner_create.initial_evaluation = true;
    runner_create.print_progress = true;
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}

