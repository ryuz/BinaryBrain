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
#include "bb/MicroMlp.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
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
void MnistDeepMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    int const mux_size = 7;

  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
#else
    auto td = bb::LoadMnist<>::Load();
#endif

#if 0
    auto net = bb::Sequential::Create();
    net->Add(bb::RealToBinary<float, float>::Create(mux_size));
    net->Add(bb::MicroMlp<>::Create({512}));
    net->Add(bb::MicroMlp<>::Create({512}));
    net->Add(bb::MicroMlp<>::Create({512}));
    net->Add(bb::MicroMlp<>::Create({512}));
    net->Add(bb::MicroMlp<>::Create({512}));
    net->Add(bb::MicroMlp<>::Create({512}));
    net->Add(bb::MicroMlp<>::Create({420}));
    net->Add(bb::MicroMlp<>::Create({70}));
    net->Add(bb::BinaryToReal<float, float>::Create({10}, mux_size));
    net->SetInputShape(td.x_shape);
#endif
    auto net = bb::Sequential::Create();
    net->Add(bb::RealToBinary<float, float>::Create(mux_size));
    net->Add(bb::MicroMlp<>::Create({16, 16, 16}));
    net->Add(bb::MicroMlp<>::Create({16, 16, 16}));
    net->Add(bb::MicroMlp<>::Create({16, 16, 16}));
    net->Add(bb::MicroMlp<>::Create({16, 16, 16}));
    net->Add(bb::MicroMlp<>::Create({16, 16, 16}));
    net->Add(bb::MicroMlp<>::Create({16, 16, 16}));
    net->Add(bb::MicroMlp<>::Create({420}));
    net->Add(bb::MicroMlp<>::Create({70}));
    net->Add(bb::BinaryToReal<float, float>::Create({10}, mux_size));
    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        net->SendCommand("binary true");
        std::cout << "binary mode" << std::endl;
    }

    bb::Runner<float>::create_t runner_create;
    runner_create.name        = "MnistDeepMicroMlp";
    runner_create.net         = net;
    runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<float>::Create();
    runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<float>::Create();
    runner_create.optimizer   = bb::OptimizerAdam<float>::Create();
    runner_create.print_progress = true;
    runner_create.file_write = true;
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);
}

