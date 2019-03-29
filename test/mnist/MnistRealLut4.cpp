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
#include "bb/RealLut4.h"
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
void MnistRealLut4(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif
    
    auto net = bb::Sequential::Create();
    net->Add(bb::RealLut4<>::Create({2560}));
    net->Add(bb::RealLut4<>::Create({640}));
    net->Add(bb::RealLut4<>::Create({160}));
    net->Add(bb::RealLut4<>::Create({40}));
    net->Add(bb::RealLut4<>::Create({10}));
    net->SetInputShape(td.x_shape);
    
    bb::Runner<float>::create_t runner_create;
    runner_create.name        = "MnistRealLut4";
    runner_create.net         = net;
    runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<float>::Create();
    runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<float>::Create();
    runner_create.optimizer   = bb::OptimizerSgd<float>::Create(0.0001);
    runner_create.print_progress = true;
    runner_create.file_write = true;
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);
    
    runner->Fitting(td, epoch_size, mini_batch_size);
}

