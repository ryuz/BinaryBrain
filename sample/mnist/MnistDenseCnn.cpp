// --------------------------------------------------------------------------
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

#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LoweringConvolution.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MaxPooling.h"
#include "bb/AccuracyCategoricalClassification.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"



void MnistDenseCnn(int epoch_size, size_t mini_batch_size)
{
  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(10, 512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    // create network
    auto net = bb::Sequential::Create();
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(32), 3, 3));
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(32), 3, 3));
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3));
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::LoweringConvolution<>::Create(bb::DenseAffine<>::Create(64), 3, 3));
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::DenseAffine<float>::Create({256}));
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::DenseAffine<float>::Create(td.t_shape));
    net->SetInputShape(td.x_shape);
    
    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name      = "MnistDenseCnn";
    runner_create.net       = net;
    runner_create.lossFunc  = bb::LossSoftmaxCrossEntropy<float>::Create();
    runner_create.accFunc   = bb::AccuracyCategoricalClassification<float>::Create();
    runner_create.optimizer = bb::OptimizerAdam<float>::Create();
    runner_create.initial_evaluation = true;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);
}

