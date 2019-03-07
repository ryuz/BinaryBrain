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
#include "bb/LossCrossEntropyWithSoftmax.h"
#include "bb/AccuracyCategoricalClassification.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"



void MnistDenseAffine(int epoch_size, size_t mini_batch_size)
{
  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto net = bb::Sequential::Create();
    net->Add(bb::DenseAffine<float>::Create({256}));
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::DenseAffine<float>::Create({10}));

    net->SetInputShape({28, 28, 1});

    bb::FrameBuffer x(BB_TYPE_FP32, mini_batch_size, {28, 28, 1});
    bb::FrameBuffer t(BB_TYPE_FP32, mini_batch_size, 10);

    bb::Runner<float>::create_t runner_create;
    runner_create.name      = "MnistDenseAffine";
    runner_create.net       = net;
    runner_create.lossFunc  = bb::LossCrossEntropyWithSoftmax<float>::Create();
    runner_create.accFunc   = bb::AccuracyCategoricalClassification<float>::Create(10);
    runner_create.optimizer = bb::OptimizerAdam<float>::Create();
    runner_create.serial_write = false;
    runner_create.initial_evaluation = true;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);
}

