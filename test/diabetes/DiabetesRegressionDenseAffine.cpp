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

#include "bb/DenseAffine.h"
#include "bb/MicroMlp.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/Sigmoid.h"
#include "bb/MetricsMeanSquaredError.h"
#include "bb/LossMeanSquaredError.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"

#include "LoadDiabetes.h"


void DiabetesAffineRegression(int epoch_size, size_t mini_batch_size)
{
    // load diabetes data
    auto td = LoadDiabetes<>();
    bb::TrainDataNormalize(td);
    
    auto net = bb::Sequential::Create();
    net->Add(bb::DenseAffine<>::Create({ 512 }));
    net->Add(bb::Sigmoid<>::Create());
    net->Add(bb::DenseAffine<>::Create({ 256 }));
    net->Add(bb::Sigmoid<>::Create());
    net->Add(bb::DenseAffine<>::Create({ 1 }));
//  net->Add(bb::Sigmoid<>::Create());
    net->SetInputShape({ 10 });

    bb::FrameBuffer x(BB_TYPE_FP32, mini_batch_size, { 10 });
    bb::FrameBuffer t(BB_TYPE_FP32, mini_batch_size, { 1 });

    bb::Runner<float>::create_t runner_create;
    runner_create.name        = "DiabetesAffineRegression";
    runner_create.net         = net;
    runner_create.lossFunc    = bb::LossMeanSquaredError<float>::Create();
    runner_create.metricsFunc = bb::MetricsMeanSquaredError<float>::Create();
//  runner_create.optimizer = bb::OptimizerSgd<float>::Create(0.0001f);
    runner_create.optimizer = bb::OptimizerAdam<float>::Create();
    runner_create.write_serial = false;
    runner_create.file_read  = false;
    runner_create.file_write = true;
    runner_create.print_progress = false;
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);
}

