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
#include "bb/MicroMlpAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/Sigmoid.h"
#include "bb/ReLU.h"
#include "bb/LossMeanSquaredError.h"
#include "bb/MetricsBinaryAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/LoadXor.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"


// MNIST CNN with LUT networks
void XorMicroMlp(int epoch_size, bool binary_mode)
{
    // load data
	auto td = bb::LoadXor<>::Load(6, 256);

    /*
    for (int i = 0; i < 64; ++i) {
        std::cout << td.t_train[i][0] << " : ";
        for (int j = 0; j < 6; ++j) {
            std::cout << td.x_train[i][j] << " ";
        }
        std::cout << std::endl;
    }
    */

    auto net = bb::Sequential::Create();
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create(td.t_shape));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::Sigmoid<float>::Create());
    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        net->SendCommand("binary true");
        std::cout << "binary mode" << std::endl;
    }

    bb::Runner<float>::create_t runner_create;
    runner_create.name        = "XorMicroMlp";
    runner_create.net         = net;
    runner_create.lossFunc    = bb::LossMeanSquaredError<float>::Create();
    runner_create.metricsFunc = bb::MetricsBinaryAccuracy<float>::Create();
    runner_create.optimizer   = bb::OptimizerAdam<float>::Create();
    runner_create.print_progress = true;
    runner_create.file_write = true;
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, (1 << 6));
}

