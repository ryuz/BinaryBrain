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

#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/MicroMlp.h"
#include "bb/StochasticLut6.h"
#include "bb/BinaryLutN.h"
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
#include "bb/ExportVerilog.h"



// MNIST CNN with LUT networks
template <typename LAYER>
void MnistCompareRun(std::string net_name, int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto net = bb::Sequential::Create();
    net->Add(LAYER::Create({360}));
    net->Add(LAYER::Create({60}));
    net->Add(LAYER::Create({10}));
    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        net->SendCommand("binary true");
        std::cout << "binary mode" << std::endl;
    }

//  net->PrintInfo();

    // fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name           = net_name;
    runner_create.net            = net;
    runner_create.lossFunc       = bb::LossSoftmaxCrossEntropy<float>::Create();
    runner_create.metricsFunc    = bb::MetricsCategoricalAccuracy<float>::Create();
    runner_create.optimizer      = bb::OptimizerAdam<float>::Create();
    runner_create.initial_evaluation = true;
    runner_create.file_write     = true;
    runner_create.print_progress = true;
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}


void MnistCompare(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    MnistCompareRun< bb::StochasticLut6<> >("StochasticLut6", epoch_size, mini_batch_size, binary_mode);
    MnistCompareRun< bb::MicroMlp<6, 32> >("MicroMlp32", epoch_size, mini_batch_size, binary_mode);
    MnistCompareRun< bb::MicroMlp<6, 16> >("MicroMlp16", epoch_size, mini_batch_size, binary_mode);
    MnistCompareRun< bb::MicroMlp<6, 8> >("MicroMlp8", epoch_size, mini_batch_size, binary_mode);
    MnistCompareRun< bb::MicroMlp<6, 4> >("MicroMlp4", epoch_size, mini_batch_size, binary_mode);
}
