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

#include "bb/BinaryModulation.h"
#include "bb/SparseLutN.h"
#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/MicroMlp.h"
#include "bb/StochasticLutN.h"
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
#include "bb/UniformDistributionGenerator.h"



// MNIST CNN with LUT networks
template <typename LAYER, typename T=bb::Bit>
void MnistCompareRun(std::string net_name, int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    auto main_net = bb::Sequential::Create();
    main_net->Add(LAYER::Create(360));
    main_net->Add(LAYER::Create(60));
    main_net->Add(LAYER::Create(10));

#if 0
    bb::BinaryModulation<T>::create_t c;
    c.layer = main_net;

    c.training_modulation_size  = 15;
    c.training_value_generator  = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f);
    c.training_framewise        = false;
    c.training_input_range_lo   = 0.0f;
    c.training_input_range_hi   = 1.0f;

    c.inference_modulation_size = 15;
    c.inference_value_generator = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f);
    c.inference_framewise       = false;
    c.inference_input_range_lo  = 0.0f;
    c.inference_input_range_hi  = 1.0f;   
    auto net = bb::BinaryModulation<T>::Create(c);
#else
    auto net = bb::BinaryModulation<T>::Create(main_net, 7);
#endif

    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        net->SendCommand("binary true");
        std::cout << "binary mode" << std::endl;
    }
    else {
        net->SendCommand("binary false");
        std::cout << "real mode" << std::endl;
    }

    net->SendCommand("lut_binarize false");
    

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
    MnistCompareRun< bb::MicroMlp<6, 16, bb::Bit>      >("MicroMlp16",     epoch_size, mini_batch_size, binary_mode);
    MnistCompareRun< bb::SparseLutN<6,   bb::Bit>     >("SparseLutN",     epoch_size, mini_batch_size, binary_mode);
    MnistCompareRun< bb::StochasticLutN<6>, float >("StochasticLutN", epoch_size, mini_batch_size, false);

    MnistCompareRun< bb::MicroMlp<6, 32, bb::Bit>      >("MicroMlp32",     epoch_size, mini_batch_size, binary_mode);
    MnistCompareRun< bb::MicroMlp<6, 8,  bb::Bit>       >("MicroMlp8",      epoch_size, mini_batch_size, binary_mode);
    MnistCompareRun< bb::MicroMlp<6, 4,  bb::Bit>       >("MicroMlp4",      epoch_size, mini_batch_size, binary_mode);
}
