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
void MnistSequentialMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto net = bb::Sequential::Create();
    net->Add(bb::RealToBinary<float, float>::Create(1));
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({1024}));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({360}));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({60}));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({10}));
    net->Add(bb::BinaryToReal<float, float>::Create({10}, 1));
    net->SetInputShape(td.x_shape);

    if ( binary_mode ) {
        net->SendCommand("binary true");
        std::cout << "binary mode" << std::endl;
    }

    bb::Runner<float>::create_t runner_create;
    runner_create.name        = "MnistSequentialMicroMlp";
    runner_create.net         = net;
    runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<float>::Create();
    runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<float>::Create();
    runner_create.optimizer   = bb::OptimizerAdam<float>::Create();
    runner_create.print_progress = false; // true;
    runner_create.file_write = true;
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);


#if 0
    auto lossFunc = bb::LossCrossEntropyWithSoftmax<float>::Create();
    auto accFunc  = bb::AccuracyCategoricalClassification<float>::Create(10);

    net->SetInputShape({28, 28, 1});

    bb::FrameBuffer x(BB_TYPE_FP32, mini_batch_size, {28, 28, 1});
    bb::FrameBuffer t(BB_TYPE_FP32, mini_batch_size, 10);

    auto optimizer = bb::OptimizerAdam<float>::Create();
//  auto optimizer = bb::OptimizerSgd<float>::Create(0.001f);

    optimizer->SetVariables(net->GetParameters(), net->GetGradients());


//  net.SendCommand("host_only true", "MicroMlpAffine");
//  net.SendCommand("host_only true", "ReLU");
//  net.SendCommand("host_only true");

    if ( binary_mode ) {
        net->SendCommand("binary true");
    }

    std::mt19937_64 mt(1);

    for ( bb::index_t epoch = 0; epoch < epoch_size; ++epoch ) {

#if 0
        if (epoch == 4) {
            net.Command("host_only true");
            std::cout << "host_only true" << std::endl;
        }
        if (epoch == 10) {
            net.Command("host_only false");
            std::cout << "host_only false" << std::endl;
        }
#endif

        accFunc->Clear();
        for (bb::index_t i = 0; i < (bb::index_t)(td.x_train.size() - mini_batch_size); i += mini_batch_size)
        {
            x.SetVector(td.x_train, i);
            t.SetVector(td.t_train, i);

            auto y = net->Forward(x);
            
            auto dy = lossFunc->CalculateLoss(y, t);
            accFunc->CalculateAccuracy(y, t);

            dy = net->Backward(dy);

            optimizer->Update();
        }
        std::cout << "accuracy : " << accFunc->GetAccuracy() << std::endl;

        bb::ShuffleDataSet(mt(), td.x_train, td.t_train);
    }
#endif
}

