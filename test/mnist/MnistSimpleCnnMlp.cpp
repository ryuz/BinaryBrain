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

#include "bb/MicroMlp.h"
#include "bb/LoweringConvolution.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/MaxPooling.h"
#include "bb/LossCrossEntropyWithSoftmax.h"
#include "bb/AccuracyCategoricalClassification.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"



// MNIST CNN with LUT networks
void MnistSimpleCnnMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto cnn_sub0 = bb::Sequential::Create();
    cnn_sub0->Add(bb::MicroMlp<>::Create(256));
    cnn_sub0->Add(bb::MicroMlp<>::Create(64));
    cnn_sub0->Add(bb::MicroMlp<>::Create(16));

    auto cnn_sub1 = bb::Sequential::Create();
    cnn_sub1->Add(bb::MicroMlp<>::Create(256));
    cnn_sub1->Add(bb::MicroMlp<>::Create(64));
    cnn_sub1->Add(bb::MicroMlp<>::Create(16));

    auto net = bb::Sequential::Create();
    net->Add(bb::LoweringConvolution<>::Create(cnn_sub0, 16, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnn_sub1, 16, 3, 3));
    net->Add(bb::MaxPooling<>::Create(3, 3));
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({360}));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({60}));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({10}));

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
        std::cout << "binary mode" << std::endl;
        net->SendCommand("binary true");
    }

    std::mt19937_64 mt(1);

    for ( bb::index_t epoch = 0; epoch < epoch_size; ++epoch ) {
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
}


// end of file
