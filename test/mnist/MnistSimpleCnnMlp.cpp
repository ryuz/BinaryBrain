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
#include "bb/Runner.h"


// MNIST CNN with LUT networks
void MnistSimpleCnnMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto cnv0_sub = bb::Sequential::Create();
    cnv0_sub->Add(bb::MicroMlp<>::Create(512));
    cnv0_sub->Add(bb::MicroMlp<>::Create(128));
    cnv0_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv1_sub = bb::Sequential::Create();
    cnv1_sub->Add(bb::MicroMlp<>::Create(512));
    cnv1_sub->Add(bb::MicroMlp<>::Create(128));
    cnv1_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv2_sub = bb::Sequential::Create();
    cnv2_sub->Add(bb::MicroMlp<>::Create(512));
    cnv2_sub->Add(bb::MicroMlp<>::Create(128));
    cnv2_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv3_sub = bb::Sequential::Create();
    cnv3_sub->Add(bb::MicroMlp<>::Create(512));
    cnv3_sub->Add(bb::MicroMlp<>::Create(128));
    cnv3_sub->Add(bb::MicroMlp<>::Create(32));

    auto net = bb::Sequential::Create();
    net->Add(bb::LoweringConvolution<>::Create(cnv0_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv1_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv2_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv3_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(3, 3));
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({360}));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({60}));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::MicroMlpAffine<6, 16, float>::Create({10}));
    net->SetInputShape({28, 28, 1});

    if ( binary_mode ) {
        std::cout << "binary mode" << std::endl;
        net->SendCommand("binary true");
    }

    bb::Runner<float>::create_t runner_create;
    runner_create.name      = "MnistSimpleCnnMlp";
    runner_create.net       = net;
    runner_create.lossFunc  = bb::LossCrossEntropyWithSoftmax<float>::Create();
    runner_create.accFunc   = bb::AccuracyCategoricalClassification<float>::Create(10);
    runner_create.optimizer = bb::OptimizerAdam<float>::Create();
    runner_create.serial_write = false;
    runner_create.initial_evaluation = false;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);

}


// end of file
