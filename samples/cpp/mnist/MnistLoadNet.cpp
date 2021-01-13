// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <fstream>

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/Convolution2d.h"
#include "bb/MaxPooling.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/ModelLoader.h"


void MnistLoadNet(int epoch_size, int mini_batch_size, std::string filename)
{
    // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    // ネット読み込み
    auto net = bb::Model_LoadFromFile(filename);
    if (!net) {
        std::cerr << "file read error : " << filename << std::endl;
        return;
    }

    // set input shape
//  net->SetInputShape(td.x_shape);

    // print model information
    net->PrintInfo();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "epoch_size            : " << epoch_size            << std::endl;
    std::cout << "mini_batch_size       : " << mini_batch_size       << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name               = net->GetName();
    runner_create.net                = net;
    runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
    runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
    runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
    runner_create.print_progress     = true;       // 途中結果を表示
    runner_create.initial_evaluation = true;       // ファイルを読んだ場合は最初に評価しておく 
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}


// end of file
