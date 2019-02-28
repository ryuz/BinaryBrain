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




void MnistDenseAffine(int epoch_size, size_t mini_batch_size)
{
  // load MNIST data
#ifdef _DEBUG
	auto data = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto data = bb::LoadMnist<>::Load(10);
#endif

    auto net = bb::Sequential::Create();
    net->Add(bb::DenseAffine<float>::Create({256}));
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::DenseAffine<float>::Create({10}));

    bb::LossCrossEntropyWithSoftmax<float>          lossFunc;
    bb::AccuracyCategoricalClassification<float>    accFunc(10);

    net->SetInputShape({28, 28, 1});

    bb::FrameBuffer x(BB_TYPE_FP32, mini_batch_size, {28, 28, 1});
    bb::FrameBuffer t(BB_TYPE_FP32, mini_batch_size, 10);

//  bb::OptimizerAdam<float> optimizer;
    bb::OptimizerSgd<float> optimizer(0.001f);

    optimizer.SetVariables(net->GetParameters(), net->GetGradients());

    std::mt19937_64 mt(1);

    for ( bb::index_t epoch = 0; epoch < epoch_size; ++epoch ) {
        lossFunc.Clear();
        accFunc.Clear();
        for (bb::index_t i = 0; i < (bb::index_t)(data.x_train.size() - mini_batch_size); i += mini_batch_size)
        {
            x.SetVector(data.x_train, i);
            t.SetVector(data.y_train, i);

            auto y = net->Forward(x);
            
            auto dy = lossFunc.CalculateLoss(y, t);
            accFunc.CalculateAccuracy(y, t);

            dy = net->Backward(dy);

            optimizer.Update();
        }
        std::cout << "train loss : " << lossFunc.GetLoss() <<  "  accuracy : " << accFunc.GetAccuracy() << std::endl;

        // test
        lossFunc.Clear();
        accFunc.Clear();
        for (bb::index_t i = 0; i < (bb::index_t)(data.x_test.size() - mini_batch_size); i += mini_batch_size)
        {
            x.SetVector(data.x_test, i);
            t.SetVector(data.y_test, i);

            auto y = net->Forward(x);
            
            auto dy = lossFunc.CalculateLoss(y, t);
            accFunc.CalculateAccuracy(y, t);
        }
        std::cout << "test loss : " << lossFunc.GetLoss() <<  "  accuracy : " << accFunc.GetAccuracy() << std::endl;

        bb::ShuffleDataSet(mt(), data.x_train, data.y_train);
    }
}

