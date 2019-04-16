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


class MnistSimpleMicroMlpNet : public bb::Model
{
protected:
    using Affine      = bb::MicroMlpAffine<6, 16, float>;
    using AffinePtr   = std::shared_ptr<Affine>;
    using Activate    = bb::ReLU<float>;
    using ActivatePtr = std::shared_ptr<Activate>;


public:
    AffinePtr   m_affine0;
    ActivatePtr m_activate0;
    AffinePtr   m_affine1;
    ActivatePtr m_activate1;
    AffinePtr   m_affine2;
    ActivatePtr m_activate2;
    AffinePtr   m_affine3;

public:

    MnistSimpleMicroMlpNet()
    {
        m_affine0   = Affine::Create({1024});
        m_activate0 = Activate::Create();
        m_affine1   = Affine::Create({360});
        m_activate1 = Activate::Create();
        m_affine2   = Affine::Create({60});
        m_activate2 = Activate::Create();
        m_affine3   = Affine::Create({10});
    }

    std::string GetClassName(void) const
    {
        return "MnistSimpleMicroMlpNet";
    }

    void SendCommand(std::string command, std::string send_to = "all")
    {
        m_affine0->SendCommand(command, send_to);
        m_activate0->SendCommand(command, send_to);
        m_affine1->SendCommand(command, send_to);
        m_activate1->SendCommand(command, send_to);
        m_affine2->SendCommand(command, send_to);
        m_activate2->SendCommand(command, send_to);
        m_affine3->SendCommand(command, send_to);
    }

    bb::indices_t SetInputShape(bb::indices_t shape)
    {
        shape = m_affine0->SetInputShape(shape);
        shape = m_activate0->SetInputShape(shape);
        shape = m_affine1->SetInputShape(shape);
        shape = m_activate1->SetInputShape(shape);
        shape = m_affine2->SetInputShape(shape);
        shape = m_activate2->SetInputShape(shape);
        shape = m_affine3->SetInputShape(shape);
        return shape;
    }

    bb::indices_t GetInputShape(void) const
    {
        return m_affine0->GetInputShape();
    }

    bb::indices_t GetOutputShape(void) const
    {
        return m_affine3->GetInputShape();
    }

    bb::Variables GetParameters(void)
    {
        bb::Variables var;
        var.PushBack(m_affine0->GetParameters());
        var.PushBack(m_activate0->GetParameters());
        var.PushBack(m_affine1->GetParameters());
        var.PushBack(m_activate2->GetParameters());
        var.PushBack(m_affine2->GetParameters());
        var.PushBack(m_activate2->GetParameters());
        var.PushBack(m_affine3->GetParameters());
        return var;
    }

    bb::Variables GetGradients(void)
    {
        bb::Variables var;
        var.PushBack(m_affine0->GetGradients());
        var.PushBack(m_activate0->GetGradients());
        var.PushBack(m_affine1->GetGradients());
        var.PushBack(m_activate1->GetGradients());
        var.PushBack(m_affine2->GetGradients());
        var.PushBack(m_activate2->GetGradients());
        var.PushBack(m_affine3->GetGradients());
        return var;
    }

    bb::FrameBuffer Forward(bb::FrameBuffer x, bool train=true)
    {
        x = m_affine0->Forward(x, train);
        x = m_activate0->Forward(x, train);
        x = m_affine1->Forward(x, train);
        x = m_activate1->Forward(x, train);
        x = m_affine2->Forward(x, train);
        x = m_activate2->Forward(x, train);
        x = m_affine3->Forward(x, train);
        return x;
    }

    bb::FrameBuffer Backward(bb::FrameBuffer dy)
    {
        dy = m_affine3->Backward(dy);
        dy = m_activate2->Backward(dy);
        dy = m_affine2->Backward(dy);
        dy = m_activate1->Backward(dy);
        dy = m_affine1->Backward(dy);
        dy = m_activate0->Backward(dy);
        dy = m_affine0->Backward(dy);
        return dy;
    }
};


// MNIST CNN with LUT networks
void MnistMicroMlpScratch(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif
    
    MnistSimpleMicroMlpNet  net;
    net.SetInputShape(td.x_shape);

    auto lossFunc    = bb::LossSoftmaxCrossEntropy<float>::Create();
    auto metricsFunc = bb::MetricsCategoricalAccuracy<float>::Create();
    
    bb::FrameBuffer x(BB_TYPE_FP32, mini_batch_size, {28, 28, 1});
    bb::FrameBuffer t(BB_TYPE_FP32, mini_batch_size, 10);
    
    auto optimizer = bb::OptimizerAdam<float>::Create();
    optimizer->SetVariables(net.GetParameters(), net.GetGradients());

    std::mt19937_64 mt(1);

    if ( binary_mode ) {
        net.SendCommand("binary true");
    }

    for ( bb::index_t epoch = 0; epoch < epoch_size; ++epoch ) {
        metricsFunc->Clear();
        for (bb::index_t i = 0; i < (bb::index_t)(td.x_train.size() - mini_batch_size); i += mini_batch_size)
        {
            x.SetVector(td.x_train, i);
            t.SetVector(td.t_train, i);
            
            auto y = net.Forward(x);
            
            auto dy = lossFunc->CalculateLoss(y, t, y.GetFrameSize());
            metricsFunc->CalculateMetrics(y, t);
            
            dy = net.Backward(dy);
            
            optimizer->Update();            
        }
        std::cout << "accuracy : " << metricsFunc->GetMetrics() << std::endl;

        bb::ShuffleDataSet(mt(), td.x_train, td.t_train);
    }
}


// end of file
