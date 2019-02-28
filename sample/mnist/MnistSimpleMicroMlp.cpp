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

#include "bb/MicroMlpAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LossCrossEntropyWithSoftmax.h"
#include "bb/AccuracyCategoricalClassification.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"


class MnistSimpleMicroMlpNet : public bb::Layer
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
void MnistSimpleMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    // load MNIST data
#ifdef _DEBUG
	auto data = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto data = bb::LoadMnist<>::Load(10);
#endif
    
    MnistSimpleMicroMlpNet  cpu_net;
    auto cpu_lossFunc = bb::LossCrossEntropyWithSoftmax<float>::Create();
    auto cpu_accFunc  = bb::AccuracyCategoricalClassification<float>::Create(10);

    MnistSimpleMicroMlpNet  gpu_net;
    auto gpu_lossFunc = bb::LossCrossEntropyWithSoftmax<float>::Create();
    auto gpu_accFunc  = bb::AccuracyCategoricalClassification<float>::Create(10);

    cpu_net.SetInputShape({28, 28, 1});
    gpu_net.SetInputShape({28, 28, 1});

    bb::FrameBuffer cpu_x(BB_TYPE_FP32, mini_batch_size, {28, 28, 1});
    bb::FrameBuffer gpu_x(BB_TYPE_FP32, mini_batch_size, {28, 28, 1});
    bb::FrameBuffer cpu_t(BB_TYPE_FP32, mini_batch_size, 10);
    bb::FrameBuffer gpu_t(BB_TYPE_FP32, mini_batch_size, 10);


    auto cpu_optimizer = bb::OptimizerAdam<float>::Create();
    auto gpu_optimizer = bb::OptimizerAdam<float>::Create();

//  auto cpu_optimizer = bb::OptimizerSgd<float>::Create(0.001f);
//  auto gpu_optimizer = bb::OptimizerSgd<float>::Create(0.001f);

    cpu_optimizer->SetVariables(cpu_net.GetParameters(), cpu_net.GetGradients());
    gpu_optimizer->SetVariables(gpu_net.GetParameters(), gpu_net.GetGradients());

    std::mt19937_64 mt(1);

    if ( binary_mode ) {
        cpu_net.SendCommand("binary true");
        gpu_net.SendCommand("binary true");
    }

    for ( bb::index_t epoch = 0; epoch < epoch_size; ++epoch ) {
        cpu_accFunc->Clear();
        gpu_accFunc->Clear();
        for (bb::index_t i = 0; i < (bb::index_t)(data.x_train.size() - mini_batch_size); i += mini_batch_size)
        {
            cpu_x.SetVector(data.x_train, i);
            gpu_x.SetVector(data.x_train, i);
            cpu_t.SetVector(data.y_train, i);
            gpu_t.SetVector(data.y_train, i);

            auto cpu_y = cpu_net.Forward(cpu_x);
            auto gpu_y = gpu_net.Forward(gpu_x);
            
            auto cpu_dy = cpu_lossFunc->CalculateLoss(cpu_y, cpu_t);
            auto gpu_dy = gpu_lossFunc->CalculateLoss(gpu_y, gpu_t);

            cpu_accFunc->CalculateAccuracy(cpu_y, cpu_t);
            gpu_accFunc->CalculateAccuracy(gpu_y, gpu_t);

            cpu_dy = cpu_net.Backward(cpu_dy);
            gpu_dy = gpu_net.Backward(gpu_dy);

            cpu_optimizer->Update();
            gpu_optimizer->Update();
        }
        std::cout << "cpu : " << cpu_accFunc->GetAccuracy() << std::endl;
        std::cout << "gpu : " << gpu_accFunc->GetAccuracy() << std::endl;

        bb::ShuffleDataSet(mt(), data.x_train, data.y_train);
    }
}


// end of file
