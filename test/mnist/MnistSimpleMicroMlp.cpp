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

public:

    MnistSimpleMicroMlpNet()
    {
        m_affine0   = Affine::Create({360});
        m_activate0 = Activate::Create();
        m_affine1   = Affine::Create({60});
        m_activate1 = Activate::Create();
        m_affine2   = Affine::Create({10});
    }

    std::string GetClassName(void) const
    {
        return "MnistSimpleMicroMlpNet";
    }

    bb::indices_t SetInputShape(bb::indices_t shape)
    {
        shape = m_affine0->SetInputShape(shape);
        shape = m_activate0->SetInputShape(shape);
        shape = m_affine1->SetInputShape(shape);
        shape = m_activate1->SetInputShape(shape);
        shape = m_affine2->SetInputShape(shape);
        return shape;
    }

    bb::Variables GetParameters(void)
    {
        bb::Variables var;
        var.PushBack(m_affine0->GetParameters());
        var.PushBack(m_activate0->GetParameters());
        var.PushBack(m_affine1->GetParameters());
        var.PushBack(m_activate1->GetParameters());
        var.PushBack(m_affine2->GetParameters());
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
        return var;
    }

    bb::FrameBuffer Forward(bb::FrameBuffer x, bool train=true)
    {
        x = m_affine0->Forward(x, train);
        x = m_activate0->Forward(x, train);
        x = m_affine1->Forward(x, train);
        x = m_activate1->Forward(x, train);
        x = m_affine2->Forward(x, train);
        return x;
    }

    bb::FrameBuffer Backward(bb::FrameBuffer dy)
    {
        dy = m_affine2->Backward(dy);
        dy = m_activate1->Backward(dy);
        dy = m_affine1->Backward(dy);
        dy = m_activate0->Backward(dy);
        dy = m_affine0->Backward(dy);
        return dy;
    }
};

void printBuffer(std::string name, bb::FrameBuffer buf)
{
    std::cout << name << " =\n";
    auto ptr = buf.GetConstPtr<float>();
    for (int f = 0; f < 3; f++) {
        for (int i = 0; i < 10; i++) {
            std::cout << ptr.Get(f, i) << ", ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

template<typename T>
void printTensorPtr(std::string name,  bb::TensorConstPtr_<T, bb::Tensor const, bb::Memory::ConstPtr> ptr)
{
    std::cout << name << " =\n";
    for (int i = 0; i < 10; i++) {
       std::cout << ptr[i] << ", ";
    }
    std::cout << std::endl;
}


// MNIST CNN with LUT networks
void MnistSimpleMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    // load MNIST data
#ifdef _DEBUG
	auto data = bb::LoadMnist<>::Load(10, 512, 128);
#else
    auto data = bb::LoadMnist<>::Load(10);
#endif

    MnistSimpleMicroMlpNet                          net;
    bb::LossCrossEntropyWithSoftmax<float>          lossFunc;
    bb::AccuracyCategoricalClassification<float>    accFunc(10);

    net.SetInputShape({28, 28, 1});

    bb::FrameBuffer x(BB_TYPE_FP32, mini_batch_size, {28, 28, 1});
    bb::FrameBuffer t(BB_TYPE_FP32, mini_batch_size, 10);

    bb::OptimizerAdam<float> optimizer;
//  bb::OptimizerSgd<float> optimizer;
    optimizer.SetVariables(net.GetParameters(), net.GetGradients());

    std::mt19937_64 mt(1);

    for ( bb::index_t epoch = 0; epoch < 32; ++epoch ) {
        double acc = 0;
        for (bb::index_t i = 0; i < (bb::index_t)(data.x_train.size() - mini_batch_size); i += mini_batch_size)
        {
            x.SetVector(data.x_train, i);
            t.SetVector(data.y_train, i);

            auto y = net.Forward(x);
            
            auto dy = lossFunc.CalculateLoss(y, t);
            acc += accFunc.CalculateAccuracy(y, t);


            auto dx = net.Backward(dy);

            printBuffer("y", y);
            printBuffer("t", t);
            printBuffer("dy", dy);
            printBuffer("dx", dx);

            printBuffer("af2_dx", net.m_affine2->m_dx);
            printTensorPtr<float>("W0", net.m_affine2->lock_W0_const());
            printTensorPtr<float>("dW0", net.m_affine2->lock_dW0_const());

            optimizer.Update();

            printTensorPtr<float>("W0", net.m_affine2->lock_W0_const());
            printTensorPtr<float>("dW0", net.m_affine2->lock_dW0_const());
        }
        std::cout << acc / data.x_train.size() << std::endl;

        bb::ShuffleDataSet(mt(), data.x_train, data.y_train);
    }

}

