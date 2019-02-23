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
//    AffinePtr   m_affineX;
//    ActivatePtr m_activateX;
    AffinePtr   m_affine0;
    ActivatePtr m_activate0;
    AffinePtr   m_affine1;
    ActivatePtr m_activate1;
    AffinePtr   m_affine2;

public:

    MnistSimpleMicroMlpNet()
    {
//        m_affineX   = Affine::Create({360});
//        m_activateX = Activate::Create();

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
//        shape = m_affineX->SetInputShape(shape);
//        shape = m_activateX->SetInputShape(shape);

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

//        var.PushBack(m_affineX->GetParameters());
//        var.PushBack(m_activateX->GetParameters());

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
//        var.PushBack(m_affineX->GetGradients());
//        var.PushBack(m_activateX->GetGradients());

        var.PushBack(m_affine0->GetGradients());
        var.PushBack(m_activate0->GetGradients());
        var.PushBack(m_affine1->GetGradients());
        var.PushBack(m_activate1->GetGradients());
        var.PushBack(m_affine2->GetGradients());
        return var;
    }

    bb::FrameBuffer Forward(bb::FrameBuffer x, bool train=true)
    {
//        x = m_affineX->Forward(x, train);
//        x = m_activateX->Forward(x, train);

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

//        dy = m_activateX->Backward(dy);
//        dy = m_affineX->Backward(dy);
        return dy;
    }
};


void printBuffer(std::ostream &os, std::string name, bb::FrameBuffer buf)
{
    os << name << " =\n";
    auto ptr = buf.GetConstPtr<float>();
//  for (int f = 0; f < 3; f++) {
    for (int f = 0; f < buf.GetFrameSize(); f++) {
//      for (int i = 0; i < 10; i++) {
        for (int i = 0; i < buf.GetNodeSize(); i++) {
            os << ptr.Get(f, i) << ", ";
        }
        os << "\n";
    }
    os << std::endl;
}

template<typename T>
void printTensorPtr(std::ostream &os, std::string name,  bb::TensorConstPtr_<T, bb::Tensor const, bb::Memory::ConstPtr> ptr)
{
    os << name << " =\n";
    for (int i = 0; i < 10; i++) {
       os << ptr[i] << ", ";
    }
    os << std::endl;
}


// MNIST CNN with LUT networks
void MnistSimpleMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    // load MNIST data
//#ifdef _DEBUG
//	auto data = bb::LoadMnist<>::Load(10, 512, 128);
//#else
  auto data = bb::LoadMnist<>::Load(10);
//#endif

  /*
#ifdef _DEBUG
#ifdef BB_WITH_CUDA
    std::ofstream ofs("log_gpu_d.txt");
#else
    std::ofstream ofs("log_cpu_d.txt");
#endif
#else
#ifdef BB_WITH_CUDA
    std::ofstream ofs("log_gpu.txt");
#else
    std::ofstream ofs("log_cpu.txt");
#endif
#endif
    */

    MnistSimpleMicroMlpNet                          net;
    bb::LossCrossEntropyWithSoftmax<float>          lossFunc;
    bb::AccuracyCategoricalClassification<float>    accFunc(10);

    net.SetInputShape({28, 28, 1});

    bb::FrameBuffer x(BB_TYPE_FP32, mini_batch_size, {28, 28, 1});
    bb::FrameBuffer t(BB_TYPE_FP32, mini_batch_size, 10);

    bb::OptimizerAdam<float> optimizer;
//  bb::OptimizerSgd<float> optimizer(0.05);
    optimizer.SetVariables(net.GetParameters(), net.GetGradients());

    std::mt19937_64 mt(1);

    for ( bb::index_t epoch = 0; epoch < epoch_size; ++epoch ) {
        double acc = 0;
        for (bb::index_t i = 0; i < (bb::index_t)(data.x_train.size() - mini_batch_size); i += mini_batch_size)
        {
            x.SetVector(data.x_train, i);
            t.SetVector(data.y_train, i);

            auto y = net.Forward(x);
            
            auto dy = lossFunc.CalculateLoss(y, t);
            acc += accFunc.CalculateAccuracy(y, t);


           dy = net.Backward(dy);

//           ofs << "\n----------------------\n" << std::endl;
           /*
           printBuffer(ofs, "x", x);
           printBuffer(ofs, "y", y);
           printBuffer(ofs, "t", t);
           printBuffer(ofs, "dy", dy);

           printBuffer(ofs, "1x",  net.m_affine1->m_x);
           printBuffer(ofs, "1y",  net.m_affine1->m_y);
           printBuffer(ofs, "1dx", net.m_affine1->m_dx);
           printBuffer(ofs, "1dy", net.m_affine1->m_dy);

           printBuffer(ofs, "af2_dx", net.m_affine2->m_dx);
           printBuffer(ofs, "af1_dx", net.m_affine1->m_dx);
           printBuffer(ofs, "af0_dy", net.m_activate0->m_dx);
           printTensorPtr<float>(ofs, "W0", net.m_affine0->lock_W0_const());
           printTensorPtr<float>(ofs, "dW0", net.m_affine0->lock_dW0_const());
           printTensorPtr<float>(ofs, "b0", net.m_affine0->lock_b0_const());
           printTensorPtr<float>(ofs, "db0", net.m_affine0->lock_db0_const());
           printTensorPtr<float>(ofs, "W1", net.m_affine0->lock_W1_const());
           printTensorPtr<float>(ofs, "dW1", net.m_affine0->lock_dW1_const());
           printTensorPtr<float>(ofs, "b1", net.m_affine0->lock_b1_const());
           printTensorPtr<float>(ofs, "db1", net.m_affine0->lock_db1_const());
           ofs << "in = " << net.m_affine0->m_input_index;
           */
//         printBuffer(ofs, "dy", net.m_affine0->m_dy);
//         printBuffer(ofs, "2_dx", net.m_affine2->m_dx);
//         printBuffer(ofs, "1_dx", net.m_affine1->m_dx);
//         printBuffer(ofs, "0_dx", net.m_affine0->m_dx);
//         printBuffer(ofs, "X_dx", net.m_affineX->m_dx);

           optimizer.Update();
//           printTensorPtr<float>(ofs, "W0", net.m_affine0->lock_W0_const());
//           printTensorPtr<float>(ofs, "dW0", net.m_affine0->lock_dW0_const());
//           ofs << "\n" << std::endl;
        }
        std::cout << acc / data.x_train.size() << std::endl;

        bb::ShuffleDataSet(mt(), data.x_train, data.y_train);
    }

}

