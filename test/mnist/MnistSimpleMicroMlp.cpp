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


/*
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
*/


void DumpLayerForward(std::ostream &os, MnistSimpleMicroMlpNet const &net0, MnistSimpleMicroMlpNet const &net1)
{
    os << "-------- forward -------"  << std::endl;
    os << "l0_x   = " << (net0.m_affine0->m_x  - net1.m_affine0->m_x  ).Norm() << std::endl;
    os << "l0_y   = " << (net0.m_affine0->m_y  - net1.m_affine0->m_y  ).Norm() << std::endl;
    os << "l0_W0  = " << (net0.m_affine0->W0() - net1.m_affine0->W0() ).Norm() << std::endl;
    os << "l0_b0  = " << (net0.m_affine0->b0() - net1.m_affine0->b0() ).Norm() << std::endl;
    os << "l0_W1  = " << (net0.m_affine0->W1() - net1.m_affine0->W1() ).Norm() << std::endl;
    os << "l0_b0  = " << (net0.m_affine0->b0() - net1.m_affine0->b0() ).Norm() << std::endl;
    os << "l1_x   = " << (net0.m_affine1->m_x  - net1.m_affine1->m_x  ).Norm() << std::endl;
    os << "l1_y   = " << (net0.m_affine1->m_y  - net1.m_affine1->m_y  ).Norm() << std::endl;
    os << "l1_W0  = " << (net0.m_affine1->W0() - net1.m_affine1->W0() ).Norm() << std::endl;
    os << "l1_b0  = " << (net0.m_affine1->b0() - net1.m_affine1->b0() ).Norm() << std::endl;
    os << "l1_W1  = " << (net0.m_affine1->W1() - net1.m_affine1->W1() ).Norm() << std::endl;
    os << "l1_b0  = " << (net0.m_affine1->b0() - net1.m_affine1->b0() ).Norm() << std::endl;    
    os << "l2_x   = " << (net0.m_affine2->m_x  - net1.m_affine2->m_x  ).Norm() << std::endl;
    os << "l2_y   = " << (net0.m_affine2->m_y  - net1.m_affine2->m_y  ).Norm() << std::endl;
    os << "l2_W0  = " << (net0.m_affine2->W0() - net1.m_affine2->W0() ).Norm() << std::endl;
    os << "l2_b0  = " << (net0.m_affine2->b0() - net1.m_affine2->b0() ).Norm() << std::endl;
    os << "l2_W1  = " << (net0.m_affine2->W1() - net1.m_affine2->W1() ).Norm() << std::endl;
    os << "l2_b0  = " << (net0.m_affine2->b0() - net1.m_affine2->b0() ).Norm() << std::endl;    
    os << "l3_x   = " << (net0.m_affine3->m_x  - net1.m_affine3->m_x  ).Norm() << std::endl;
    os << "l3_y   = " << (net0.m_affine3->m_y  - net1.m_affine3->m_y  ).Norm() << std::endl;
    os << "l3_W0  = " << (net0.m_affine3->W0() - net1.m_affine3->W0() ).Norm() << std::endl;
    os << "l3_b0  = " << (net0.m_affine3->b0() - net1.m_affine3->b0() ).Norm() << std::endl;
    os << "l3_W1  = " << (net0.m_affine3->W1() - net1.m_affine3->W1() ).Norm() << std::endl;
    os << "l3_b0  = " << (net0.m_affine3->b0() - net1.m_affine3->b0() ).Norm() << std::endl;    
}

void DumpLayerBackward(std::ostream &os, MnistSimpleMicroMlpNet const &net0, MnistSimpleMicroMlpNet const &net1)
{
    os << "-------- backward -------"  << std::endl;
    os << "l3_dy  = " << (net0.m_affine3->m_dy  - net1.m_affine3->m_dy  ).Norm() << std::endl;
    os << "l3_dx  = " << (net0.m_affine3->m_dx  - net1.m_affine3->m_dx  ).Norm() << std::endl;
    os << "l3_dW0 = " << (net0.m_affine3->dW0() - net1.m_affine3->dW0() ).Norm() << ", " << net0.m_affine3->dW0().Norm() << ", " << net1.m_affine3->dW0().Norm() << std::endl;
    os << "l3_db0 = " << (net0.m_affine3->db0() - net1.m_affine3->db0() ).Norm() << ", " << net0.m_affine3->db0().Norm() << ", " << net1.m_affine3->db0().Norm() << std::endl;
    os << "l3_dW1 = " << (net0.m_affine3->dW1() - net1.m_affine3->dW1() ).Norm() << ", " << net0.m_affine3->dW1().Norm() << ", " << net1.m_affine3->dW1().Norm() << std::endl;
    os << "l3_db0 = " << (net0.m_affine3->db0() - net1.m_affine3->db0() ).Norm() << ", " << net0.m_affine3->db0().Norm() << ", " << net1.m_affine3->db0().Norm() << std::endl;
    os << "l2_dy  = " << (net0.m_affine2->m_dy  - net1.m_affine2->m_dy  ).Norm() << std::endl;
    os << "l2_dx  = " << (net0.m_affine2->m_dx  - net1.m_affine2->m_dx  ).Norm() << std::endl;
    os << "l2_dW0 = " << (net0.m_affine2->dW0() - net1.m_affine2->dW0() ).Norm() << std::endl;
    os << "l2_db0 = " << (net0.m_affine2->db0() - net1.m_affine2->db0() ).Norm() << std::endl;
    os << "l2_dW1 = " << (net0.m_affine2->dW1() - net1.m_affine2->dW1() ).Norm() << std::endl;
    os << "l2_db0 = " << (net0.m_affine2->db0() - net1.m_affine2->db0() ).Norm() << std::endl;    
    os << "l1_dy  = " << (net0.m_affine1->m_dy  - net1.m_affine1->m_dy  ).Norm() << std::endl;
    os << "l1_dx  = " << (net0.m_affine1->m_dx  - net1.m_affine1->m_dx  ).Norm() << std::endl;
    os << "l1_dW0 = " << (net0.m_affine1->dW0() - net1.m_affine1->dW0() ).Norm() << std::endl;
    os << "l1_db0 = " << (net0.m_affine1->db0() - net1.m_affine1->db0() ).Norm() << std::endl;
    os << "l1_dW1 = " << (net0.m_affine1->dW1() - net1.m_affine1->dW1() ).Norm() << std::endl;
    os << "l1_db0 = " << (net0.m_affine1->db0() - net1.m_affine1->db0() ).Norm() << std::endl;    
    os << "l0_dy  = " << (net0.m_affine0->m_dy  - net1.m_affine0->m_dy  ).Norm() << std::endl;
    os << "l0_dx  = " << (net0.m_affine0->m_dx  - net1.m_affine0->m_dx  ).Norm() << std::endl;
    os << "l0_dW0 = " << (net0.m_affine0->dW0() - net1.m_affine0->dW0() ).Norm() << std::endl;
    os << "l0_db0 = " << (net0.m_affine0->db0() - net1.m_affine0->db0() ).Norm() << std::endl;
    os << "l0_dW1 = " << (net0.m_affine0->dW1() - net1.m_affine0->dW1() ).Norm() << std::endl;
    os << "l0_db0 = " << (net0.m_affine0->db0() - net1.m_affine0->db0() ).Norm() << std::endl;    
}

void DumpLayerUpdate(std::ostream &os, MnistSimpleMicroMlpNet const &net0, MnistSimpleMicroMlpNet const &net1)
{
    os << "-------- update -------"  << std::endl;
    os << "l0_W0 = " << (net0.m_affine0->W0() - net1.m_affine0->W0() ).Norm() << std::endl;
    os << "l0_b0 = " << (net0.m_affine0->b0() - net1.m_affine0->b0() ).Norm() << std::endl;
    os << "l0_W1 = " << (net0.m_affine0->W1() - net1.m_affine0->W1() ).Norm() << std::endl;
    os << "l0_b0 = " << (net0.m_affine0->b0() - net1.m_affine0->b0() ).Norm() << std::endl;
    os << "l1_W0 = " << (net0.m_affine1->W0() - net1.m_affine1->W0() ).Norm() << std::endl;
    os << "l1_b0 = " << (net0.m_affine1->b0() - net1.m_affine1->b0() ).Norm() << std::endl;
    os << "l1_W1 = " << (net0.m_affine1->W1() - net1.m_affine1->W1() ).Norm() << std::endl;
    os << "l1_b0 = " << (net0.m_affine1->b0() - net1.m_affine1->b0() ).Norm() << std::endl;    
    os << "l2_W0 = " << (net0.m_affine2->W0() - net1.m_affine2->W0() ).Norm() << std::endl;
    os << "l2_b0 = " << (net0.m_affine2->b0() - net1.m_affine2->b0() ).Norm() << std::endl;
    os << "l2_W1 = " << (net0.m_affine2->W1() - net1.m_affine2->W1() ).Norm() << std::endl;
    os << "l2_b0 = " << (net0.m_affine2->b0() - net1.m_affine2->b0() ).Norm() << std::endl;    
    os << "l3_W0 = " << (net0.m_affine3->W0() - net1.m_affine3->W0() ).Norm() << std::endl;
    os << "l3_b0 = " << (net0.m_affine3->b0() - net1.m_affine3->b0() ).Norm() << std::endl;
    os << "l3_W1 = " << (net0.m_affine3->W1() - net1.m_affine3->W1() ).Norm() << std::endl;
    os << "l3_b0 = " << (net0.m_affine3->b0() - net1.m_affine3->b0() ).Norm() << std::endl;    
}


void DumpAffineLayer(std::ostream &os, std::string name, bb::MicroMlpAffine<6, 16, float> const &affine)
{
    static int num = 0;
    os << num << ":" << name << " W0 = " << *affine.m_W0 << std::endl;
    os << num << ":" << name << " b0 = " << *affine.m_b0 << std::endl;
    os << num << ":" << name << " W1 = " << *affine.m_W1 << std::endl;
    os << num << ":" << name << " b0 = " << *affine.m_b0 << std::endl;
    os << num << ":" << name << " dW0 = " << *affine.m_dW0 << std::endl;
    os << num << ":" << name << " db0 = " << *affine.m_db0 << std::endl;
    os << num << ":" << name << " dW1 = " << *affine.m_dW1 << std::endl;
    os << num << ":" << name << " db0 = " << *affine.m_db0 << std::endl;

//   os << num << ":" << name << " x  = " << affine.m_x << std::endl;
//   os << num << ":" << name << " y  = " << affine.m_y << std::endl;
//   os << num << ":" << name << " dy = " << affine.m_dy << std::endl;
//   os << num << ":" << name << " dx = " << affine.m_dx << std::endl;

    num++;
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

    MnistSimpleMicroMlpNet                          cpu_net;
    bb::LossCrossEntropyWithSoftmax<float>          cpu_lossFunc;
    bb::AccuracyCategoricalClassification<float>    cpu_accFunc(10);

    MnistSimpleMicroMlpNet                          gpu_net;
    bb::LossCrossEntropyWithSoftmax<float>          gpu_lossFunc;
    bb::AccuracyCategoricalClassification<float>    gpu_accFunc(10);

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

#ifdef BB_WITH_CUDA
//  std::ofstream ofs("log_gpu.txt");
//  std::ofstream ofs("log_relu.txt");  cpu_net.SendCommand("host_only true", "MicroMlpAffine");
//  std::ofstream ofs("log_all.txt");   net.SendCommand("host_only true");
#else
//  std::ofstream ofs("log_cpu.txt");
#endif


    cpu_net.SendCommand("host_only true", "MicroMlpAffine");

//    cpu_net.SendCommand("binary true");
//    gpu_net.SendCommand("binary true");

    std::ofstream ofs("dump_norm.txt");

    int dbg = 0;

    for ( bb::index_t epoch = 0; epoch < epoch_size; ++epoch ) {
        cpu_accFunc.Clear();
        gpu_accFunc.Clear();
        for (bb::index_t i = 0; i < (bb::index_t)(data.x_train.size() - mini_batch_size); i += mini_batch_size)
        {
            cpu_x.SetVector(data.x_train, i);
            gpu_x.SetVector(data.x_train, i);
            cpu_t.SetVector(data.y_train, i);
            gpu_t.SetVector(data.y_train, i);

            auto cpu_y = cpu_net.Forward(cpu_x);
            auto gpu_y = gpu_net.Forward(gpu_x);
            
            DumpLayerForward(ofs, cpu_net, gpu_net);

            auto cpu_dy = cpu_lossFunc.CalculateLoss(cpu_y, cpu_t);
            auto gpu_dy = gpu_lossFunc.CalculateLoss(gpu_y, gpu_t);

            cpu_accFunc.CalculateAccuracy(cpu_y, cpu_t);
            gpu_accFunc.CalculateAccuracy(gpu_y, gpu_t);

            cpu_dy = cpu_net.Backward(cpu_dy);
            gpu_dy = gpu_net.Backward(gpu_dy);

            DumpLayerBackward(ofs, cpu_net, gpu_net);

            cpu_optimizer->Update();
            gpu_optimizer->Update();

            DumpLayerUpdate(ofs, cpu_net, gpu_net);
        }
        std::cout << "cpu : " << cpu_accFunc.GetAccuracy() << std::endl;
        std::cout << "gpu : " << gpu_accFunc.GetAccuracy() << std::endl;

        bb::ShuffleDataSet(mt(), data.x_train, data.y_train);
    }

}

