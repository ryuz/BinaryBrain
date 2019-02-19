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

//#include "bb/OptimizerAdam.h"
//#include "bb/LoadMnist.h"
//#include "bb/ShuffleSet.h"


class MnistSimpleMicroMlpNet : public bb::Layer
{
protected:
    using Affine      = bb::MicroMlpAffine<6, 16, float>;
    using AffinePtr   = std::shared_ptr<Affine>;
    using Activate    = bb::ReLU<float>;
    using ActivatePtr = std::shared_ptr<Activate>;


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

    bb::indices_t SetInputShape(bb::indices_t shape)
    {
        shape = m_affine0->SetInputShape(shape);
        shape = m_activate0->SetInputShape(shape);
        shape = m_affine1->SetInputShape(shape);
        shape = m_activate1->SetInputShape(shape);
        shape = m_affine2->SetInputShape(shape);
        return shape;
    }

    bb::FrameBuffer Forward(bb::FrameBuffer x, bool train=true)
    {
        x = m_affine0->Forward(x, train);
        x = m_activate0->Forward(x);
        x = m_affine1->Forward(x, train);
        x = m_activate1->Forward(x);
        x = m_affine2->Forward(x, train);
        return x;
    }

    bb::FrameBuffer Backward(bb::FrameBuffer dy)
    {
        dy = m_affine2->Backward(dy);
        dy = m_activate1->Forward(dy);
        dy = m_affine1->Backward(dy);
        dy = m_activate0->Forward(dy);
        dy = m_affine0->Backward(dy);
        return dy;
    }
};



// MNIST CNN with LUT networks
void MnistSimpleMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
}

