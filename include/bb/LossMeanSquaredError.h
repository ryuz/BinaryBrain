// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>


#include "bb/LossFunction.h"


namespace bb {

template <typename T = float>
class LossMeanSquaredError : public LossFunction
{
protected:
    FrameBuffer m_dy;
    double      m_loss;
    double      m_frames;

protected:
    LossMeanSquaredError() {}

public:
    ~LossMeanSquaredError() {}
    
    static std::shared_ptr<LossMeanSquaredError> Create(void)
    {
        auto self = std::shared_ptr<LossMeanSquaredError>(new LossMeanSquaredError);
        return self;
    }

    void Clear(void)
    {
        m_loss = 0;
        m_frames = 0;
    }

    double GetLoss(void) const 
    {
        return m_loss / m_frames;
    }

    FrameBuffer CalculateLoss(FrameBuffer y, FrameBuffer t, index_t batch_size)
    {
        index_t frame_size  = y.GetFrameSize();
        index_t node_size   = y.GetNodeSize();

        m_dy.Resize(y.GetFrameSize(), y.GetShape(), DataType<T>::type);

        auto y_ptr = y.LockConst<T>();
        auto t_ptr = t.LockConst<T>();
        auto dy_ptr = m_dy.Lock<T>();

        for (index_t frame = 0; frame < frame_size; ++frame) {
            for (index_t node = 0; node < node_size; ++node) {
                auto signal = y_ptr.Get(frame, node);
                auto target = t_ptr.Get(frame, node);
                auto grad = signal - target;
                auto error = grad * grad;

                dy_ptr.Set(frame, node, grad / (T)batch_size);
                m_loss += error / (double)node_size;
            }
        }
        m_frames += frame_size;

        return m_dy;
    }
};


}

