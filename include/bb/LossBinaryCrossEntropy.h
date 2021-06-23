// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                 Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                 https://github.com/ryuz
//                                 ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>
#include <valarray>

#include "bb/LossFunction.h"


namespace bb {

template <typename T = float>
class LossBinaryCrossEntropy : public LossFunction
{
    using _super = LossFunction;

public:
    static inline std::string LossFunctionName(void) { return "LossBinaryCrossEntropy"; }
    static inline std::string ObjectName(void){ return LossFunctionName() + "_" + DataType<T>::Name(); }
    
    std::string GetLossFunctionName(void) const override { return LossFunctionName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    double           m_loss;
    index_t          m_frame_count = 0;

protected:
    LossBinaryCrossEntropy() {
        Clear();
    }

public:
    ~LossBinaryCrossEntropy() {}
    
    static std::shared_ptr<LossBinaryCrossEntropy> Create(void)
    {
        auto self = std::shared_ptr<LossBinaryCrossEntropy>(new LossBinaryCrossEntropy);
        return self;
    }

    void Clear(void)
    {
        m_loss        = 0;
        m_frame_count = 0;
    }

    double GetLoss(void) const 
    {
        if ( m_frame_count == 0 ) {
            return 0;
        }
        return m_loss / (double)m_frame_count;
    }

    FrameBuffer CalculateLoss(FrameBuffer y_buf, FrameBuffer t_buf, index_t batch_size)
    {
        BB_ASSERT(y_buf.GetType() == DataType<T>::type);
        BB_ASSERT(t_buf.GetType() == DataType<T>::type);
        BB_ASSERT(y_buf.GetNodeSize()  == t_buf.GetNodeSize());
        BB_ASSERT(y_buf.GetFrameSize() == t_buf.GetFrameSize());

        FrameBuffer dy_buf(y_buf.GetFrameSize(), y_buf.GetShape(), y_buf.GetType());
        
        index_t frame_size  = t_buf.GetFrameSize();
        index_t node_size   = t_buf.GetNodeSize();
        

        {
            auto y_ptr  = y_buf.LockConst<T>();
            auto t_ptr  = t_buf.LockConst<T>();
            auto dy_ptr = dy_buf.Lock<T>(true);

            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t node = 0; node < node_size; ++node) {
                    auto y = y_ptr.Get(frame, node);
                    auto t = t_ptr.Get(frame, node);
                    T   dy = 0;
                    if ( t > 0 ) {
                        m_loss -= (double)std::log(y);
                        dy = (T)-1 / y;
                    }
                    else {
                        y = (T)1 - y;
                        m_loss -= (double)std::log(y);
                        dy = (T)1 / y;
                    }
                    dy_ptr.Set(frame, node, dy / batch_size);
                }
            }

            m_frame_count += frame_size;

            return dy_buf;
        }
    }
};


}

