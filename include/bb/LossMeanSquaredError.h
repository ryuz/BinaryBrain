// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>
#include <cmath>


#include "bb/LossFunction.h"


namespace bb {

template <typename T = float>
class LossMeanSquaredError : public LossFunction
{
    using _super = LossFunction;

public:
    static inline std::string LossFunctionName(void) { return "LossMeanSquaredError"; }
    static inline std::string ObjectName(void){ return LossFunctionName() + "_" + DataType<T>::Name(); }
    
    std::string GetLossFunctionName(void) const override { return LossFunctionName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }


protected:
//  FrameBuffer     m_dy_buf;
//  double          m_loss = 0;
    double          m_frames = 0;
    Tensor_<double> m_loss_buf;

protected:
    LossMeanSquaredError() {
        m_loss_buf.Resize(1024);
    }

public:
    ~LossMeanSquaredError() {}
    
    static std::shared_ptr<LossMeanSquaredError> Create(void)
    {
        auto self = std::shared_ptr<LossMeanSquaredError>(new LossMeanSquaredError);
        return self;
    }

    void Clear(void)
    {
//      m_loss = 0;
        m_loss_buf = 0;
        m_frames = 0;
    }

    double GetLoss(void) const 
    {
        double loss = 0;
        auto loss_buf_ptr = m_loss_buf.LockConst();
        for (int i = 0; i < 1024; ++i) {
            loss += loss_buf_ptr[i];
        }
        return loss / m_frames;
    }

    FrameBuffer CalculateLoss(FrameBuffer y_buf, FrameBuffer t_buf, index_t batch_size)
    {
        BB_ASSERT(y_buf.GetFrameSize() == t_buf.GetFrameSize());
        BB_ASSERT(y_buf.GetNodeSize()  == t_buf.GetNodeSize());

        index_t frame_size  = y_buf.GetFrameSize();
        index_t node_size   = y_buf.GetNodeSize();

        FrameBuffer dy_buf(y_buf.GetFrameSize(), y_buf.GetShape(), DataType<T>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32
                    && y_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {

            // まだバグってる
            auto y_ptr        = y_buf.LockDeviceMemoryConst();
            auto t_ptr        = t_buf.LockDeviceMemoryConst();
            auto dy_ptr       = dy_buf.LockDeviceMemory(true);
            auto loss_buf_ptr = m_loss_buf.LockDeviceMemory();
            
            bbcu_fp32_LossMeanSquaredError
                (
                    (float const *)y_ptr.GetAddr(),
                    (float const *)t_ptr.GetAddr(),
                    (float       *)dy_ptr.GetAddr(),
                    (double      *)loss_buf_ptr.GetAddr(),
                    (int          )1024,
                    (int          )y_buf.GetNodeSize(),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float)),
                    (int          )batch_size
                );
            
            m_frames += frame_size;
            return dy_buf;
        }
#endif
        
        {
            auto y_ptr = y_buf.LockConst<T>();
            auto t_ptr = t_buf.LockConst<T>();
            auto dy_ptr = dy_buf.Lock<T>();
            auto loss_buf_ptr = m_loss_buf.Lock();

            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t node = 0; node < node_size; ++node) {
                    auto signal = y_ptr.Get(frame, node);
                    auto target = t_ptr.Get(frame, node);
                    auto grad = signal - target;
                    auto error = grad * grad;

                    dy_ptr.Set(frame, node, grad / (T)batch_size);
                    if ( !std::isnan(error) ) {
                        loss_buf_ptr[0] += error / (double)node_size;
                    }
                }
            }

            m_frames += frame_size;

            return dy_buf;
        }
    }
};


}

