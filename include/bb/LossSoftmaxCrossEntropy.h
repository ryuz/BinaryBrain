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
class LossSoftmaxCrossEntropy : public LossFunction
{
protected:
//    FrameBuffer m_dy;
    Tensor_<T>  m_loss_buf;
    Tensor_<T>  m_loss;
    index_t     m_frames = 0;

protected:
    LossSoftmaxCrossEntropy() {
        m_loss.Resize(1);
        Clear();
    }

public:
    ~LossSoftmaxCrossEntropy() {}
    
    static std::shared_ptr<LossSoftmaxCrossEntropy> Create(void)
    {
        auto self = std::shared_ptr<LossSoftmaxCrossEntropy>(new LossSoftmaxCrossEntropy);
        return self;
    }

    void Clear(void)
    {
        m_loss = 0;
        m_frames = 0;
    }

    double GetLoss(void) const 
    {
        auto loss_ptr = m_loss.LockConst();
        return (double)loss_ptr[0] / (double)m_frames;
    }

    FrameBuffer CalculateLoss(FrameBuffer y_buf, FrameBuffer t_buf, index_t batch_size)
    {
        FrameBuffer dy_buf(y_buf.GetFrameSize(), y_buf.GetShape(), y_buf.GetType());

        m_loss_buf.Resize(y_buf.GetFrameSize());

#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32
                && y_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {

            auto y_ptr        = y_buf.LockDeviceMemoryConst();
            auto t_ptr        = t_buf.LockDeviceMemoryConst();
            auto dy_ptr       = dy_buf.LockDeviceMemory(true);
            auto loss_buf_ptr = m_loss_buf.LockDeviceMemory(true);
            auto loss_ptr     = m_loss.LockDeviceMemory();

            bbcu_fp32_LossSoftmaxCrossEntropy
                (
                    (float const *)y_ptr.GetAddr(),
                    (float const *)t_ptr.GetAddr(),
                    (float       *)dy_ptr.GetAddr(),
                    (float       *)loss_buf_ptr.GetAddr(),
                    (float       *)loss_ptr.GetAddr(),
                    (int          )y_buf.GetNodeSize(),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float)),
                    (int          )batch_size
                );

            m_frames += y_buf.GetFrameSize();

            return dy_buf;
        }
#endif

        {
            index_t frame_size  = y_buf.GetFrameSize();
            index_t node_size   = y_buf.GetNodeSize();
//          index_t stride_size = y_buf.GetFrameStride() / sizeof(T);

            auto y_ptr  = y_buf.LockConst<T>();
            auto t_ptr  = t_buf.LockConst<T>();
            auto dy_ptr = dy_buf.Lock<T>(true);
            auto loss_buf_ptr = m_loss_buf.Lock(true);
            auto loss_ptr     = m_loss.Lock();

            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                // max
                auto c = y_ptr.Get(frame, 0);
                for (index_t node = 1; node < node_size; ++node) {
                    c = std::max(c, y_ptr.Get(frame, node));
                }
                if (!Real_IsValid(c)) {
                    std::cout << "loss c : nan" << std::endl;
                }

                // sum(exp(y - c))
                T sum = 0;
                for (index_t node = 0; node < node_size; ++node) {
                    sum += std::exp(y_ptr.Get(frame, node) - c);
                }

                for (index_t node = 0; node < node_size; ++node) {
                    T softmax = std::exp(y_ptr.Get(frame, node) - c) / sum;
                    if (t_ptr.Get(frame, node) > 0) {
                        loss_buf_ptr[frame] = std::log(softmax + (T)1.0e-7);
                    }
                    T dy = (softmax - t_ptr.Get(frame, node)) / (T)batch_size;
                    if (!Real_IsValid(dy)) {
                        std::cout << "loss dy : nan" << std::endl;
                    }

                    dy_ptr.Set(frame, node, dy);
                }
            }

            T loss_sum = 0;
            for ( index_t frame = 0; frame < frame_size; ++frame ) {
                loss_sum += loss_buf_ptr[frame];
            }

            loss_ptr[0] += -loss_sum;
            m_frames    += frame_size;

            return dy_buf;
        }
    }
};


}

