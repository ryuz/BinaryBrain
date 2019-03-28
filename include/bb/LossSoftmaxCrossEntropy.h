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
    FrameBuffer m_dy;
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

    FrameBuffer CalculateLoss(FrameBuffer y, FrameBuffer t)
    {
        m_dy.Resize(y.GetType(), y.GetFrameSize(), y.GetShape());
        m_loss_buf.Resize(y.GetFrameSize());

#if BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32
                && y.IsDeviceAvailable() && m_dy.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {

            auto y_ptr        = y.LockDeviceMemoryConst();
            auto t_ptr        = t.LockDeviceMemoryConst();
            auto dy_ptr       = m_dy.LockDeviceMemory(true);
            auto loss_buf_ptr = m_loss_buf.LockDeviceMemory(true);
            auto loss_ptr     = m_loss.LockDeviceMemory();

            bbcu_fp32_LossSoftmaxCrossEntropy
        		(
			        (float const *)y_ptr.GetAddr(),
			        (float const *)t_ptr.GetAddr(),
			        (float       *)dy_ptr.GetAddr(),
			        (float       *)loss_buf_ptr.GetAddr(),
			        (float       *)loss_ptr.GetAddr(),
			        (int          )y.GetNodeSize(),
			        (int          )y.GetFrameSize(),
			        (int          )(y.GetFrameStride() / sizeof(float))
                );

            m_frames += y.GetFrameSize();
            return m_dy;
        }
#endif

        {
            index_t frame_size = y.GetFrameSize();
            index_t node_size = y.GetNodeSize();
            index_t stride_size = y.GetFrameStride() / sizeof(T);

            auto y_ptr = y.LockConst<T>();
            auto t_ptr = t.LockConst<T>();
            auto dy_ptr = m_dy.Lock<T>(true);
            auto loss_buf_ptr = m_loss_buf.Lock(true);
            auto loss_ptr = m_loss.Lock();

#pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                // max
                auto c = y_ptr.Get(frame, 0);
                for (index_t node = 1; node < node_size; ++node) {
                    c = std::max(c, y_ptr.Get(frame, node));
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
                    T dy = (softmax - t_ptr.Get(frame, node)) / (T)frame_size;
                    dy_ptr.Set(frame, node, dy);
                }
            }

            T loss_sum = 0;
            for ( index_t frame = 0; frame < frame_size; ++frame ) {
                loss_sum += loss_buf_ptr[frame];
            }

            loss_ptr[0] += -loss_sum;
            m_frames    += frame_size;

            return m_dy;
        }
    }
};


}

