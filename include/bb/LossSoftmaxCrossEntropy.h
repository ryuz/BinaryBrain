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
    using _super = LossFunction;

public:
    static inline std::string LossFunctionName(void) { return "LossSoftmaxCrossEntropy"; }
    static inline std::string ObjectName(void){ return LossFunctionName() + "_" + DataType<T>::Name(); }
    
    std::string GetLossFunctionName(void) const override { return LossFunctionName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
//    FrameBuffer m_dy;
    Tensor_<double>  m_loss_buf;
    Tensor_<double>  m_loss;
    index_t          m_frame_count = 0;

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
        m_loss        = 0;
        m_frame_count = 0;
    }

    double GetLoss(void) const 
    {
        if ( m_frame_count == 0 ) {
            return 0;
        }

        auto loss_ptr = m_loss.LockConst();
        return loss_ptr[0] / (double)m_frame_count;
    }

    FrameBuffer CalculateLoss(FrameBuffer y_buf, FrameBuffer t_buf, index_t batch_size)
    {
        BB_ASSERT(y_buf.GetType() == DataType<T>::type);
        BB_ASSERT(t_buf.GetType() == DataType<T>::type);
        BB_ASSERT(y_buf.GetNodeSize()  == t_buf.GetNodeSize());
        BB_ASSERT(y_buf.GetFrameSize() == t_buf.GetFrameSize());

        FrameBuffer dy_buf(y_buf.GetFrameSize(), y_buf.GetShape(), y_buf.GetType());

        m_loss_buf.Resize(y_buf.GetFrameSize());

        
        index_t frame_size  = t_buf.GetFrameSize();
        index_t node_size   = t_buf.GetNodeSize();
//      index_t stride_size = t_buf.GetFrameStride() / sizeof(T);

        auto shape    = t_buf.GetShape();
        auto ch_size  = shape[0];
        auto pix_size = node_size / ch_size;
        
#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && pix_size == 1
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
                    (double      *)loss_buf_ptr.GetAddr(),
                    (double      *)loss_ptr.GetAddr(),
                    (int          )y_buf.GetNodeSize(),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float)),
                    (int          )batch_size
                );

            m_frame_count += y_buf.GetFrameSize();

            return dy_buf;
        }
#endif

        {
            m_loss_buf = 0;

            auto y_ptr  = y_buf.LockConst<T>();
            auto t_ptr  = t_buf.LockConst<T>();
            auto dy_ptr = dy_buf.Lock<T>(true);
            auto loss_buf_ptr = m_loss_buf.Lock(true);
            auto loss_ptr     = m_loss.Lock();

            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t pix = 0; pix < pix_size; ++pix) {
                    // max
                    auto c = y_ptr.Get(frame, 0);
                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        c = std::max(c, y_ptr.Get(frame, node));
                    }
                    if (!Real_IsValid(c)) {
                        std::cout << "loss c : nan" << std::endl;
                        c = 0;
                    }

                    // sum(exp(y - c))
                    T y_sum = 0;
                    T t_sum = 0;
                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        y_sum += std::exp(y_ptr.Get(frame, node) - c);
                        t_sum += t_ptr.Get(frame, node);
                    }
                    
                    if (y_sum == 0) { y_sum = (T)1.0e-7; }

                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        T y = y_ptr.Get(frame, node);
                        T t = t_ptr.Get(frame, node);
                        T softmax = std::exp(y - c) / y_sum;
                        if ( t > 0) {
                            loss_buf_ptr[frame] += std::log(softmax + (T)1.0e-7);
                            t = (T)1.0;
                        }
                        T dy = (softmax - t) / (T)batch_size;
                        if (!Real_IsValid(dy)) {
                            std::cout << "loss dy : nan" << std::endl;
                            dy = 0;
                        }

                        dy_ptr.Set(frame, node, dy * t_sum);
                    }
                }
            }

            double loss_sum = 0;
            for ( index_t frame = 0; frame < frame_size; ++frame ) {
                loss_sum += loss_buf_ptr[frame];
            }

            loss_ptr[0]   += -loss_sum;
            m_frame_count += frame_size;

            return dy_buf;
        }
    }
};


}

