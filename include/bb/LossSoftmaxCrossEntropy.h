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
    Tensor_<double> m_loss_buf;
    double          m_loss;
    double          m_frame_count = 0;

protected:
    LossSoftmaxCrossEntropy() {
//      m_loss.Resize(1);
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
        
        return m_loss / m_frame_count;
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

        auto shape    = t_buf.GetShape();
        auto ch_size  = shape[0];
        auto pix_size = node_size / ch_size;
        
#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && pix_size == 1
                && y_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {

            T   t_sum = (T)t_buf.Sum();
            
            {
                auto y_ptr        = y_buf.LockDeviceMemoryConst();
                auto t_ptr        = t_buf.LockDeviceMemoryConst();
                auto dy_ptr       = dy_buf.LockDeviceMemory(true);
                auto loss_buf_ptr = m_loss_buf.LockDeviceMemory(true);

                bbcu_LossSoftmaxCrossEntropy<T>
                    (
                        (T const *)y_ptr.GetAddr(),
                        (T const *)t_ptr.GetAddr(),
                        (T       *)dy_ptr.GetAddr(),
                        (double  *)loss_buf_ptr.GetAddr(),
                        (T        )t_sum,
                        (int      )pix_size,
                        (int      )ch_size,
                        (int      )y_buf.GetFrameSize(),
                        (int      )(y_buf.GetFrameStride() / sizeof(float))
                    );
            }

            m_loss        += -m_loss_buf.Sum();
            m_frame_count += t_sum;

            return dy_buf;
        }
#endif

        
#if 0
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
            T   eps = (T)1.0e-7;

            m_loss_buf = 0;

            auto y_ptr  = y_buf.LockConst<T>();
            auto t_ptr  = t_buf.LockConst<T>();
            auto dy_ptr = dy_buf.Lock<T>(true);
            auto loss_buf_ptr = m_loss_buf.Lock(true);

            T t_sum = 0;
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t node = 0; node < node_size; ++node) {
                    t_sum += t_ptr.Get(frame, node);
                }
            }


            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t pix = 0; pix < pix_size; ++pix) {
                    // max
                    T   c = std::numeric_limits<T>::lowest();
                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        c = std::max(c, y_ptr.Get(frame, node));
                    }
//                  if (!Real_IsValid(c)) {
////                    std::cout << "loss c : nan" << std::endl;
//                      c = 0;
//                  }

                    // sum(exp(y - c))
                    T y_sum = 0;
                    T t_max = 0;
                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        y_sum += std::exp(y_ptr.Get(frame, node) - c);
                        t_max += t_ptr.Get(frame, node);    // ワンホットなので足していけばそのチャネルのWeightが得られる
                    }
                    
                    // 0以下での除算回避
                    if (y_sum <= eps) { y_sum = eps; }

                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        T y = y_ptr.Get(frame, node);
                        T t = t_ptr.Get(frame, node);
                        T softmax = std::exp(y - c) / y_sum;
                        if ( t > 0) {
                            loss_buf_ptr[frame] += std::log(softmax + eps)*t_max;
//                          t = (T)1.0;
                        }

                        T dy = (t_max * softmax - t) / (T)t_sum;
//                        if (!Real_IsValid(dy)) {
////                          std::cout << "loss dy : nan" << std::endl;
//                            dy = 0;
//                        }

                        dy_ptr.Set(frame, node, dy);
                    }
                }
            }

            double loss_sum = 0;
            for ( index_t frame = 0; frame < frame_size; ++frame ) {
                loss_sum += loss_buf_ptr[frame];
            }

            m_loss        += -loss_sum;
            m_frame_count += t_sum;

            return dy_buf;
        }
    }
};


}

