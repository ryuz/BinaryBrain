// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>

#include "bb/MetricsFunction.h"


namespace bb {


template <typename T = float>
class MetricsCategoricalAccuracy : public MetricsFunction
{
protected:
    index_t         m_frames = 0;
    Tensor_<int>    m_accuracy;

protected:
    MetricsCategoricalAccuracy() : m_accuracy(1)
    {
        Clear();
    }

public:
    ~MetricsCategoricalAccuracy() {}

    static std::shared_ptr<MetricsCategoricalAccuracy> Create()
    {
        auto self = std::shared_ptr<MetricsCategoricalAccuracy>(new MetricsCategoricalAccuracy);
        return self;
    }
    
    void Clear(void)
    {
        m_accuracy = 0;
        m_frames   = 0;
    }


    double GetMetrics(void) const
    {
        auto ptr = m_accuracy.LockConst();
        auto acc = ptr[0];
        return (double)acc / (double)m_frames;
    }

    void CalculateMetrics(FrameBuffer y, FrameBuffer t)
    {
        BB_ASSERT(y.GetType() == DataType<T>::type);
        BB_ASSERT(t.GetType() == DataType<T>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && y.IsDeviceAvailable() && t.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto y_ptr   = y.LockDeviceMemoryConst();
            auto t_ptr   = t.LockDeviceMemoryConst();
            auto acc_ptr = m_accuracy.LockDeviceMemory();

            bbcu_fp32_AccuracyCategoricalClassification
                (
                    (float const *)y_ptr.GetAddr(),
                    (float const *)t_ptr.GetAddr(),
                    (int         *)acc_ptr.GetAddr(),
                    (int          )y.GetNodeSize(),
                    (int          )y.GetFrameSize(),
                    (int          )(y.GetFrameStride() / sizeof(float))
                );

            m_frames += y.GetFrameSize();

            return;
        }
#endif

        {
            index_t frame_size = y.GetFrameSize();
            index_t node_size = y.GetNodeSize();
            auto acc_ptr = m_accuracy.Lock();

            m_frames += frame_size;

            auto y_ptr  = y.LockConst<T>();
            auto t_ptr  = t.LockConst<T>();
 
            for (index_t frame = 0; frame < frame_size; ++frame) {
                index_t max_node   = 0;
                T       max_signal = y_ptr.Get(frame, 0);
                for (index_t node = 1; node < node_size; ++node) {
                    T   sig = y_ptr.Get(frame, node);
                    if (sig > max_signal) {
                        max_node   = node;
                        max_signal = sig;
                    }
                }
                if ( t_ptr.Get(frame, max_node) > 0) {
                    acc_ptr[0] += 1;
                }
            }
        }
    }
};


}

