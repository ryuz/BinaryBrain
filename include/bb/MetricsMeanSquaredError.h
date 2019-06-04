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
class MetricsMeanSquaredError : public MetricsFunction
{
protected:
    double  m_accuracy = 0;
    index_t m_frames;


protected:
    MetricsMeanSquaredError()   {}

public:
    ~MetricsMeanSquaredError() {}

    std::string GetMetricsString(void) { return "MSE"; }

    static std::shared_ptr<MetricsMeanSquaredError> Create(void)
    {
        auto self = std::shared_ptr<MetricsMeanSquaredError>(new MetricsMeanSquaredError);
        return self;
    }
     
    void Clear(void)
    {
        m_accuracy = 0;
        m_frames   = 0;
    }


    double GetMetrics(void) const
    {
        return m_accuracy / (double)m_frames;
    }

    void CalculateMetrics(FrameBuffer y, FrameBuffer t)
    {
        BB_ASSERT(y.GetType() == DataType<T>::type);
        BB_ASSERT(t.GetType() == DataType<T>::type);

        index_t frame_size = y.GetFrameSize();
        index_t node_size = y.GetNodeSize();

        auto y_ptr = y.LockConst<T>();
        auto t_ptr = t.LockConst<T>();

        for (index_t frame = 0; frame < frame_size; ++frame) {
            for (index_t node = 0; node < node_size; ++node) {
                auto signal = y_ptr.Get(frame, node);
                auto target = t_ptr.Get(frame, node);
                auto grad = target - signal;
                auto error = grad * grad;
                m_accuracy += error / (double)node_size;
            }
            m_frames++;
        }
    }

};


}

