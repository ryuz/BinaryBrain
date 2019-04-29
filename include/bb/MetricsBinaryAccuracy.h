// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <iostream>
#include <vector>

#include "bb/MetricsFunction.h"


namespace bb {


template <typename T = float>
class MetricsBinaryAccuracy : public MetricsFunction
{
protected:
    index_t     m_frames = 0;
    double      m_accuracy = 0;

protected:
    MetricsBinaryAccuracy() {}

public:
    ~MetricsBinaryAccuracy() {}

    static std::shared_ptr<MetricsBinaryAccuracy> Create()
    {
        auto self = std::shared_ptr<MetricsBinaryAccuracy>(new MetricsBinaryAccuracy);
        return self;
    }
    
    void Clear(void)
    {
        m_accuracy = 0;
        m_frames   = 0;
    }
    
    double GetMetrics(void) const
    {
        return m_accuracy/ (double)m_frames;
    }
    
    void CalculateMetrics(FrameBuffer y, FrameBuffer t)
    {
        BB_ASSERT(y.GetType() == DataType<T>::type);
        BB_ASSERT(t.GetType() == DataType<T>::type);

        index_t frame_size = y.GetFrameSize();
        index_t node_size  = y.GetNodeSize();

        auto y_ptr = y.LockConst<T>();
        auto t_ptr = t.LockConst<T>();

        for (index_t frame = 0; frame < frame_size; ++frame) {
            for (index_t node = 0; node < node_size; ++node) {
                T   sig_val = y_ptr.Get(frame, node);
                T   exp_val = t_ptr.Get(frame, node);

                bool    sig = (y_ptr.Get(frame, node) >= (T)0.5);
                bool    exp = (t_ptr.Get(frame, node) >= (T)0.5);

//              std::cout << " " << exp << " " << sig << std::endl;

                if (sig == exp) {
                    m_accuracy += 1.0;
                }
            }
        }

        m_frames += frame_size;
    }
};


}

