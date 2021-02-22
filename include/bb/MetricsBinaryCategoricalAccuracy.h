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
class MetricsBinaryCategoricalAccuracy : public MetricsFunction
{
    using _super = MetricsFunction;

public:
    static inline std::string MetricsFunctionName(void) { return "MetricsBinaryCategoricalAccuracy"; }
    static inline std::string ObjectName(void){ return MetricsFunctionName() + "_" + DataType<T>::Name(); }
    
    std::string GetMetricsFunctionName(void) const override { return MetricsFunctionName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }


protected:
    int     m_accuracy;
    int     m_category_count = 0;

protected:
    MetricsBinaryCategoricalAccuracy() : m_accuracy(1)
    {
        Clear();
    }

public:
    ~MetricsBinaryCategoricalAccuracy() {}

    static std::shared_ptr<MetricsBinaryCategoricalAccuracy> Create()
    {
        auto self = std::shared_ptr<MetricsBinaryCategoricalAccuracy>(new MetricsBinaryCategoricalAccuracy);
        return self;
    }
    
    void Clear(void)
    {
        m_accuracy       = 0;
        m_category_count = 0;
    }


    double GetMetrics(void) const
    {
        if ( m_category_count == 0 ) {
            return 0;
        }
        return (double)m_accuracy / m_category_count;
    }

    void CalculateMetrics(FrameBuffer y_buf, FrameBuffer t_buf)
    {
        BB_ASSERT(y_buf.GetType() == DataType<T>::type);
        BB_ASSERT(t_buf.GetType() == DataType<T>::type);
        BB_ASSERT(y_buf.GetNodeSize()  == t_buf.GetNodeSize());
        BB_ASSERT(y_buf.GetFrameSize() == t_buf.GetFrameSize());

        index_t frame_size  = t_buf.GetFrameSize();
        index_t node_size   = t_buf.GetNodeSize();

        {
            auto y_ptr  = y_buf.LockConst<T>();
            auto t_ptr  = t_buf.LockConst<T>();
 
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t node = 0; node < node_size; ++node) {
                    T   y = y_ptr.Get(frame, node);
                    T   t = t_ptr.Get(frame, node);
                    if ( (t > 0 && y > (T)0.5) || (t == 0 && y <= (T)0.5) ) {
                        m_accuracy += 1;
                    }
                    m_category_count += 1;
                }
            }
        }
    }
};


}

