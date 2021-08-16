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
    using _super = MetricsFunction;

public:
    static inline std::string MetricsFunctionName(void) { return "MetricsCategoricalAccuracy"; }
    static inline std::string ObjectName(void){ return MetricsFunctionName() + "_" + DataType<T>::Name(); }
    
    std::string GetMetricsFunctionName(void) const override { return MetricsFunctionName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }


protected:
    int             m_accuracy = 0;
    int             m_category_count = 0;

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
        auto shape          = t_buf.GetShape();
        auto ch_size        = shape[0];
        auto pix_size       = node_size / ch_size;
        
#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && pix_size == 1 && y_buf.IsDeviceAvailable() && t_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            FrameBuffer accuracy_buf(frame_size, {1}, BB_TYPE_INT32);
            FrameBuffer category_buf(frame_size, {1}, BB_TYPE_INT32);

            {
                auto y_ptr        = y_buf.LockDeviceMemoryConst();
                auto t_ptr        = t_buf.LockDeviceMemoryConst();
                auto accuracy_ptr = accuracy_buf.LockDeviceMemory(true);
                auto category_ptr = category_buf.LockDeviceMemory(true);
                bbcu_MetricsCategoricalAccuracy<T>
                    (
                        (T   const  *)y_ptr.GetAddr(),
                        (T   const  *)t_ptr.GetAddr(),
                        (int        *)accuracy_ptr.GetAddr(),
                        (int        *)category_ptr.GetAddr(),
                        (int         )pix_size,
                        (int         )ch_size,
                        (int         )frame_size,
                        (int         )(y_buf.GetFrameStride() / sizeof(float))
                    );
            }

            m_accuracy       += (int)accuracy_buf.Sum();
            m_category_count += (int)category_buf.Sum();

            return;
        }
#endif

        /*
#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && pix_size == 1 && y_buf.IsDeviceAvailable() && t_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto y_ptr   = y_buf.LockDeviceMemoryConst();
            auto t_ptr   = t_buf.LockDeviceMemoryConst();
            auto acc_ptr = m_accuracy.LockDeviceMemory();

            bbcu_fp32_AccuracyCategoricalClassification
                (
                    (float const *)y_ptr.GetAddr(),
                    (float const *)t_ptr.GetAddr(),
                    (int         *)acc_ptr.GetAddr(),
                    (int          )y_buf.GetNodeSize(),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float))
                );

            m_category_count += y_buf.GetFrameSize();

            return;
        }
#endif
        */

        {
            auto y_ptr  = y_buf.LockConst<T>();
            auto t_ptr  = t_buf.LockConst<T>();
 
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t pix = 0; pix < pix_size; ++pix) {
                    T       max_y  = std::numeric_limits<T>::lowest();
                    T       max_t  = std::numeric_limits<T>::lowest();
                    bool    valid  = false;
                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        auto    y = y_ptr.Get(frame, node);
                        auto    t = t_ptr.Get(frame, node);
                        if (t > 0) {
                            valid = true;
                        }
                        if ( y > max_y ) {
                            max_y  = y;
                            max_t  = t;
                        }
                    }

                    if ( valid ) {
                        m_category_count += 1;
                        if ( max_t > 0 ) {
                            m_accuracy += 1;
                        }
                    }
                }
            }
        }
    }
};


}

