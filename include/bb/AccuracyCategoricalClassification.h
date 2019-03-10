// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>

#include "bb/AccuracyFunction.h"


namespace bb {


template <typename T = float>
class AccuracyCategoricalClassification : public AccuracyFunction
{
protected:
	index_t m_num_classes;
	double  m_accuracy = 0;
    index_t m_frames;


protected:
    AccuracyCategoricalClassification()	{}

public:
	~AccuracyCategoricalClassification() {}

    static std::shared_ptr<AccuracyCategoricalClassification> Create(index_t num_classes)
    {
        auto self = std::shared_ptr<AccuracyCategoricalClassification>(new AccuracyCategoricalClassification);
		self->m_num_classes = num_classes;
        return self;
    }
     
    void Clear(void)
    {
        m_accuracy = 0;
        m_frames   = 0;
    }


    double GetAccuracy(void) const
    {
        return m_accuracy / (double)m_frames;
    }

	void CalculateAccuracy(FrameBuffer y, FrameBuffer t)
	{
        BB_ASSERT(y.GetType() == DataType<T>::type);
        BB_ASSERT(t.GetType() == DataType<T>::type);

		index_t frame_size = y.GetFrameSize();
		index_t node_size = y.GetNodeSize();

        m_frames += frame_size;

        auto y_ptr  = y.LockConst<T>();
        auto t_ptr  = t.LockConst<T>();
 
		for (index_t frame = 0; frame < frame_size; ++frame) {
			index_t	max_node = 0;
			T		max_signal = 0;
			for (index_t node = 0; node < node_size; ++node) {
				T	sig = y_ptr.Get(frame, node);
				if (sig > max_signal) {
					max_node   = node;
					max_signal = sig;
				}
			}
			if ( t_ptr.Get(frame, max_node) > 0) {
				m_accuracy += 1.0;
			}
		}
	}
};


}

