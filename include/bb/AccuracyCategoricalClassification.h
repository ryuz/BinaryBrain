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

public:
	AccuracyCategoricalClassification(index_t num_classes)
	{
		m_num_classes = num_classes;
	}

	~AccuracyCategoricalClassification() {}

	double CalculateAccuracy(FrameBuffer y, FrameBuffer t)
	{
        BB_ASSERT(y.GetType() == DataType<T>::type);
        BB_ASSERT(t.GetType() == DataType<T>::type);

		double accuracy = 0;

		index_t frame_size = y.GetFrameSize();
		index_t node_size = y.GetNodeSize();

        auto y_ptr  = y.GetConstPtr<T>();
        auto t_ptr  = t.GetConstPtr<T>();
 
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
				accuracy += 1.0;
			}
		}

		return accuracy;
	}
};


}

