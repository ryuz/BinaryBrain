// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>
#include "NeuralNetAccuracyFunction.h"


namespace bb {


template <typename T = float, typename INDEX = size_t>
class NeuralNetAccuracyCategoricalClassification : public NeuralNetAccuracyFunction<T, INDEX>
{
protected:
	INDEX m_num_classes;

public:
	NeuralNetAccuracyCategoricalClassification(INDEX num_classes)
	{
		m_num_classes = num_classes;
	}

	~NeuralNetAccuracyCategoricalClassification() {}

	double CalculateAccuracy(NeuralNetBuffer<T, INDEX> buf_sig, typename std::vector< std::vector<T> >::const_iterator exp_begin) const
	{
		double accuracy = 0;

		INDEX frame_size = buf_sig.GetFrameSize();
		INDEX node_size = buf_sig.GetNodeSize();
		for (INDEX frame = 0; frame < frame_size; ++frame) {
			INDEX	max_node = 0;
			T		max_signal = 0;
			for (INDEX node = 0; node < node_size; ++node) {
				T	sig = buf_sig.Get<T>(frame, node);
				if (sig > max_signal) {
					max_node = node;
					max_signal = sig;
				}
			}
			if ((*exp_begin)[max_node] > 0) {
				accuracy += 1.0;
			}
			exp_begin++;
		}

		return accuracy;
	}
};


}

