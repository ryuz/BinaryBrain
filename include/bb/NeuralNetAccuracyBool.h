// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>

#include "bb/NeuralNetAccuracyFunction.h"


namespace bb {


template <typename T = float>
class NeuralNetAccuracyBool : public NeuralNetAccuracyFunction<T>
{
public:
	NeuralNetAccuracyBool()
	{
	}

	~NeuralNetAccuracyBool() {}

	double CalculateAccuracy(NeuralNetBuffer<T> buf_sig, typename std::vector< std::vector<T> >::const_iterator exp_begin) const
	{
		double accuracy = 0;

		INDEX frame_size = buf_sig.GetFrameSize();
		INDEX node_size = buf_sig.GetNodeSize();
		for (INDEX frame = 0; frame < frame_size; ++frame) {
			for (INDEX node = 0; node < node_size; ++node) {
				bool	sig = (buf_sig.template Get<T>(frame, node) >= 0.5);
				bool	exp = ((*exp_begin)[node] >= 0.5);

				if (sig == exp) {
					accuracy += 1.0;
				}
			}
			exp_begin++;
		}

		accuracy /= (double)node_size;

		return accuracy;
	}
};


}

