// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>
#include "NeuralNetBuffer.h"


namespace bb {


template <typename T = float, typename INDEX = size_t>
class NeuralNetLossFunction
{
public:
	virtual ~NeuralNetLossFunction() {}
	
//	virtual T CalculateLoss(NeuralNetBuffer buf_sig, NeuralNetBuffer buf_err, std::vector< std::vector<T> >::iterator exp_begin) const = 0;
};


}

