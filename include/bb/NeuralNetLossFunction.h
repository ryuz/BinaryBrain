// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>

#include "bb/NeuralNetBuffer.h"


namespace bb {


template <typename T = float, typename INDEX = size_t>
class NeuralNetLossFunction
{
public:
	virtual double CalculateLoss(NeuralNetBuffer<T, INDEX> buf_sig, NeuralNetBuffer<T, INDEX> buf_err, typename std::vector< std::vector<T> >::const_iterator t_begin) const = 0;
};


}

