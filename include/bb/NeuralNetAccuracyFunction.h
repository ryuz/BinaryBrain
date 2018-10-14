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
class NeuralNetAccuracyFunction
{
public:
	virtual ~NeuralNetAccuracyFunction() {}
	
	virtual double CalculateAccuracy(NeuralNetBuffer<T, INDEX> buf_sig, typename std::vector< std::vector<T> >::const_iterator t_begin) const = 0;
};


}

