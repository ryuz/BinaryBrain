// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <vector>

namespace bb {


template <typename T = float, typename INDEX = size_t>
class NeuralNetOptimizer
{
public:
	virtual ~NeuralNetOptimizer() {}

	virtual void Optimize(std::vector<T>& param, const std::vector<T>& grad) = 0;
};


template <typename T = float, typename INDEX = size_t>
class NeuralNetOptimizerCreator
{
public:
	virtual NeuralNetOptimizer* Create(INDEX param_size) = 0;
};


}