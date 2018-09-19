// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include "NeuralNetOptimizer.h"


namespace bb {


template <typename T = float, typename INDEX = size_t>
class NeuralNetOptimizerSgd : public NeuralNetOptimizer
{
protected:
	T	m_learning_rate;

public:
	NeuralNetOptimizerSgd(T learning_rate)
	{
		m_learning_rate = learning_rate;
	}
	
	~NeuralNetOptimizer()
	{
	}

	void Optimize(std::vector<T>& param, const std::vector<T>& grad)
	{
		for (INDEX i = 0; i < (INDEX)param.size(); ++i) {
			param[i] -= learning_rate * grad[i];
		}
	}
};


template <typename T = float, typename INDEX = size_t>
class NeuralNetOptimizerSgdCreator : public NeuralNetOptimizerCreator
{
protected:
	T	m_learning_rate;

public:
	NeuralNetOptimizerSgdCreator(T learning_rate)
	{
		m_learning_rate = learning_rate;
	}

	NeuralNetOptimizer* Create(INDEX param_size)
	{
		return new NeuralNetOptimizerSgd(m_learning_rate);
	}
};


}


