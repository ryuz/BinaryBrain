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
class NeuralNetOptimizerSgd : public NeuralNetOptimizer<T, INDEX>
{
protected:
	T	m_learning_rate;

public:
	NeuralNetOptimizerSgd(T learning_rate=0.01)
	{
		m_learning_rate = learning_rate;
	}
	
	~NeuralNetOptimizerSgd()
	{
	}

protected:
	void UpdateParam(INDEX index, T& param, const T grad)
	{
		param -= m_learning_rate * grad;
	}
};


template <typename T = float, typename INDEX = size_t>
class NeuralNetOptimizerSgdCreator : public NeuralNetOptimizerCreator<T, INDEX>
{
protected:
	T	m_learning_rate;

public:
	NeuralNetOptimizerSgdCreator(T learning_rate = 0.01)
	{
		m_learning_rate = learning_rate;
	}

	NeuralNetOptimizer<T, INDEX>* Create(INDEX param_size) const
	{
		return new NeuralNetOptimizerSgd<T, INDEX>(m_learning_rate);
	}
};


}


