// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include "bb/NeuralNetOptimizer.h"


namespace bb {


template <typename T = float, typename INDEX = size_t>
class ParamOptimizerSgd : public ParamOptimizer<T, INDEX>
{
protected:
	T	m_learning_rate;

public:
	ParamOptimizerSgd(T learning_rate=0.01)
	{
		m_learning_rate = learning_rate;
	}
	
	~ParamOptimizerSgd()
	{
	}

protected:
	void UpdateParam(INDEX index, T& param, const T grad)
	{
		param -= m_learning_rate * grad;
	}
};


template <typename T = float, typename INDEX = size_t>
class NeuralNetOptimizerSgd : public NeuralNetOptimizer<T, INDEX>
{
protected:
	T	m_learning_rate;

public:
	NeuralNetOptimizerSgd(T learning_rate = 0.01)
	{
		m_learning_rate = learning_rate;
	}

	ParamOptimizer<T, INDEX>* Create(INDEX param_size) const
	{
		return new ParamOptimizerSgd<T, INDEX>(m_learning_rate);
	}
};


}


