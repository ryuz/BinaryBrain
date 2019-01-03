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


template <typename T = float>
class ParamOptimizerSgd : public ParamOptimizer<T>
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


template <typename T = float>
class NeuralNetOptimizerSgd : public NeuralNetOptimizer<T>
{
protected:
	T	m_learning_rate;

public:
	NeuralNetOptimizerSgd(T learning_rate = 0.01)
	{
		m_learning_rate = learning_rate;
	}

	ParamOptimizer<T>* Create(INDEX param_size) const
	{
		return new ParamOptimizerSgd<T>(m_learning_rate);
	}
};


}


