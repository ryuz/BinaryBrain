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
class ParamOptimizerAdam : public ParamOptimizer<T, INDEX>
{
protected:
	T				m_learning_rate;
	T				m_lr_t;
	T				m_beta1;
	T				m_beta2;
	int				m_iter;
	std::vector<T>	m_m;
	std::vector<T>	m_v;
	
public:
	ParamOptimizerAdam(INDEX param_size, T learning_rate = (T)0.001, T beta1 = (T)0.9, T beta2 = (T)0.999)
	{
		m_learning_rate = learning_rate;
		m_beta1         = beta1;
		m_beta2         = beta2;
		m_iter          = 0;
		m_m.resize(param_size, (T)0);
		m_v.resize(param_size, (T)0);
	}
	
	~ParamOptimizerAdam()
	{
	}

protected:
	void PreUpdate(void)
	{
		m_iter++;
		m_lr_t = m_learning_rate * sqrt((T)1.0 - pow(m_beta2, (T)m_iter)) / ((T)1.0 - pow(m_beta1, (T)m_iter));
	}

	void UpdateParam(INDEX index, T& param, const T grad)
	{
		BB_ASSERT(index >= 0 && index < m_m.size());
		
		m_m[index] += ((T)1 - m_beta1) * (grad - m_m[index]);
		m_v[index] += ((T)1 - m_beta2) * (grad*grad - m_v[index]);
		
		param -= m_lr_t * m_m[index] / (sqrt(m_v[index]) + (T)1e-7);
	}
};


template <typename T = float, typename INDEX = size_t>
class NeuralNetOptimizerAdam: public NeuralNetOptimizer<T, INDEX>
{
protected:
	T	m_learning_rate;
	T	m_beta1;
	T	m_beta2;

public:
	NeuralNetOptimizerAdam(T learning_rate = (T)0.001, T beta1 = (T)0.9, T beta2 = (T)0.999)
	{
		m_learning_rate = learning_rate;
		m_beta1         = beta1;
		m_beta2         = beta2;
	}
	
	ParamOptimizer<T, INDEX>* Create(INDEX param_size) const
	{
		return new ParamOptimizerAdam<T, INDEX>(param_size, m_learning_rate, m_beta1, m_beta2);
	}
};


}


