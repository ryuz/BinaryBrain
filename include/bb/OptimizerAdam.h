// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include "bb/Optimizer.h"
#include "bb/Variables.h"


namespace bb {


template <typename T = float>
class OptimizerAdam : public Optimizer
{
protected:
	T				m_learning_rate;
//	T				m_lr_t;
	T				m_beta1;
	T				m_beta2;
	int				m_iter;
	T				m_b1;
	T				m_b2;

    Variables       m_m;
    Variables       m_v;

    Variables       m_params;
    Variables       m_grads;
	
public:
    struct create_t
    {
        T learning_rate = (T)0.001;
        T beta1         = (T)0.9;
        T beta2         = (T)0.999;
    };

   	OptimizerAdam(create_t const &create) 
	{
		m_learning_rate = create.learning_rate;
		m_beta1         = create.beta1;
		m_beta2         = create.beta2;
		m_iter          = 0;

        m_b1            = m_beta1;
        m_b2            = m_beta2;
	}

  	OptimizerAdam(T learning_rate = (T)0.001, T beta1 = (T)0.9, T beta2 = (T)0.999) 
	{
		m_learning_rate = learning_rate;
		m_beta1         = beta1;
		m_beta2         = beta2;
		m_iter          = 0;

        m_b1            = m_beta1;
        m_b2            = m_beta2;
	}

	OptimizerAdam(create_t const &create, Variables params, Variables grads) 
        : m_m(params.GetTypes(), params.GetShapes()), m_v(params.GetTypes(), params.GetShapes())
	{
        BB_ASSERT(params.GetShapes() == grads.GetShapes());
        m_params        = params;
        m_grads         = grads;

        m_m = 0;
        m_v = 0;

		m_learning_rate = create.learning_rate;
		m_beta1         = create.beta1;
		m_beta2         = create.beta2;
		m_iter          = 0;

        m_b1            = m_beta1;
        m_b2            = m_beta2;
    }
	
	~OptimizerAdam()
	{
	}

	void SetVariables(Variables params, Variables grads)
    {
        BB_ASSERT(params.GetShapes() == grads.GetShapes());
        m_params = params;
        m_grads  = grads;

        m_m = Variables(params.GetTypes(), params.GetShapes());
        m_v = Variables(params.GetTypes(), params.GetShapes());
        m_m = 0;
        m_v = 0;
    }
    
	void Update(void)
	{
#if 1
        auto lr_t = m_learning_rate * sqrt((T)1.0 - m_b2) / ((T)1.0 - m_b1 + 1.0e-7);

        m_m += ((T)1.0 - m_beta1) * (m_grads - m_m);
        m_v += ((T)1.0 - m_beta2) * (m_grads * m_grads - m_v);
        m_params -= lr_t * m_m / (m_v.Sqrt() + (T)1e-7);

        m_b1 *= m_beta1;
        m_b2 *= m_beta2;

#else
		m_iter++;
        m_b1 = pow(m_beta1, (T)m_iter);
        m_b2 = pow(m_beta2, (T)m_iter);

        auto lr_t = m_learning_rate * sqrt((T)1.0 - m_b2) / ((T)1.0 - m_b1 + 1.0e-7);
        m_m += ((T)1 - m_beta1) * (m_grads - m_m);
        m_v += ((T)1 - m_beta2) * (m_grads * m_grads - m_v);
        m_params -= lr_t * m_m / (m_v.sqrt() + (T)1e-7);


//      auto lr_t = m_learning_rate * sqrt((T)1.0 - pow(m_beta2, (T)m_iter)) / ((T)1.0 - pow(m_beta1, (T)m_iter) + 1.0e-7);
//      m_m += ((T)1 - m_beta1) * (m_grads - m_m);
//      m_v += ((T)1 - m_beta2) * (m_grads * m_grads - m_v);
//      m_params -= lr_t * m_m / (m_v.sqrt() + (T)1e-7);
#endif
    }
};


}


