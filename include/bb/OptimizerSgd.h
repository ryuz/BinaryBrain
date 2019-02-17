// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include "bb/Optimizer.h"


namespace bb {


template <typename T = float>
class OptimizerSgd : public Optimizer
{
protected:
	T	            m_learning_rate;
    Variables       m_params;
    Variables       m_grads;

public:
    struct create_t
    {
        T learning_rate = (T)0.01;
    };

   	OptimizerSgd(create_t const &create) 
	{
		m_learning_rate = create.learning_rate;
	}

    OptimizerSgd(T learning_rate=(T)0.01)
	{
		m_learning_rate = learning_rate;
	}
	
	~OptimizerSgd()
	{
	}

    void SetVariables(Variables params, Variables grads)
    {
        BB_ASSERT(params.GetShapes() == grads.GetShapes());
        m_params = params;
        m_grads  = grads;
    }
    
	void Update(void)
	{
   		m_params -= m_learning_rate * m_grads;
    }
};


}


