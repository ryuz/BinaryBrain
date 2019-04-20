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
    T               m_learning_rate;
    Variables       m_params;
    Variables       m_grads;

protected:
    OptimizerSgd() {}

public:
    ~OptimizerSgd() {}

    struct create_t
    {
        T learning_rate = (T)0.01;
    };

    static std::shared_ptr<OptimizerSgd> Create(create_t const &create) 
    {
        auto self = std::shared_ptr<OptimizerSgd>(new OptimizerSgd);

        self->m_learning_rate = create.learning_rate;

        return self;
    }

    static std::shared_ptr<OptimizerSgd> Create(T learning_rate=(T)0.01)
    {
        auto self = std::shared_ptr<OptimizerSgd>(new OptimizerSgd);

        self->m_learning_rate = learning_rate;

        return self;
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
        m_grads   = 0;
    }
};


}


