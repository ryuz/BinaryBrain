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
class OptimizerAdaGrad : public Optimizer
{
protected:
    T               m_learning_rate;
    
    Variables       m_h;
    
    Variables       m_params;
    Variables       m_grads;
    
protected:
    OptimizerAdaGrad() {}
    
public:
    ~OptimizerAdaGrad() {}
    
    struct create_t
    {
        T learning_rate = (T)0.01;
    };
    
    static std::shared_ptr<OptimizerAdaGrad> Create(create_t const &create) 
    {
        auto self = std::shared_ptr<OptimizerAdaGrad>(new OptimizerAdaGrad);
        
        self->m_learning_rate = create.learning_rate;
        
        return self;
    }
    
    static std::shared_ptr<OptimizerAdaGrad> Create(T learning_rate = (T)0.01) 
    {
        auto self = std::shared_ptr<OptimizerAdaGrad>(new OptimizerAdaGrad);
        
        self->m_learning_rate = learning_rate;
        
        return self;
    }
    
    void SetVariables(Variables params, Variables grads)
    {
        BB_ASSERT(params.GetShapes() == grads.GetShapes());
        m_params = params;
        m_grads  = grads;

        m_h = Variables(params.GetTypes(), params.GetShapes());
        m_h = 0;
    }
    

    void Update(void)
    {

#if 0 // #ifdef BB_WITH_CUDA
        if ( m_params.IsDeviceAvailable() && m_grads.IsDeviceAvailable() && m_m.IsDeviceAvailable() && m_v.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto lr_t = m_learning_rate * std::sqrt((T)1.0 - m_b2) / ((T)1.0 - m_b1 );
            bbcu_fp32_Adam
                    (
                        (int            )m_params.GetSize(),
                        (int     const *)m_params.GetDeviceSizeTable(),
                        (float * const *)m_params.GetDeviceAddrTable(),
                        (float * const *)m_grads.GetDeviceAddrTable(),
                        (float * const *)m_m.GetDeviceAddrTable(),
                        (float * const *)m_v.GetDeviceAddrTable(),
                        (float          )lr_t,
                        (float          )m_beta1,
                        (float          )m_beta2
                    );
            
            m_b1 *= m_beta1;
            m_b2 *= m_beta2;       
            return;
        }
#endif
        
        {
            // 汎用版
            m_h += (m_grads * m_grads);

            m_params -= m_learning_rate * m_grads / (Sqrt(m_h) + (T)1e-7);
            m_grads   = 0;
        }
    }
};


}


