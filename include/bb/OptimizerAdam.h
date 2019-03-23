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
	T				m_beta1;
	T				m_beta2;
	int				m_iter;
	T				m_b1;
	T				m_b2;

    Variables       m_m;
    Variables       m_v;

    Variables       m_params;
    Variables       m_grads;

#ifdef BB_WITH_CUDA
    void *m_dev_size_table = nullptr;
	void *m_dev_params_buf_table = nullptr;
	void *m_dev_grads_buf_table = nullptr;
    void *m_dev_m_buf_table = nullptr;
    void *m_dev_v_buf_table = nullptr;
#endif

protected:
    OptimizerAdam() {}

public:
    ~OptimizerAdam() {
#ifdef BB_WITH_CUDA
        if ( m_dev_size_table       != nullptr ) { bbcu::Free(m_dev_size_table      ); }
        if ( m_dev_params_buf_table != nullptr ) { bbcu::Free(m_dev_params_buf_table); }
        if ( m_dev_grads_buf_table  != nullptr ) { bbcu::Free(m_dev_grads_buf_table ); }
        if ( m_dev_m_buf_table      != nullptr ) { bbcu::Free(m_dev_m_buf_table     ); }
        if ( m_dev_v_buf_table      != nullptr ) { bbcu::Free(m_dev_v_buf_table     ); }
#endif
    }

    struct create_t
    {
        T learning_rate = (T)0.001;
        T beta1         = (T)0.9;
        T beta2         = (T)0.999;
    };

   	static std::shared_ptr<OptimizerAdam> Create(create_t const &create) 
	{
        auto self = std::shared_ptr<OptimizerAdam>(new OptimizerAdam);

		self->m_learning_rate = create.learning_rate;
		self->m_beta1         = create.beta1;
		self->m_beta2         = create.beta2;
		self->m_iter          = 0;

        self->m_b1            = self->m_beta1;
        self->m_b2            = self->m_beta2;

        return self;
	}

  	static std::shared_ptr<OptimizerAdam> Create(T learning_rate = (T)0.001, T beta1 = (T)0.9, T beta2 = (T)0.999) 
	{
        auto self = std::shared_ptr<OptimizerAdam>(new OptimizerAdam);

		self->m_learning_rate = learning_rate;
		self->m_beta1         = beta1;
		self->m_beta2         = beta2;
		self->m_iter          = 0;

        self->m_b1            = self->m_beta1;
        self->m_b2            = self->m_beta2;

        return self;
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
	
	void SetVariables(Variables params, Variables grads)
    {
        BB_ASSERT(params.GetShapes() == grads.GetShapes());
        m_params = params;
        m_grads  = grads;

        m_m = Variables(params.GetTypes(), params.GetShapes());
        m_v = Variables(params.GetTypes(), params.GetShapes());
        m_m = 0;
        m_v = 0;

#ifdef BB_WITH_CUDA
        if ( m_dev_size_table       != nullptr ) { bbcu::Free(m_dev_size_table      ); }
        if ( m_dev_params_buf_table != nullptr ) { bbcu::Free(m_dev_params_buf_table); }
        if ( m_dev_grads_buf_table  != nullptr ) { bbcu::Free(m_dev_grads_buf_table ); }
        if ( m_dev_m_buf_table      != nullptr ) { bbcu::Free(m_dev_m_buf_table     ); }
        if ( m_dev_v_buf_table      != nullptr ) { bbcu::Free(m_dev_v_buf_table     ); }

        if ( m_params.IsDeviceAvailable() && m_grads.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            int size = (int)params.GetSize();
            std::vector<int>    size_table(size);
            std::vector<float*> params_buf_table(size);
            std::vector<float*> grads_buf_table(size);
            std::vector<float*> m_buf_table(size);
            std::vector<float*> v_buf_table(size);
            for ( index_t i = 0; i < params.GetSize(); ++i ) {
                auto param_tensor = m_params[i];
                auto grad_tensor  = m_grads[i];
                auto m_tensor     = m_m[i];
                auto v_tensor     = m_v[i];
            
                auto param_ptr = param_tensor.LockDeviceMemory();
                auto grad_ptr  = grad_tensor.LockDeviceMemory();
                auto m_ptr     = m_tensor.LockDeviceMemory();
                auto v_ptr     = v_tensor.LockDeviceMemory();     
            
                size_table[i]       = (int)param_tensor.GetSize();
                params_buf_table[i] = (float *)param_ptr.GetAddr();
                grads_buf_table[i]  = (float *)grad_ptr.GetAddr();
                m_buf_table[i]      = (float *)m_ptr.GetAddr();
                v_buf_table[i]      = (float *)v_ptr.GetAddr();
            }

            // デバイスメモリ確保
            bbcu::Malloc(&m_dev_size_table      , size * sizeof(int*));
            bbcu::Malloc(&m_dev_params_buf_table, size * sizeof(float*));
            bbcu::Malloc(&m_dev_grads_buf_table , size * sizeof(float*));
            bbcu::Malloc(&m_dev_m_buf_table     , size * sizeof(float*));
            bbcu::Malloc(&m_dev_v_buf_table     , size * sizeof(float*));

            bbcu::Memcpy(m_dev_size_table,       &size_table[0],       size * sizeof(int*),   cudaMemcpyHostToDevice);
            bbcu::Memcpy(m_dev_params_buf_table, &params_buf_table[0], size * sizeof(float*), cudaMemcpyHostToDevice);
            bbcu::Memcpy(m_dev_grads_buf_table,  &grads_buf_table[0],  size * sizeof(float*), cudaMemcpyHostToDevice);
            bbcu::Memcpy(m_dev_m_buf_table,      &m_buf_table[0],      size * sizeof(float*), cudaMemcpyHostToDevice);
            bbcu::Memcpy(m_dev_v_buf_table,      &v_buf_table[0],      size * sizeof(float*), cudaMemcpyHostToDevice);
        }
#endif
    }
    
	void Update(void)
	{
#ifdef BB_WITH_CUDA
        if ( m_dev_size_table && m_dev_params_buf_table && m_dev_grads_buf_table && m_dev_m_buf_table && m_dev_v_buf_table ) {
            auto lr_t = m_learning_rate * std::sqrt((T)1.0 - m_b2) / ((T)1.0 - m_b1 /* + 1.0e-7 */ );

            for ( index_t i = 0; i < m_params.GetSize(); ++i ) {
                auto param_tensor = m_params[i];
                auto grad_tensor  = m_grads[i];
                auto m_tensor     = m_m[i];
                auto v_tensor     = m_v[i];
            
                auto param_ptr = param_tensor.LockDeviceMemory();
                auto grad_ptr  = grad_tensor.LockDeviceMemoryConst();
                auto m_ptr     = m_tensor.LockDeviceMemory();
                auto v_ptr     = v_tensor.LockDeviceMemory();
            }

            bbcu_fp32_Adam
		            (
                        (int                  )m_params.GetSize(),
                        (int           const *)m_dev_size_table,
			            (float       * const *)m_dev_params_buf_table,
			            (float const * const *)m_dev_grads_buf_table,
    		            (float       * const *)m_dev_m_buf_table,
    		            (float       * const *)m_dev_v_buf_table,
 	                    (float				  )lr_t,
	                    (float				  )m_beta1,
	                    (float				  )m_beta2
                    );
            
            m_b1 *= m_beta1;
            m_b2 *= m_beta2;       
            return;
        }
#endif

        auto lr_t = m_learning_rate * std::sqrt((T)1.0 - m_b2) / ((T)1.0 - m_b1 + 1.0e-7 );

        m_m += ((T)1.0 - m_beta1) * (m_grads - m_m);
        m_v += ((T)1.0 - m_beta2) * (m_grads * m_grads - m_v);
        m_params -= lr_t * m_m / (Sqrt(m_v) + (T)1e-7);

        m_b1 *= m_beta1;
        m_b2 *= m_beta2;
    }

#if 0
    Variables       m_lr_t;
    Variables       m_tmp0;
    Variables       m_tmp1;
	void Update2(void)
	{
        m_lr_t  = 1.0f;
        m_lr_t -= m_b2;
        m_lr_t.Sqrt();

        m_tmp0  = 1.0f;
        m_tmp0 -= m_b1;

        m_lr_t /= m_tmp0;

        m_tmp2 
    }
#endif

};


}


