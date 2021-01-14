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
    using _super = Optimizer;

public:
    static inline std::string OptimizerName(void) { return "OptimizerAdam"; }
    static inline std::string ObjectName(void){ return OptimizerName() + "_" + DataType<T>::Name(); }
    
    std::string GetOptimizerName(void) const override { return OptimizerName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    T               m_learning_rate;
    T               m_beta1;
    T               m_beta2;
    int             m_iter;
    T               m_b1;
    T               m_b2;

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


protected:
    OptimizerAdam(create_t const &create)
    {
        m_learning_rate = create.learning_rate;
        m_beta1         = create.beta1;
        m_beta2         = create.beta2;
        m_iter          = 0;

        m_b1            = m_beta1;
        m_b2            = m_beta2;
    }

public:
    ~OptimizerAdam() {}

    static std::shared_ptr<OptimizerAdam> Create(create_t const &create) 
    {
        return std::shared_ptr<OptimizerAdam>(new OptimizerAdam(create));
    }

    static std::shared_ptr<OptimizerAdam> Create(T learning_rate = (T)0.001, T beta1 = (T)0.9, T beta2 = (T)0.999) 
    {
        create_t create;
        create.learning_rate = learning_rate;
        create.beta1         = beta1;
        create.beta2         = beta2;
        return std::shared_ptr<OptimizerAdam>(new OptimizerAdam(create));
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<OptimizerAdam> CreatePy(T learning_rate = (T)0.001, T beta1 = (T)0.9, T beta2 = (T)0.999) 
    {
        create_t create;
        create.learning_rate = learning_rate;
        create.beta1         = beta1;
        create.beta2         = beta2;
        return std::shared_ptr<OptimizerAdam>(new OptimizerAdam(create));
    }
#endif

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
    }
    

    void Update(void)
    {
        if ( m_params.IsEmpty() ) {
            return;
        }

#ifdef BB_WITH_CUDA
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
            auto lr_t = m_learning_rate * std::sqrt((T)1.0 - m_b2) / ((T)1.0 - m_b1 );

            m_m += ((T)1.0 - m_beta1) * (m_grads - m_m);
            m_v += ((T)1.0 - m_beta2) * (m_grads * m_grads - m_v);

            m_params -= lr_t * m_m / (Sqrt(m_v) + (T)1e-7);
            m_grads   = 0;

            m_b1 *= m_beta1;
            m_b2 *= m_beta2;
        }
    }


    // シリアライズ
protected:
    void DumpObjectData(std::ostream &os) const override
    {
        // バージョン
        std::int64_t ver = 1;
        bb::SaveValue(os, ver);

        // 親クラス
        _super::DumpObjectData(os);

        // メンバ
        bb::SaveValue(os, m_learning_rate);
        bb::SaveValue(os, m_beta1);
        bb::SaveValue(os, m_beta2);
        bb::SaveValue(os, m_iter);
        bb::SaveValue(os, m_b1);
        bb::SaveValue(os, m_b2);
    }

    void LoadObjectData(std::istream &is) override
    {
        // バージョン
        std::int64_t ver;
        bb::LoadValue(is, ver);

        BB_ASSERT(ver == 1);

        // 親クラス
        _super::LoadObjectData(is);

        // メンバ
        bb::LoadValue(is, m_learning_rate);
        bb::LoadValue(is, m_beta1);
        bb::LoadValue(is, m_beta2);
        bb::LoadValue(is, m_iter);
        bb::LoadValue(is, m_b1);
        bb::LoadValue(is, m_b2);
    }

};


}
