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
    using _super = Optimizer;

public:
    static inline std::string OptimizerName(void) { return "OptimizerSgd"; }
    static inline std::string ObjectName(void){ return OptimizerName() + "_" + DataType<T>::Name(); }
    
    std::string GetOptimizerName(void) const override { return OptimizerName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

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

    void SetVariables(Variables params, Variables grads) override
    {
        BB_ASSERT(params.GetShapes() == grads.GetShapes());
        m_params = params;
        m_grads  = grads;
    }
    
    void Update(void) override
    {
        if ( m_params.IsEmpty() ) {
            return;
        }

        m_params -= m_learning_rate * m_grads;
        m_grads   = 0;
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
    }
};


}


