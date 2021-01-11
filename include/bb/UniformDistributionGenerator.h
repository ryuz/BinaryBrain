// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                 Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                 https://github.com/ryuz
//                                 ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <memory>
#include <random>

#include "bb/DataType.h"
#include "bb/ValueGenerator.h"

namespace bb {

template <typename T>
class UniformDistributionGenerator : public ValueGenerator<T>
{
    using _super = ValueGenerator<T>;

public:
    static inline std::string ValueGeneratorName(void) { return "UniformDistributionGenerator"; }
    static inline std::string ObjectName(void){ return ValueGeneratorName() + "_" + DataType<T>::Name(); }
    
    std::string GetValueGeneratorName(void) const override { return ValueGeneratorName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    T                                                       m_a;
    T                                                       m_b;
    std::int64_t                                            m_seed;

    std::unique_ptr< std::uniform_real_distribution<T> >    m_uniform_dist;
    std::mt19937_64                                         m_mt;

protected:
    UniformDistributionGenerator(T a = (T)0.0, T b = (T)1.0, std::int64_t seed = 1)
    {
        m_a    = a;
        m_b    = b;
        m_seed = seed;

        m_uniform_dist = std::unique_ptr< std::uniform_real_distribution<T> >(new std::uniform_real_distribution<T>(m_a, m_b));
        m_mt.seed(m_seed);
     }

public:
    static std::shared_ptr<UniformDistributionGenerator> Create(T a = (T)0.0, T b = (T)1.0, std::int64_t seed = 1)
    {
        return std::shared_ptr<UniformDistributionGenerator>(new UniformDistributionGenerator(a, b, seed));
    }

    void Seed(std::int64_t seed)
    {
        m_seed = seed;
        m_mt.seed(m_seed);
        m_uniform_dist->reset();
    }

    void Reset(void)
    {
        m_mt.seed(m_seed);
        m_uniform_dist->reset();
    }
    
    T GetValue(void)
    {
        return (*m_uniform_dist)(m_mt);
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
        bb::SaveValue(os, m_a);
        bb::SaveValue(os, m_b);
        bb::SaveValue(os, m_seed);
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
        bb::LoadValue(is, m_a);
        bb::LoadValue(is, m_b);
        bb::LoadValue(is, m_seed);

        // 再構築
        m_uniform_dist = std::unique_ptr< std::uniform_real_distribution<T> >(new std::uniform_real_distribution<T>(m_a, m_b));
        m_mt.seed(m_seed);
    }
};



// バイナリ用の特殊化
template <>
class UniformDistributionGenerator<Bit> : public ValueGenerator<Bit>
{
    using _super = ValueGenerator<Bit>;

public:
    static inline std::string ValueGeneratorName(void) { return "UniformDistributionGenerator"; }
    static inline std::string ObjectName(void){ return ValueGeneratorName() + "_" + DataType<Bit>::Name(); }
    
    std::string GetValueGeneratorName(void) const override { return ValueGeneratorName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    Bit                                                     m_a;
    Bit                                                     m_b;
    std::int64_t                                            m_seed;

    std::unique_ptr< std::uniform_int_distribution<int> >   m_uniform_dist;
    std::mt19937_64                                         m_mt;

protected:
    UniformDistributionGenerator(Bit a = 0, Bit b = 1, std::int64_t seed = 1)
    {
        m_a    = a;
        m_b    = b;
        m_seed = seed;

        m_uniform_dist = std::unique_ptr< std::uniform_int_distribution<int> >(new std::uniform_int_distribution<int>(m_a, m_b));
        m_mt.seed(m_seed);
    }

public:
    static std::shared_ptr<UniformDistributionGenerator> Create(Bit a = 0, Bit b = 1, std::int64_t seed = 1)
    {
        return std::shared_ptr<UniformDistributionGenerator>(new UniformDistributionGenerator(a, b, seed));
    }

    void Seed(std::int64_t seed)
    {
        m_seed = seed;
        m_mt.seed(m_seed);
        m_uniform_dist->reset();
    }

    void Reset(void)
    {
        m_mt.seed(m_seed);
        m_uniform_dist->reset();
    }
    
    Bit GetValue(void)
    {
        return (*m_uniform_dist)(m_mt);
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
        bb::SaveValue(os, m_a);
        bb::SaveValue(os, m_b);
        bb::SaveValue(os, m_seed);
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
        bb::LoadValue(is, m_a);
        bb::LoadValue(is, m_b);
        bb::LoadValue(is, m_seed);

        // 再構築
        m_uniform_dist = std::unique_ptr< std::uniform_int_distribution<int> >(new std::uniform_int_distribution<int>(m_a, m_b));
        m_mt.seed(m_seed);
    }
};


}
