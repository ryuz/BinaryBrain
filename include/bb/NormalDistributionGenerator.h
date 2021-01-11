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

#include "bb/ValueGenerator.h"

namespace bb {

template <typename T>
class NormalDistributionGenerator : public ValueGenerator<T>
{
    using _super = ValueGenerator<T>;

public:
    static inline std::string ValueGeneratorName(void) { return "NormalDistributionGenerator"; }
    static inline std::string ObjectName(void){ return ValueGeneratorName() + "_" + DataType<T>::Name(); }
    
    std::string GetValueGeneratorName(void) const override { return ValueGeneratorName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    T                                               m_mean = (T)0.0;
    T                                               m_stddev = (T)1.0;
    std::int64_t                                    m_seed = 1;

    std::unique_ptr< std::normal_distribution<T> >  m_norm_dist;
    std::mt19937_64                                 m_mt;

protected:
    NormalDistributionGenerator(T mean = (T)0.0, T stddev = (T)1.0, std::int64_t seed = 1)
    {
        m_mean   = mean;
        m_stddev = stddev;
        m_seed   = seed;

        m_norm_dist = std::unique_ptr< std::normal_distribution<T> >(new std::normal_distribution<T>(m_mean, m_stddev));
        m_mt.seed(m_seed);
    }

public:
    static std::shared_ptr<NormalDistributionGenerator> Create(T mean = (T)0.0, T stddev = (T)1.0, std::int64_t seed = 1)
    {
        return std::shared_ptr<NormalDistributionGenerator>(new NormalDistributionGenerator(mean, stddev, seed));
    }

    void Seed(std::int64_t seed)
    {
        m_seed = seed;
        m_mt.seed(m_seed);
        m_norm_dist->reset();
    }

    void Reset(void)
    {
        m_mt.seed(m_seed);
        m_norm_dist->reset();
    }
    
    T GetValue(void)
    {
        return (*m_norm_dist)(m_mt);
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
        bb::SaveValue(os, m_mean);
        bb::SaveValue(os, m_stddev);
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
        bb::LoadValue(is, m_mean);
        bb::LoadValue(is, m_stddev);
        bb::LoadValue(is, m_seed);

        // 再構築
        m_norm_dist = std::unique_ptr< std::normal_distribution<T> >(new std::normal_distribution<T>(m_mean, m_stddev));
        m_mt.seed(m_seed);
    }
};


}
