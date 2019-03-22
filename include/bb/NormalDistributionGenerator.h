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
protected:
    std::normal_distribution<T> m_norm_dist;
    std::mt19937_64             m_mt;
    std::int64_t                m_seed;

protected:
    NormalDistributionGenerator(T mean = (T)0.0, T stddev = (T)1.0, std::int64_t seed = 1)
        : m_norm_dist(mean, stddev), m_mt(seed), m_seed(seed)
    {
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
    }

    void Reset(void)
    {
        m_norm_dist.reset();
        m_mt.seed(m_seed);
    }
    
    T GetValue(void)
    {
        return m_norm_dist(m_mt);
    }
};


}
