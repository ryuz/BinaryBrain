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
protected:
    std::uniform_real_distribution<T>   m_uniform_dist;
    std::mt19937_64                     m_mt;
    std::int64_t                        m_seed;

protected:
    UniformDistributionGenerator(T a = (T)0.0, T b = (T)1.0, std::int64_t seed = 1)
        : m_uniform_dist(a, b), m_mt(seed), m_seed(seed)
    {
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
    }

    void Reset(void)
    {
        m_uniform_dist.reset();
        m_mt.seed(m_seed);
    }
    
    T GetValue(void)
    {
        return m_uniform_dist(m_mt);
    }
};


template <>
class UniformDistributionGenerator<Bit> : public ValueGenerator<Bit>
{
protected:
    std::uniform_int_distribution<int>  m_uniform_dist;
    std::mt19937_64                     m_mt;
    std::int64_t                        m_seed;

protected:
    UniformDistributionGenerator(Bit a = 0, Bit b = 1, std::int64_t seed = 1)
        : m_uniform_dist(a, b), m_mt(seed), m_seed(seed)
    {
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
    }

    void Reset(void)
    {
        m_uniform_dist.reset();
        m_mt.seed(m_seed);
    }
    
    Bit GetValue(void)
    {
        return m_uniform_dist(m_mt);
    }
};


}
