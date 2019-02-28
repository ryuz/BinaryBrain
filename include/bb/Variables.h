// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include "bb/DataType.h"
#include "bb/Tensor.h"

namespace bb
{


class Variables
{
protected:
    std::vector< std::shared_ptr<Tensor> >    m_tensors;

public:
    Variables(){}

    Variables(std::vector<int> const &types, std::vector<indices_t> const &shapes)
    {
        BB_ASSERT(shapes.size() == types.size());

        for ( size_t i = 0; i < shapes.size(); ++i ) {
            m_tensors.push_back(std::make_shared<Tensor>(types[i], shapes[i]));
        }
    }

    Variables(Variables const &v)
    {
        m_tensors = v.m_tensors;
    }


    ~Variables(){}

    index_t GetSize(void) const
    {
        return (index_t)m_tensors.size();
    }

    std::vector<int> GetTypes(void) const
    {
        std::vector<int> types;
        for ( auto& t : m_tensors ) {
            types.push_back(t->GetType());
        }
        return types;
    }

    std::vector<indices_t> GetShapes(void) const
    {
        std::vector<indices_t> shapes;
        for ( auto& t : m_tensors ) {
            shapes.push_back(t->GetShape());
        }
        return shapes;
    }
    
    void PushBack(std::shared_ptr<Tensor> t)
    {
        m_tensors.push_back(t);
    }

    void PushBack(Variables const &v)
    {
        for ( auto& t : v.m_tensors ) {
            m_tensors.push_back(t);
        }
    }
    
    // access operators
    Tensor const &operator[](index_t index) const
    {
        BB_DEBUG_ASSERT(index >= 0 && index < GetSize());
        return *m_tensors[index];
    }

    Tensor &operator[](index_t index)
    {
        BB_DEBUG_ASSERT(index >= 0 && index < GetSize());
        return *m_tensors[index];
    }


    // arithmetic operator
    Variables &operator=(Variables const &src)
    {
        m_tensors = src.m_tensors;
        return *this;
    }
    
    template<typename Tp>
    Variables &operator=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] = src;
        }
        return *this;
    }


    Variables &operator+=(Variables const &src)
    {
        BB_ASSERT(m_tensors.size() == src.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] += *src.m_tensors[i];
        }
        return *this;
    }

    template<typename Tp>
    Variables &operator+=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] += src;
        }
        return *this;
    }

    Variables &operator-=(Variables const &src)
    {
        BB_ASSERT(m_tensors.size() == src.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] -= *src.m_tensors[i];
        }
        return *this;
    }

    template<typename Tp>
    Variables &operator-=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] -= src;
        }
        return *this;
    }

    Variables &operator*=(Variables const &src)
    {
        BB_ASSERT(m_tensors.size() == src.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] *= *src.m_tensors[i];
        }
        return *this;
    }

    template<typename Tp>
    Variables &operator*=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] *= src;
        }
        return *this;
    }

    Variables &operator/=(Variables const &src)
    {
        BB_ASSERT(m_tensors.size() == src.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] /= *src.m_tensors[i];
        }
        return *this;
    }

    template<typename Tp>
    Variables &operator/=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] /= src;
        }
        return *this;
    }

    friend  Variables operator + (Variables const &src0, Variables const &src1);
    friend  Variables operator + (Variables const &src0, double src1);
    friend  Variables operator + (double src0, Variables const &src1);
    friend  Variables operator - (Variables const &src0, Variables const &src1);
    friend  Variables operator - (Variables const &src0, double src1);
    friend  Variables operator - (double src0, Variables const &src1);
    friend  Variables operator * (Variables const &src0, Variables const &src1);
    friend  Variables operator * (Variables const &src0, double src1);
    friend  Variables operator * (double src0, Variables const &src1);
    friend  Variables operator / (Variables const &src0, Variables const &src1);
    friend  Variables operator / (Variables const &src0, double src1);
    friend  Variables operator / (double src0, Variables const &src1);

    Variables Sqrt(void)
    {
        Variables var(GetTypes(), GetShapes());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *var.m_tensors[i] = m_tensors[i]->Sqrt();
        }
        return var;
    }

    Variables Exp(void)
    {
        Variables var(GetTypes(), GetShapes());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *var.m_tensors[i] = m_tensors[i]->Exp();
        }
        return var;
    }
};

inline Variables operator+(Variables const &src0, Variables const &src1)
{
    BB_ASSERT(src0.GetTypes() == src1.GetTypes());
    BB_ASSERT(src0.GetShapes() == src1.GetShapes());

    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] + *src1.m_tensors[i];
    }
    return var;
}

inline Variables operator+(Variables const & src0, double src1)
{
    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] + src1;
    }
    return var;
}

inline Variables operator+(double src0, Variables const & src1)
{
    Variables var(src1.GetTypes(), src1.GetShapes());
    for ( size_t i = 0; i < src1.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = src0 + *src1.m_tensors[i];
    }
    return var;
}


inline Variables operator-(Variables const &src0, Variables const &src1)
{
    BB_ASSERT(src0.GetTypes() == src1.GetTypes());
    BB_ASSERT(src0.GetShapes() == src1.GetShapes());

    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] - *src1.m_tensors[i];
    }
    return var;
}

inline Variables operator-(Variables const & src0, double src1)
{
    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] - src1;
    }
    return var;
}

inline Variables operator-(double src0, Variables const & src1)
{
    Variables var(src1.GetTypes(), src1.GetShapes());
    for ( size_t i = 0; i < src1.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = src0 - *src1.m_tensors[i];
    }
    return var;
}


inline Variables operator*(Variables const &src0, Variables const &src1)
{
    BB_ASSERT(src0.GetTypes() == src1.GetTypes());
    BB_ASSERT(src0.GetShapes() == src1.GetShapes());

    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] * *src1.m_tensors[i];
    }
    return var;
}

inline Variables operator*(Variables const & src0, double src1)
{
    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] * src1;
    }
    return var;
}

inline Variables operator*(double src0, Variables const & src1)
{
    Variables var(src1.GetTypes(), src1.GetShapes());
    for ( size_t i = 0; i < src1.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = src0 * *src1.m_tensors[i];
    }
    return var;
}


inline Variables operator/(Variables const &src0, Variables const &src1)
{
    BB_ASSERT(src0.GetTypes() == src1.GetTypes());
    BB_ASSERT(src0.GetShapes() == src1.GetShapes());

    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] / *src1.m_tensors[i];
    }
    return var;
}

inline Variables operator/(Variables const & src0, double src1)
{
    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] / src1;
    }
    return var;
}

inline Variables operator/(double src0, Variables const & src1)
{
    Variables var(src1.GetTypes(), src1.GetShapes());
    for ( size_t i = 0; i < src1.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = src0 / *src1.m_tensors[i];
    }
    return var;
}


inline std::ostream& operator<<(std::ostream& os, Variables const &v)
{
    for (index_t i = 0; i < v.GetSize(); ++i) {
        os << v[i];
    }
    return os;
}



}


// end of file