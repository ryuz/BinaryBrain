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
    std::vector<Tensor>    m_tensors;

public:
    Variables(){}

    Variables(std::vector<int> const &types, std::vector<indices_t> const &shapes)
    {
        BB_ASSERT(shapes.size() == types.size());

        for ( size_t i = 0; i < shapes.size(); ++i ) {
            m_tensors.push_back(Tensor(types[i], shapes[i]));
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
            types.push_back(t.GetType());
        }
    }

    std::vector<indices_t> GetShapes(void) const
    {
        std::vector<indices_t> shapes;
        for ( auto& t : m_tensors ) {
            shapes.push_back(t.GetShape());
        }
    }
    
    void PushBack(Tensor const &t)
    {
        m_tensors.push_back(t);
    }
    

    // access operators
    Tensor const &operator[](index_t index) const
    {
        BB_DEBUG_ASSERT(index >= 0 && index < GetSize());
        return m_tensors[index];
    }

    Tensor &operator[](index_t index)
    {
        BB_DEBUG_ASSERT(index >= 0 && index < GetSize());
        return m_tensors[index];
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
        BB_ASSERT(m_tensors.size() == v.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            m_tensors[i] = src;
        }
        return *this;
    }

    Variables &operator+=(Variables const &src)
    {
        BB_ASSERT(m_tensors.size() == src.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            m_tensors[i] += src.m_tensors[i];
        }
        return *this;
    }

    template<typename Tp>
    Variables &operator+=(Tp src)
    {
        BB_ASSERT(m_tensors.size() == v.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            m_tensors[i] += src;
        }
        return *this;
    }

};



}


// end of file