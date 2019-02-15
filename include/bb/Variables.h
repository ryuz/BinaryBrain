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

    Variables& operator=(Variables const &v)
    {
        m_tensors = v.m_tensors;
        return *this;
    }

    Variables& operator+=(Variables const &v)
    {
        BB_ASSERT(m_tensors.size() == v.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            m_tensors[i] += v.m_tensors[i];
        }
        return *this;
    }

};



}


// end of file