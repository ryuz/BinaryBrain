// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                 Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                 https://github.com/ryuz
//                                 ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include "bb/Object.h"


namespace bb {

template <typename T>
class ValueGenerator : public Object
{
public:
    virtual std::string GetValueGeneratorName(void) const = 0;

    virtual ~ValueGenerator(){}
    virtual void Reset(void)    = 0;
    virtual T    GetValue(void) = 0;
};


}
