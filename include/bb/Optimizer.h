// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include "bb/Object.h"
#include "bb/Variables.h"


namespace bb {

class Optimizer : public Object
{
public:
    virtual ~Optimizer() {}

public:
    virtual std::string GetOptimizerName(void) const = 0;

    virtual void SetVariables(Variables params, Variables grads) = 0;
    virtual void Update(void) = 0;
};


}
