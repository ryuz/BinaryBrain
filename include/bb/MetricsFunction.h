// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <string>
#include <vector>

#include "bb/FrameBuffer.h"


namespace bb {


class MetricsFunction
{
public:
    virtual ~MetricsFunction() {}
    
    virtual std::string GetMetricsString(void) { return "accuracy"; }
    
    virtual void        Clear(void) = 0;
    virtual double      GetMetrics(void) const = 0;
    virtual void        CalculateMetrics(FrameBuffer y, FrameBuffer t) = 0;
};


}

