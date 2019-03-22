// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>

#include "bb/FrameBuffer.h"


namespace bb {


class AccuracyFunction
{
public:
	virtual ~AccuracyFunction() {}
	
    virtual void   Clear(void) = 0;
    virtual double GetAccuracy(void) const = 0;
	virtual void   CalculateAccuracy(FrameBuffer y, FrameBuffer t) = 0;
};


}

