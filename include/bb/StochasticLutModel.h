// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/SparseModel.h"


namespace bb {


// 確率的LUT関連の基底クラス
class StochasticLutModel : public SparseModel
{
public:
    virtual Tensor       &W(void)        = 0;
    virtual Tensor const &W(void) const  = 0;
    
    virtual Tensor       &dW(void)       = 0;
    virtual Tensor const &dW(void) const = 0;
};


}
