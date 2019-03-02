// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                 Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                 https://github.com/ryuz
//                                 ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <vector>


#include "bb/Model.h"


namespace bb {


//! Layer class
template<typename FT=float, typename BT=float>
class Layer : public Model
{
public:
    // ノード単位でのForward計算
    virtual std::vector<FT> ForwardNode(index_t node, std::vector<FT> x) const { return x; }
//  virtual std::vector<BT> BackwardNode(index_t node, std::vector<BT> dy) const { return dy; }
};


}