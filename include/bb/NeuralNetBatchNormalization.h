// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#if 0

#include "bb/NeuralNetBatchNormalizationEigen.h"

namespace bb {

template <typename T = float>
using NeuralNetBatchNormalization = NeuralNetBatchNormalizationEigen<T>;

}

#else

#include "bb/NeuralNetBatchNormalizationAvx.h"

namespace bb {

template <typename T = float>
using NeuralNetBatchNormalization = NeuralNetBatchNormalizationAvx<T>;

}

#endif

