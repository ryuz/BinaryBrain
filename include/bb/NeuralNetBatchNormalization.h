// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#if 0

#include "NeuralNetBatchNormalizationEigen.h"

namespace bb {

template <typename T = float, typename INDEX = size_t>
using NeuralNetBatchNormalization = NeuralNetBatchNormalizationEigen<T, INDEX>;

}

#else

#include "NeuralNetBatchNormalizationAvx.h"

namespace bb {

template <typename T = float, typename INDEX = size_t>
using NeuralNetBatchNormalization = NeuralNetBatchNormalizationAvx<T, INDEX>;

}

#endif

