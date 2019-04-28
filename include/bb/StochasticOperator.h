// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>
#include <random>

#include "bb/Model.h"
#include "bb/StochasticLut2.h"
#include "bb/StochasticLut4.h"
#include "bb/StochasticLut6.h"
#include "bb/BatchNormalization.h"


namespace bb {


template <typename T=float>
inline void StochasticAnd_Forward
        (
            T const *x,
            T       *y,
            index_t size
        )
{
    T v = (T)1.0;
    for (index_t i = 0; i < size; ++i) {
        v *= x[i];
    }
    *y = v;
}

template <typename T=float>
void StochasticAnd_Backward
        (
            T const *x,
            T const *dy,
            T       *dx,
            index_t size
        )
{
    for (index_t i = 0; i < size; ++i) {
        T v = *dy;
        for (index_t j = 0; j < size; ++j) {
            if ( i != j ) {
                v *= x[j];
            }
        }
        dx[i] = v;
    }
}


template <typename T=float>
inline void StochasticOr_Forward
        (
            T const *x,
            T       *y,
            index_t size
        )
{
    T v = (T)1.0;
    for (index_t i = 0; i < size; ++i) {
        v *= (T)1.0 - x[i];
    }
    *y = (T)1.0 - v;
}

template <typename T=float>
void StochasticOr_Backward
        (
            T const *x,
            T const *dy,
            T       *dx,
            index_t size
        )
{
    for (index_t i = 0; i < size; ++i) {
        T v = (T)1.0 - *dy;
        for (index_t j = 0; j < size; ++j) {
            if ( i != j ) {
                v *= (T)1.0 - x[j];
            }
        }
        dx[i] = (T)1.0 - v;
    }
}


}


// end of file
