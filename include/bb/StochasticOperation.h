// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <array>
#include <algorithm>

#include "bb/DataType.h"


namespace bb {


template <typename T=float>
inline void StochasticOperation_And_Forward
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
void StochasticOperation_And_Backward
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
inline void StochasticOperation_Or_Forward
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
void StochasticOperation_Or_Backward
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


template <typename T=float>
inline void StochasticOperation_Lut_Forward
        (
            T const *x,
            T       *y,
            T const *W,
            int     N
        )
{
    T acc = (T)0;
    for (int i = 0; i < (1 << N); ++i) {
        T w = W[i];
        for (int j = 0; j < N; ++j) {
            w *= ((i >> j) & 1) ? x[j] : ((T)1.0 - x[j]);
        }
        acc += w;
    }

    *y = acc;
}



template <typename T=float>
inline void StochasticOperation_Lut_Backward
        (
            T const *x,
            T       *dx,
            T const *dy,
            T const *W,
            T       *dW,
            int     N
        )
{
    // calcurate dW
    for (int i = 0; i < (1 << N); ++i) {
        T dw = *dy;
        for (int j = 0; j < N; ++j) {
            dw *= ((i >> j) & 1) ? x[j] : ((T)1.0 - x[j]);
        }
        dW[i] += dw;
    }

    // calcurate dx
    for (int i = 0; i < N; ++i) {
        dx[i] = 0;
        for (int j = 0; j < (1 << N); ++j) {
            T w = ((j >> i) & 1) ? +W[j] : -W[j];
            for (int k = 0; k < N; ++k) {
                if (i != k) {
                    w *= ((j >> k) & 1) ? x[k] : ((T)1.0 - x[k]);
                }
            }
            dx[i] += w;
        }
        dx[i] *= *dy;
    }
}

}


// end of file
