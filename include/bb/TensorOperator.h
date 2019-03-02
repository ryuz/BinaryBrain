// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <iostream>
#include <array>
#include <vector>
#include <memory>


namespace bb {


// -------------------------------------
//  基本演算定義
// -------------------------------------

template<typename T>
inline void Tensor_Vector_set
(
    T       *dst,
    T	    a,
    index_t size
)
{
#pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = a;
    }
}


template<typename T>
inline void Tensor_Vector_add_ex
(
    T       *dst,
    T const *src0,
    T const *src1,
    T	    a,
    T	    b,
    T	    c,
    index_t size
)
{
#pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = a * src0[i] + b * src1[i] + c;
    }
}


template<typename T>
inline void Tensor_Vector_sub_ex
(
    T       *dst,
    T const *src0,
    T const *src1,
    T	    a,
    T	    b,
    T	    c,
    index_t size
)
{
#pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = a * src0[i] - b * src1[i] - c;
    }
}


template<typename T>
inline void Tensor_Vector_mul_ex
(
    T       *dst,
    T const *src0,
    T const *src1,
    T	    a,
    T	    b,
    index_t size
)
{
    #pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = a * src0[i] * src1[i] + b;
    }
}

template<typename T>
inline void Tensor_Vector_div_ex
(
    T       *dst,
    T const *src0,
    T const *src1,
    T	    a,
    T	    b,
    T	    c,
    T	    d,
    index_t size
)
{
    #pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = (a * src0[i] + b) / (c * src1[i] + d);
    }
}


template<typename T>
inline void Tensor_Vector_sqrt(
    T       *dst,
    T const *src,
    index_t size
)
{
    #pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = (T)sqrt((double)src[i]);
    }
}

template<>
inline void Tensor_Vector_sqrt<float>(
    float       *dst,
    float const *src,
    index_t size
)
{
    #pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = sqrt(src[i]);
    }
}


template<typename T>
inline void Tensor_Vector_exp(
    T       *dst,
    T const *src,
    index_t size
)
{
    #pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = (T)exp((double)src[i]);
    }
}

template<>
inline void Tensor_Vector_exp<float>(
    float       *dst,
    float const *src,
    index_t size
)
{
    #pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = exp(src[i]);
    }
}


}

//endof file

