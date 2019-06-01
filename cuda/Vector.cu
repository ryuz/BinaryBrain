#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



// -------------------------------------------------
//  set
// -------------------------------------------------

__global__ void kernal_fp32_Vector_set
        (
            float*          dst,
            float           a,
            int             size
        )
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = a;
        index += blockDim.x;
    }
}

BBCU_DLL_EXPORT int bbcu_fp32_Vector_set(
            float*          dev_dst,
            float           a,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_set<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            a,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}




// -------------------------------------------------
//  add_ex
// -------------------------------------------------

__global__ void kernal_fp32_Vector_add_ex
        (
            float*          dst,
            const float*    src0,
            const float*    src1,
            float           a,
            float           b,
            float           c,
            int             size
        )
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = a * src0[index] + b * src1[index] + c;
        index += blockDim.x;
    }
}

BBCU_DLL_EXPORT int bbcu_fp32_Vector_add_ex(
            float*          dev_dst,
            const float*    dev_src0,
            const float*    dev_src1,
            float           a,
            float           b,
            float           c,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_add_ex<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src0,
            dev_src1,
            a,
            b,
            c,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



// -------------------------------------------------
//  mul_ex
// -------------------------------------------------

__global__ void kernal_fp32_Vector_mul_ex(
            float*          dst,
            const float*    src0,
            const float*    src1,
            float           a,
            float           b,
            int             size)
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = a * src0[index] * src1[index] + b;
        index += blockDim.x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Vector_mul_ex
        (
            float*          dev_dst,
            const float*    dev_src0,
            const float*    dev_src1,
            float           a,
            float           b,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_mul_ex<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src0,
            dev_src1,
            a,
            b,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// -------------------------------------------------
//  div_ex
// -------------------------------------------------

__global__ void kernal_fp32_Vector_div_ex(
            float*          dst,
            const float*    src0,
            const float*    src1,
            float           a,
            float           b,
            float           c,
            float           d,
            int             size)
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = (a * src0[index] + b) / (c * src1[index] + d);
        index += blockDim.x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Vector_div_ex(
            float           *dev_dst,
            float const     *dev_src0,
            float const     *dev_src1,
            float           a,
            float           b,
            float           c,
            float           d,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_div_ex<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src0,
            dev_src1,
            a,
            b,
            c,
            d,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



// -------------------------------------------------
//  sqrt
// -------------------------------------------------

__global__ void kernal_fp32_Vector_sqrt(
            float*          dst,
            const float*    src,
            int             size)
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = sqrt(src[index]);
        index += blockDim.x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Vector_sqrt(
            float           *dev_dst,
            float const     *dev_src,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_sqrt<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}




// -------------------------------------------------
//  exp
// -------------------------------------------------

__global__ void kernal_fp32_Vector_exp(
            float*          dst,
            const float*    src,
            int             size)
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = exp(src[index]);
        index += blockDim.x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Vector_exp(
            float           *dev_dst,
            float const     *dev_src,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_exp<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



// -------------------------------------------------
//  min
// -------------------------------------------------

// ベクトル同士
__global__ void kernal_fp32_Vector_min(
            float*          dst,
            const float*    src0,
            const float*    src1,
            int             size)
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = min(src0[index], src1[index]);
        index += blockDim.x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Vector_min(
            float           *dev_dst,
            float const     *dev_src0,
            float const     *dev_src1,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_min<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src0,
            dev_src1,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// 係数
__global__ void kernal_fp32_Vector_min_v(
            float*          dst,
            const float*    src,
            float           a,
            int             size)
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = min(a, src[index]);
        index += blockDim.x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Vector_min_v(
            float           *dev_dst,
            float const     *dev_src,
            float           a,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_min_v<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src,
            a,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// -------------------------------------------------
//  max
// -------------------------------------------------


// ベクトル同士
__global__ void kernal_fp32_Vector_max(
            float*          dst,
            const float*    src0,
            const float*    src1,
            int             size)
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = max(src0[index], src1[index]);
        index += blockDim.x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Vector_max(
            float           *dev_dst,
            float const     *dev_src0,
            float const     *dev_src1,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_max<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src0,
            dev_src1,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// 係数
__global__ void kernal_fp32_Vector_max_v(
            float*          dst,
            const float*    src,
            float           a,
            int             size)
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = max(a, src[index]);
        index += blockDim.x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Vector_max_v(
            float           *dev_dst,
            float const     *dev_src,
            float           a,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_max_v<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src,
            a,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// -------------------------------------------------
//  clamp
// -------------------------------------------------

__global__ void kernal_fp32_Vector_clamp(
            float*          dst,
            const float*    src,
            float           lo,
            float           hi,
            int             size)
{
    int index = threadIdx.x;
    while ( index < size ) {
        dst[index] = max(lo, min(hi, src[index]));
        index += blockDim.x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Vector_clamp(
            float           *dev_dst,
            float const     *dev_src,
            float           lo,
            float           hi,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    kernal_fp32_Vector_clamp<<<1, 1024, 0, streamId>>>
        (
            dev_dst,
            dev_src,
            lo,
            hi,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}





// end of file
