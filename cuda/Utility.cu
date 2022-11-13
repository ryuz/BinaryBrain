#include <algorithm>
#include <limits>
#include <nppdefs.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#include "Common.cuh"



// ---------------------------------
//  IsNan
// ---------------------------------

template<typename T>
__global__ void kernal_Tensor_IsnNan
        (
            int             *result,
            const T         *buf,
            int             size
        )
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if ( index >= size ) {
        return;
    }

    if ( isnan(buf[index]) ) {
        result[0] = 1;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_Tensor_IsnNan
        (
            int             *dev_result,
            T   const       *dev_buf,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(int)));

    dim3    block(1024);
    dim3    grid((size+1023)/1024);
    
    kernal_Tensor_IsnNan<T><<<grid, block, 0, streamId>>>(
            dev_result,
            dev_buf,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_Tensor_IsnNan<float> (int *, float  const *, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_Tensor_IsnNan<double>(int *, double const *, int, cudaStream_t);



template<typename T>
__global__ void kernal_FrameBuf_IsnNan
        (
            int             *result,
            const T         *buf,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;
    int node  = blockDim.y * blockIdx.y + threadIdx.y;

    if (frame >= frame_size || node >= node_size) {
        return;
    }

    if ( isnan(buf[frame_stride*node + frame]) ) {
        *result = 1;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_FrameBuf_IsnNan
        (
            int             *dev_result,
            const T         *dev_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(int)));

    dim3    block;
    dim3    grid;

    block.x = std::min(frame_size, 1024);
    block.y = std::min(node_size, 1024);
    while (block.y > 1 && block.x * block.y > 1024) {
        block.y = (block.y + 1) / 2;
    }
    while (block.x > 1 && block.x * block.y > 1024) {
        block.x = (block.x + 1) / 2;
    }
    grid.x = (frame_size + (block.x - 1)) /  block.x;
    grid.y = (node_size  + (block.y - 1)) /  block.y;
    
    kernal_FrameBuf_IsnNan<T><<<grid, block, 0, streamId>>>(
            dev_result,
            dev_buf,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_FrameBuf_IsnNan<float> (int *, float  const *, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_FrameBuf_IsnNan<double>(int *, double const *, int, int, int, cudaStream_t);





// ---------------------------------
//  min
// ---------------------------------

template<typename T>
__global__ void kernal_Tensor_Min
        (
            T               *result,
            const T         *buf,
            int             size
        )
{
    int id   = threadIdx.x;
    int step = blockDim.x;

    T   value = bb_type_max<T>();
    for (int index = id; index < size; index += step) {
        value = min(value, buf[index]);
    }

    value = device_ShuffleMin(value);
    if (id == 0) {
        *result = value;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_Tensor_Min
        (
            T               *dev_result,
            T   const       *dev_buf,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

//  BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(T)));

    dim3    block(32);
    dim3    grid(1);
    
    kernal_Tensor_Min<T><<<grid, block, 0, streamId>>>(
            dev_result,
            dev_buf,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_Tensor_Min<float> (float  *, float  const *, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_Tensor_Min<double>(double *, double const *, int, cudaStream_t);



template<typename T>
__global__ void kernal_FrameBuf_Min
        (
            T               *result,
            T   const       *buf,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int id   = threadIdx.x;
    int step = blockDim.x;
    
    T   value = bb_type_max<T>();
    for (int node = 0; node < node_size; node++) {
        for (int frame = id; frame < frame_size; frame += step) {
            value = min(value, buf[node*frame_stride + frame]);
        }
    }

    value = device_ShuffleMin(value);
    if (id == 0) {
        *result = value;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_FrameBuf_Min
        (
            T               *dev_result,
            T   const       *dev_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

//  BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(int)));

    dim3    block(32);
    dim3    grid(1);
    kernal_FrameBuf_Min<T><<<grid, block, 0, streamId>>>(
            dev_result,
            dev_buf,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_FrameBuf_Min<float> (float  *, float  const *, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_FrameBuf_Min<double>(double *, double const *, int, int, int, cudaStream_t);





// ---------------------------------
//  max
// ---------------------------------

template<typename T>
__global__ void kernal_Tensor_Max
        (
            T               *result,
            const T         *buf,
            int             size
        )
{
    int id   = threadIdx.x;
    int step = blockDim.x;

    T   value = bb_type_lowest<T>();
    for (int index = id; index < size; index += step) {
        value = max(value, buf[index]);
    }

    value = device_ShuffleMax(value);
    if (id == 0) {
        *result = value;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_Tensor_Max
        (
            T               *dev_result,
            T   const       *dev_buf,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

//  BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(T)));

    dim3    block(32);
    dim3    grid(1);
    
    kernal_Tensor_Max<T><<<grid, block, 0, streamId>>>(
            dev_result,
            dev_buf,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_Tensor_Max<float> (float  *, float  const *, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_Tensor_Max<double>(double *, double const *, int, cudaStream_t);



template<typename T>
__global__ void kernal_FrameBuf_Max
        (
            T               *result,
            T   const       *buf,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int id   = threadIdx.x;
    int step = blockDim.x;
    
    T   value = bb_type_lowest<T>();
    for (int node = 0; node < node_size; node++) {
        for (int frame = id; frame < frame_size; frame += step) {
            value = max(value, buf[node*frame_stride + frame]);
        }
    }

    value = device_ShuffleMax(value);
    if (id == 0) {
        *result = value;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_FrameBuf_Max
        (
            T               *dev_result,
            T   const       *dev_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

//  BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(int)));

    dim3    block(32);
    dim3    grid(1);
    kernal_FrameBuf_Max<T><<<grid, block, 0, streamId>>>(
            dev_result,
            dev_buf,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_FrameBuf_Max<float> (float  *, float  const *, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_FrameBuf_Max<double>(double *, double const *, int, int, int, cudaStream_t);







// -------------------------------------------------
//  Quantize
// -------------------------------------------------


template<typename T>
__global__ void kernal_Tensor_Quantize
        (
            T               *dst,
            const T         *src,
            T               lo,
            T               hi,
            T               scale,
            T               scale_recip,
            int             size
        )
{
    int id   = threadIdx.x;
    int step = blockDim.x;

    for (int index = id; index < size; index += step) {
        T   real_val = src[index];
        real_val = max(real_val, lo);
        real_val = min(real_val, hi);
        int int_val = (int)(real_val * scale_recip + (T)0.5);
        real_val = (T)int_val * scale;
        dst[index] = real_val;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_Tensor_Quantize
        (
            T               *dev_result,
            T   const       *dev_buf,
            T               lo,
            T               hi,
            T               scale,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

//  BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(T)));

    dim3    block(32);
    dim3    grid(1);
    
    kernal_Tensor_Quantize<T><<<grid, block, 0, streamId>>>(
            dev_result,
            dev_buf,
            lo,
            hi,
            scale,
            (T)1 / scale,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_Tensor_Quantize<float> (float  *, float  const *, float,  float,  float,  int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_Tensor_Quantize<double>(double *, double const *, double, double, double, int, cudaStream_t);



template<typename T>
__global__ void kernal_FrameBuf_Quantize
        (
            T               *dst,
            T   const       *src,
            T               lo,
            T               hi,
            T               scale,
            T               scale_recip,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int id   = threadIdx.x;
    int step = blockDim.x;
    
    for (int node = 0; node < node_size; node++) {
        for (int frame = id; frame < frame_size; frame += step) {
            T   real_val = src[node*frame_stride + frame];;
            real_val = max(real_val, lo);
            real_val = min(real_val, hi);
            int int_val = (int)(real_val * scale_recip + (T)0.5);
            real_val = (T)int_val * scale;
            dst[node*frame_stride + frame] = real_val;
        }
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_FrameBuf_Quantize
        (
            T               *dev_dst,
            T   const       *dev_src,
            T               lo,
            T               hi,
            T               scale,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(32);
    dim3    grid(1);

    kernal_FrameBuf_Quantize<T><<<grid, block, 0, streamId>>>(
            dev_dst,
            dev_src,
            lo,
            hi,
            scale,
            (T)1 / scale,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_FrameBuf_Quantize<float> (float  *, float  const *, float,  float,  float, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_FrameBuf_Quantize<double>(double *, double const *, double, double, double,int, int, int, cudaStream_t);





// ---------------------------------
//  momnet
// ---------------------------------

template<typename T>
__global__ void kernal_Tensor_Moment
        (
            double          *result,
            const T         *buf,
            int             size
        )
{
    int id   = threadIdx.x;
    int step = blockDim.x;

    double  m0_sum = 0;
    double  m1_sum = 0;
    double  m2_sum = 0;
    double  m3_sum = 0;
    double  m4_sum = 0;
    T       min_val = bb_type_max<T>();
    T       max_val = bb_type_lowest<T>();

    for (int index = id; index < size; index += step) {
        T   v = buf[index];
        if ( ! isnan(v) ) {
            min_val = min(min_val, v);
            max_val = max(max_val, v);

            double  m0 = 1.0;
            double  m1 = (double)v;
            double  m2 = m1*m1;
            double  m3 = m2*m1;
            double  m4 = m2*m2;
            m0_sum += m0;
            m1_sum += m1;
            m2_sum += m2;
            m3_sum += m3;
            m4_sum += m4;
        }
    }

    m0_sum  = device_ShuffleSum(m0_sum);
    m1_sum  = device_ShuffleSum(m1_sum);
    m2_sum  = device_ShuffleSum(m2_sum);
    m3_sum  = device_ShuffleSum(m3_sum);
    m4_sum  = device_ShuffleSum(m4_sum);
    min_val = device_ShuffleMin(min_val);
    max_val = device_ShuffleMax(max_val);

    if (id == 0) {
        result[0] = m0_sum;
        result[1] = m1_sum;
        result[2] = m2_sum;
        result[3] = m3_sum;
        result[4] = m4_sum;
        result[5] = (double)min_val;
        result[6] = (double)max_val;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_Tensor_Moment
        (
            double          *dev_result,
            T   const       *dev_buf,
            int             size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(32);
    dim3    grid(1);
    
    kernal_Tensor_Moment<T><<<grid, block, 0, streamId>>>(
            dev_result,
            dev_buf,
            size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_Tensor_Moment<float> (double *, float  const *, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_Tensor_Moment<double>(double *, double const *, int, cudaStream_t);



template<typename T>
__global__ void kernal_FrameBuf_Moment
        (
            double          *result,
            T   const       *buf,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int id   = threadIdx.x;
    int step = blockDim.x;

    double  m0_sum = 0;
    double  m1_sum = 0;
    double  m2_sum = 0;
    double  m3_sum = 0;
    double  m4_sum = 0;
    T       min_val = bb_type_max<T>();
    T       max_val = bb_type_lowest<T>();

    for (int node = 0; node < node_size; node++) {
        for (int frame = id; frame < frame_size; frame += step) {
            T   v = buf[node*frame_stride + frame];
            if ( ! isnan(v) ) {
                min_val = min(min_val, v);
                max_val = max(max_val, v);

                double  m0 = 1.0;
                double  m1 = (double)v;
                double  m2 = m1*m1;
                double  m3 = m2*m1;
                double  m4 = m2*m2;
                m0_sum += m0;
                m1_sum += m1;
                m2_sum += m2;
                m3_sum += m3;
                m4_sum += m4;
            }
        }
    }

    m0_sum  = device_ShuffleSum(m0_sum);
    m1_sum  = device_ShuffleSum(m1_sum);
    m2_sum  = device_ShuffleSum(m2_sum);
    m3_sum  = device_ShuffleSum(m3_sum);
    m4_sum  = device_ShuffleSum(m4_sum);
    min_val = device_ShuffleMin(min_val);
    max_val = device_ShuffleMax(max_val);

    if (id == 0) {
        result[0] = m0_sum;
        result[1] = m1_sum;
        result[2] = m2_sum;
        result[3] = m3_sum;
        result[4] = m4_sum;
        result[5] = (double)min_val;
        result[6] = (double)max_val;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_FrameBuf_Moment
        (
            double          *dev_result,
            T   const       *dev_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

//  BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(int)));

    dim3    block(32);
    dim3    grid(1);
    kernal_FrameBuf_Moment<T><<<grid, block, 0, streamId>>>(
            dev_result,
            dev_buf,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

template BBCU_DLL_EXPORT int bbcu_FrameBuf_Moment<float> (double *, float  const *, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_FrameBuf_Moment<double>(double *, double const *, int, int, int, cudaStream_t);





// end of file
