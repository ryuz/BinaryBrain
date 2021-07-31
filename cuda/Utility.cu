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
        *result = true;
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

    T   value = NPP_MAXABS_32F; // std::numeric_limits<T>::max();
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

    BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(T)));

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
    
    T   value = NPP_MAXABS_32F;
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

    BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(int)));

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

    T   value = -NPP_MAXABS_32F;
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

    BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(T)));

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
    
    T   value = -NPP_MAXABS_32F;
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

    BB_CUDA_SAFE_CALL(cudaMemset(dev_result, 0, sizeof(int)));

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



// end of file
