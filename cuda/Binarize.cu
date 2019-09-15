#include <iostream>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#include "Common.cuh"



//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_Binarize_Forward(
            float const *x_buf,
            float       *y_buf,
            float       binary_th,
            int         node_size,
            int         frame_size,
            int         frame_stride
        )
{
    int const node    = blockIdx.y * blockDim.y + threadIdx.y;
    int const id      = threadIdx.x;
    int const id_step = blockDim.x;
    
    if ( node < node_size ) {
        for ( int frame = id; frame < frame_size; frame += id_step ) {
            float x = x_buf[frame_stride*node + frame];
            x = (x > binary_th) ? 1 : 0;
            y_buf[frame_stride*node + frame] = x;
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Binarize_Forward
        (
            float const *   dev_x_buf,
            float*          dev_y_buf,
            float           binary_th,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 1024;
    unsigned int const MAX_FRAME_UNIT = 1024;
    unsigned int const MAX_NODE_UNIT  = 1024;

#if 1
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size )  { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size)  { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (node_size + (block.y - 1)) / block.y);

    kernal_fp32_Binarize_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            binary_th,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


////////////////

#if 0
__device__ __forceinline__ int device_int_LocalOr(int v, int id, int *sbuf)
{
    sbuf[id] = v;
    __syncthreads();

    // スレッド間集計
    int comb = 1;
    while (comb < 32) {
        int next = comb * 2;
        int mask = next - 1;
        if ((id & mask) == 0) {
            sbuf[id] |= sbuf[id + comb];
        }
        comb = next;
        __syncthreads();
    }

    int sum = sbuf[0];
    __syncthreads();
    
    return sum;
}
#endif


__global__ void kernal_fp32_bit_Binarize_Forward(
            float const *x_buf,
            int         *y_buf,
            float       binary_th,
            int         node_size,
            int         frame_size,
            int         x_frame_stride,
            int         y_frame_stride
        )
{
    int const frame = blockIdx.x * blockDim.x + threadIdx.x;
    int const node  = blockIdx.y * blockDim.y + threadIdx.y;
    int const id    = threadIdx.y * blockDim.x + threadIdx.x;
    
    __shared__  int sbuf[32][32];

    float const *x_ptr = &x_buf[x_frame_stride * node];
    int         *y_ptr = &y_buf[y_frame_stride * node];

    int bit  = (frame & 0x1f);
    int unit = ((id >> 5) & 0x1f);

    int mask = 0;
    if ( node < node_size && frame < frame_size ) {
        float x = x_ptr[frame];
        mask = (x > binary_th) ? (1 << bit) : 0;
    }

    // 統合
    mask = device_int_LocalOr(mask, bit, sbuf[unit]);

    // 書き出し
    if ( node < node_size && frame < frame_size ) {
        if ( bit == 0 ) {
            y_ptr[frame >> 5] = mask;
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_bit_Binarize_Forward
        (
            float const     *dev_x_buf,
            int             *dev_y_buf,
            float           binary_th,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 1024;
    unsigned int const MAX_FRAME_UNIT = 1024;
    unsigned int const MAX_NODE_UNIT  = 1024;

#if 1
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size && block.x > 32 ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size )                  { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size)                   { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size && block.x > 32 ) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid((frame_size + (block.x - 1)) / block.x , (node_size + (block.y - 1)) / block.y);

    kernal_fp32_bit_Binarize_Forward<<<grid, block, 0, streamId>>>
        (
            dev_x_buf,
            dev_y_buf,
            binary_th,
            node_size,
            frame_size,
            x_frame_stride,
            y_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

