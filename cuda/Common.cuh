
#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



template<typename T=float>
__device__ __forceinline__ T device_LocalSumX(T v, T *sbuf)
{
    sbuf[threadIdx.x] = v;
    __syncthreads();

    // スレッド間集計
    int comb = 1;
    while (comb < blockDim.x) {
        int next = comb * 2;
        int mask = next - 1;
        if ((threadIdx.x & mask) == 0) {
            sbuf[threadIdx.x] += sbuf[threadIdx.x + comb];
        }
        comb = next;
        __syncthreads();
    }

    T sum = sbuf[0];
    __syncthreads();
    
    return sum;
}


__device__ __forceinline__ float device_fp32_LocalSum(float v, float *buf)
{
    buf[threadIdx.x] = v;
    __syncthreads();

    // スレッド間集計
    int comb = 1;
    while (comb < blockDim.x) {
        int next = comb * 2;
        int mask = next - 1;
        if ((threadIdx.x & mask) == 0) {
            buf[threadIdx.x] += buf[threadIdx.x + comb];
        }
        comb = next;
        __syncthreads();
    }

    float sum = buf[0];
    __syncthreads();
    
    return sum;
}



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


__device__ __forceinline__ int device_int_ShuffleOr(int v)
{
    v |= __shfl_xor_sync(0xffffffff, v,  1, 32);
    v |= __shfl_xor_sync(0xffffffff, v,  2, 32);
    v |= __shfl_xor_sync(0xffffffff, v,  4, 32);
    v |= __shfl_xor_sync(0xffffffff, v,  8, 32);
    v |= __shfl_xor_sync(0xffffffff, v, 16, 32);    
    return v;
}


template<typename T = float>
__device__ __forceinline__ T device_ShuffleSum(T v)
{
    v += __shfl_xor_sync(0xffffffff, v,  1, 32);
    v += __shfl_xor_sync(0xffffffff, v,  2, 32);
    v += __shfl_xor_sync(0xffffffff, v,  4, 32);
    v += __shfl_xor_sync(0xffffffff, v,  8, 32);
    v += __shfl_xor_sync(0xffffffff, v, 16, 32);    
    return v;
}



template<int N=6, typename T = float>
__global__ void kernal_NodeIntegrate(
            T   const   *src_buf,
            T           *dst_buf,
            int const   *input_index,
            int         node_size,
            int         frame_size,
            int         src_frame_stride,
            int         dst_frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;

    for ( int node = 0; node < node_size; ++node ) {
        if ( frame < frame_size ) {
            for ( int n = 0; n < N; ++n ) {
                int in_idx = input_index[node*N + n];
                T       *dst_buf_ptr = &dst_buf[in_idx * dst_frame_stride];
                T       prev_data    = dst_buf_ptr[frame];
                T const *src_buf_ptr = &src_buf[(N * node + n) * src_frame_stride];
                
                dst_buf_ptr[frame] = prev_data + src_buf_ptr[frame];
            }
        }
    }
}


template<typename T = float>
__global__ void kernal_NodeIntegrateWithTable(
            T   const   *src_buf,
            T           *dst_buf,
            int const   *index_table,
            int         table_stride,
            int         node_size,
            int         frame_size,
            int         src_frame_stride,
            int         dst_frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;
    int node  = blockDim.y * blockIdx.y + threadIdx.y;

    if ( frame < frame_size && node < node_size ) {
        T   const *src_ptr   = &src_buf[frame];
        int const *index_ptr = &index_table[table_stride * node];
        
        int size = index_ptr[0];
        T   sum  = 0;
        for ( int i = 1; i <= size; ++i ) {
            int index = index_ptr[i];
            sum += src_ptr[index * src_frame_stride];
        }

        dst_buf[node * dst_frame_stride + frame] = sum;
    }
}



// end of file
