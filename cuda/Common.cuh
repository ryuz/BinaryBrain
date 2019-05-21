
#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


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
    v = (v | __shfl_xor(v, 1));
    v = (v | __shfl_xor(v, 2));
    v = (v | __shfl_xor(v, 4));
    v = (v | __shfl_xor(v, 8));
    v = (v | __shfl_xor(v, 16));    
    return v;
}




// end of file
