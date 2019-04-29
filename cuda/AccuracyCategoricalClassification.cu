#include <algorithm>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


#define THREAD_UNIT     1024


__global__ void kernal_fp32_AccuracyCategoricalClassification(
            float const *y_buf,
            float const *t_buf,
            int         *accuracy,
            int         node_size,
            int         frame_size,
            int         frame_stride
        )
{
    __shared__   int    buf[THREAD_UNIT];

    int frame_base = threadIdx.x;
    int frame_step = blockDim.x;
    
    int acc_sum = 0;
    
    for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
        // max探索
        float max_val = y_buf[frame];
        int   max_idx = 0;
        for ( int node = 1; node < node_size; ++node) {
            float val = y_buf[node * frame_stride + frame];
            if ( val > max_val ) {
                max_val = val;
                max_idx = node;
            }
        }
        
        if ( t_buf[max_idx * frame_stride + frame] > 0 ) {
            acc_sum++;
        }
    }

    int prev_accuracy;
    if ( threadIdx.x == 0 ) {
        prev_accuracy = accuracy[0];
    }

    // スレッド間集計
    buf[threadIdx.x] = acc_sum;
    __syncthreads();

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

    if ( threadIdx.x == 0 ) {
        accuracy[0] = prev_accuracy + buf[0];
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_AccuracyCategoricalClassification
        (
            const float*    dev_y_buf,
            const float*    dev_t_buf,
            int*            dev_accuracy,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());
    
    // 計算
    dim3    block(THREAD_UNIT);
    dim3    grid(1);
    block.x = std::min((int)block.x, (int)frame_size);
    kernal_fp32_AccuracyCategoricalClassification<<<grid, block, 0, streamId>>>(
            dev_y_buf,
            dev_t_buf,
            dev_accuracy,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}


// end of file
