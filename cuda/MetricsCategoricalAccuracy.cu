#include <algorithm>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#include "Common.cuh"


template<typename T>
__global__ void kernal_MetricsCategoricalAccuracy(
            T const *y_buf,
            T const *t_buf,
            int     *accuracy_buf,
            int     *category_buf,
            int     pix_size,
            int     ch_size,
            int     frame_size,
            int     frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;

    if ( frame < frame_size ) {
        int accuracy = 0;
        int category = 0;

        for ( int pix = 0; pix < pix_size; ++pix ) {
            T       max_y  = bb_type_lowest<T>();
            T       max_t  = bb_type_lowest<T>();
            bool    valid  = false;
            for (int ch = 0; ch < ch_size; ++ch) {
                int node = ch * pix_size + pix;
                T   y = y_buf[node * frame_stride + frame];
                T   t = t_buf[node * frame_stride + frame];
                if (t > 0) {
                    valid = true;
                }
                if ( y > max_y ) {
                    max_y  = y;
                    max_t  = t;
                }
            }

            if ( valid ) {
                category += 1;
                if ( max_t > 0 ) {
                    accuracy += 1;
                }
            }
        }

        accuracy_buf[frame] = accuracy;
        category_buf[frame] = category;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_MetricsCategoricalAccuracy
        (
            T   const       *dev_y_buf,
            T   const       *dev_t_buf,
            int             *dev_accuracy_buf,
            int             *dev_category_buf,
            int             pix_size,
            int             ch_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());
    
    // 計算
    dim3    block(512);
    dim3    grid((frame_size + (block.x-1)) / block.x);
    block.x = std::min((int)block.x, (int)frame_size);
    kernal_MetricsCategoricalAccuracy<T><<<grid, block, 0, streamId>>>(
                 dev_y_buf,
                 dev_t_buf,
                 dev_accuracy_buf,
                 dev_category_buf,
                 pix_size,
                 ch_size,
                 frame_size,
                 frame_stride
            );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}

// 実体化
template BBCU_DLL_EXPORT int bbcu_MetricsCategoricalAccuracy<double>(double const*, double const*, int*, int*, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_MetricsCategoricalAccuracy<float> (float  const*, float  const*, int*, int*, int, int, int, int, cudaStream_t);



#if 0


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

    int id      = threadIdx.x;
    int id_step = blockDim.x;
    
    int acc_sum = 0;
    
    for ( int frame = id; frame < frame_size; frame += id_step ) {
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
    if ( id == 0 ) {
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
            buf[id] += buf[id + comb];
        }
        comb = next;
        __syncthreads();
    }

    if ( id == 0 ) {
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
    while ((int)block.x / 2 < frame_size) { block.x /= 2; }

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

#endif


// end of file
