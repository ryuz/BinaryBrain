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



// end of file
