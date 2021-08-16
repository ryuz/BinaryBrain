#include <algorithm>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Common.cuh"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



template<typename T>
__global__ void kernal_LossSoftmaxCrossEntropy(
            T   const   *y_buf,
            T   const   *t_buf,
            T           *dy_buf,
            double      *loss_buf,
            T           reciprocal_t_sum,
            int         pix_size,
            int         ch_size,
            int         frame_size,
            int         frame_stride
        )
{
    const   T   eps   = (T)1.0e-7;

    int frame = blockDim.x * blockIdx.x + threadIdx.x;

    if ( frame < frame_size ) {
        double loss = 0;

        for ( int pix = 0; pix < pix_size; ++pix ) {
            // max
            T   c = bb_type_lowest<T>();
            for ( int ch = 0; ch < ch_size; ++ch ) {
                int node = ch * pix_size + pix;
                c = max(c, y_buf[node * frame_stride + frame]);
            }

    //      if ( isnan(c) ) {
    //          c = 0;
    //      }
        
            // sum(exp(y - c))
            T y_sum = 0;
            T t_max = 0;
            for ( int ch = 0; ch < ch_size; ++ch ) {
                int node = ch * pix_size + pix;
                y_sum += exp(y_buf[node * frame_stride + frame] - c);
                t_max += t_buf[node * frame_stride + frame];    // ワンホットなので足していけばそのチャネルのWeightが得られる
            }
        
            // 0以下での除算回避
            y_sum = max(eps, y_sum);

            // loass
            for ( int ch = 0; ch < ch_size; ++ch ) {
                int node = ch * pix_size + pix;
                T   y = y_buf[node * frame_stride + frame];
                T   t = t_buf[node * frame_stride + frame];
                T   softmax = std::exp(y - c) / y_sum;
                if ( t > 0 ) {
                    loss += log(softmax + eps) * t_max;
                }

                T   dy = (t_max * softmax - t) * reciprocal_t_sum;
                dy_buf[node * frame_stride + frame] = dy;
            }
        }
        loss_buf[frame] = loss;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_LossSoftmaxCrossEntropy
        (
            T   const       *dev_y_buf,
            T   const       *dev_t_buf,
            T               *dev_dy_buf,
            double          *dev_loss_buf,
            T               t_sum,
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
    kernal_LossSoftmaxCrossEntropy<T><<<grid, block, 0, streamId>>>(
            dev_y_buf,
            dev_t_buf,
            dev_dy_buf,
            dev_loss_buf,
            (T)1.0 / t_sum,
            pix_size,
            ch_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}

// 実体化
template BBCU_DLL_EXPORT int bbcu_LossSoftmaxCrossEntropy<double>(double const*, double const*, double*, double*, double, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_LossSoftmaxCrossEntropy<float> (float  const*, float  const*, float*,  double*, float,  int, int, int, int, cudaStream_t);





__global__ void kernal_fp32_LossSoftmaxCrossEntropy(
            const float*    y_buf,
            const float*    t_buf,
            float*          dy_buf,
            double*         loss_buf,
            float           reciprocal_frame_size,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;
    if (frame >= frame_size) {
        return;
    }

    // max探索
    float c = y_buf[frame];
    for ( int node = 1; node < node_size; ++node) {
        c = max(c, y_buf[node * frame_stride + frame]);
    }

    // sum(exp(y - c))
    float sum = 0;
    for ( int node = 0; node < node_size; ++node) {
        sum += std::exp(y_buf[node * frame_stride + frame] - c);
    }

    sum = max(sum, 1.0e-7f);

    float loss_softmax;
    for ( int node = 0; node < node_size; ++node) {
        float y = y_buf[node * frame_stride + frame];
        float t = t_buf[node * frame_stride + frame];

        float softmax = exp(y - c) / sum;
        float dy = (softmax - t) * reciprocal_frame_size;
        dy_buf[node * frame_stride + frame] = dy;

        if ( t > 0 ) {
            loss_softmax = softmax;
        }
    }
    loss_buf[frame] = log((double)loss_softmax + 1.0e-7);
}

__global__ void kernal_fp32_LossSoftmaxCrossEntropy_Sum(
            double*         loss_buf,
            double*         loss,
            int             frame_size
        )
{
    double sum = 0;
    for ( int frame = 0; frame < frame_size; ++frame) {
        sum += loss_buf[frame];
    }
    *loss += -sum;
}


BBCU_DLL_EXPORT int bbcu_fp32_LossSoftmaxCrossEntropy
        (
            const float*    dev_y_buf,
            const float*    dev_t_buf,
            float*          dev_dy_buf,
            double*         dev_loss_buf,
            double*         dev_loss,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             batch_size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());
    
    // 計算
    dim3    block(512);
    dim3    grid((frame_size + (block.x-1)) / block.x);
    block.x = std::min((int)block.x, (int)frame_size);
    kernal_fp32_LossSoftmaxCrossEntropy<<<grid, block, 0, streamId>>>(
            dev_y_buf,
            dev_t_buf,
            dev_dy_buf,
            dev_loss_buf,
            1.0f / (float)batch_size,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    // 損失集計
    kernal_fp32_LossSoftmaxCrossEntropy_Sum << <1, 1, 0, streamId >> > (
            dev_loss_buf,
            dev_loss,
            frame_size
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}


// end of file
