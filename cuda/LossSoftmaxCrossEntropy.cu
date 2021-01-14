#include <algorithm>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"





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
    float sum = 0;
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
