#include <algorithm>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#include "Common.cuh"


__global__ void kernal_fp32_LossMeanSquaredError(
            const float*    y_buf,
            const float*    t_buf,
            float*          dy_buf,
            double*         loss_buf,
            float           reduction,
            double          reciprocal_node_size,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int node_id   = blockIdx.y * blockDim.y + threadIdx.y;
    int node_step = gridDim.y * blockDim.y; 
    int id        = threadIdx.x;
    int id_step   = blockDim.x;
    int blk_id    = threadIdx.y;

    __shared__  double  sbuf[32][32];

    double loss = 0.0;
    for (int node = node_id; node < node_size; node += node_step) {
        for (int frame = id; frame < frame_size; frame += id_step) {
            float y     = y_buf[node * frame_stride + frame];
            float t     = t_buf[node * frame_stride + frame];
            float dy    = y - t;
            dy_buf[node * frame_stride + frame] = dy * reduction;
            loss += (double)(dy * dy) * reciprocal_node_size;
        }
    }

    loss = device_LocalSumX<double>(loss, sbuf[blk_id]);
    
    if ( id == 0 ) {
        loss_buf[node_id] += loss;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_LossMeanSquaredError
        (
            const float*    dev_y_buf,
            const float*    dev_t_buf,
            float*          dev_dy_buf,
            double*         dev_loss_buf,
            int             loss_buf_size,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            float           grad_reduction,
            double          loss_reduction,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    // 計算
    dim3    block(32, 32);
    dim3    grid(1, 32);
    grid.y = std::min((int)grid.y, (int)((node_size + 31)/32));
    kernal_fp32_LossMeanSquaredError<<<grid, block, 0, streamId>>>(
            dev_y_buf,
            dev_t_buf,
            dev_dy_buf,
            dev_loss_buf,
            grad_reduction,
            loss_reduction,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}


// end of file
