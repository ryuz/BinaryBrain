#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



// kernel
__global__ void kernel_fp32_MatrixRowwiseSetVector(
            const float*    x_vec,
            float*          y_mat,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    // 初期化
    int frame_base = threadIdx.x;
    int frame_step = blockDim.x;
    int node       = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (node >= node_size) {
        return;
    }

    // 読み込み
    float x = x_vec[node];
    
    float *y_ptr = &y_mat[node * frame_stride];
    for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
        y_ptr[frame] = x;
    }
}


int bbcu_fp32_MatrixRowwiseSetVector
        (
            const float*    dev_x_vec,
            float*          dev_y_mat,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());
        
    dim3    block(32, 32);
    dim3    grid(1, (node_size+block.y-1)/block.y);
    
    kernel_fp32_MatrixRowwiseSetVector<<<grid, block, 0, streamId>>>(
            dev_x_vec,
            dev_y_mat,
            node_size,
            frame_size,
            frame_stride);
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}


