#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_ReLU_Forward(
            const float*    x_buf,
            float*          y_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;
    int node  = blockDim.y * blockIdx.y + threadIdx.y;

    if (frame >= frame_size || node >= node_size) {
        return;
    }

    float x = x_buf[frame_stride*node + frame];
    if ( x <= 0 ) { x = 0; }
    y_buf[frame_stride*node + frame] = x;
}


BBCU_DLL_EXPORT int bbcu_fp32_ReLU_Forward
        (
            const float*    dev_x_buf,
            float*          dev_y_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block;
    dim3    grid;

    block.x = std::min(frame_size, 1024);
    block.y = std::min(node_size, 1024);
    while (block.y > 1 && block.x * block.y > 1024) {
        block.y = (block.y + 1) / 2;
    }
    while (block.x > 1 && block.x * block.y > 1024) {
        block.x = (block.x + 1) / 2;
    }
    grid.x = (frame_size + (block.x - 1)) /  block.x;
    grid.y = (node_size  + (block.y - 1)) /  block.y;
    
    kernal_fp32_ReLU_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_ReLU_Backward
        (
            const float*    x_buf,
            const float*    dy_buf,
            float*          dx_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;
    int node  = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (frame >= frame_size || node >= node_size) {
        return;
    }

    float x  = x_buf[frame_stride*node + frame];
    float dy = dy_buf[frame_stride*node + frame];
    if ( x <= 0 ) { dy = 0; }
    dx_buf[frame_stride*node + frame] = dy;
}


BBCU_DLL_EXPORT int bbcu_fp32_ReLU_Backward
        (
            const float*    dev_x_buf,
            const float*    dev_dy_buf,
            float*          dev_dx_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block;
    dim3    grid;

    block.x = std::min(frame_size, 1024);
    block.y = std::min(node_size, 1024);
    while (block.y > 1 && block.x * block.y > 1024) {
        block.y = (block.y + 1) / 2;
    }
    while (block.x > 1 && block.x * block.y > 1024) {
        block.x = (block.x + 1) / 2;
    }
    grid.x = (frame_size + (block.x - 1)) /  block.x;
    grid.y = (node_size  + (block.y - 1)) /  block.y;
    
    kernal_fp32_ReLU_Backward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

// end of file
