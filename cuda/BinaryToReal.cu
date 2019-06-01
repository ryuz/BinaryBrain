#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_BinaryToReal_Forward(
            const float*    x_buf,
            float*          y_buf,
            float           gain,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride
        )
{
    int y_frame = blockDim.x * blockIdx.x + threadIdx.x;
    int y_node  = blockDim.y * blockIdx.y + threadIdx.y;

    if (y_frame >= y_frame_size || y_node >= y_node_size) {
        return;
    }

    float sum = 0;
    int x_frame = y_frame * frame_mux_size;
    for ( int node = 0;  node < node_mux_size; ++node ) {
        int x_node = y_node_size * node + y_node;
        for ( int frame = 0; frame < frame_mux_size; ++frame ) {
            float x = x_buf[x_node * x_frame_stride + x_frame + frame];
            sum += x;
        }
    }
    y_buf[y_node * y_frame_stride + y_frame] = sum * gain;
}


BBCU_DLL_EXPORT int bbcu_fp32_BinaryToReal_Forward
        (
            const float*    dev_x_buf,
            float*          dev_y_buf,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(y_frame_size, y_node_size);
    dim3    grid(1, 1);
    while ( block.y > 1 && block.x * block.y > 1024 ) {
        block.y = (block.y + 1) / 2;
    }
    grid.y = (y_node_size + (block.y - 1)) / block.y;
    while ( block.x > 1 && block.x * block.y > 1024 ) {
        block.x = (block.x + 1) / 2;
    }
    grid.x = (y_frame_size + (block.x - 1)) / block.x;
    
    kernal_fp32_BinaryToReal_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            1.0f / (node_mux_size * frame_mux_size),
            node_mux_size,
            frame_mux_size,
            y_node_size,
            x_frame_stride,
            y_frame_size,
            y_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



////////////////


__global__ void kernal_bit_fp32_BinaryToReal_Forward(
            int   const     *x_buf,
            float           *y_buf,
            float           gain,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride
        )
{
    int y_frame = blockDim.x * blockIdx.x + threadIdx.x;
    int y_node  = blockDim.y * blockIdx.y + threadIdx.y;

    if (y_frame >= y_frame_size || y_node >= y_node_size) {
        return;
    }


    float sum = 0;
    int x_frame    = y_frame * frame_mux_size;
    for ( int node = 0;  node < node_mux_size; ++node ) {
        int x_node = y_node_size * node + y_node;
        for ( int frame = 0; frame < frame_mux_size; ++frame ) {
            int x_unit     = ((x_frame + frame) >> 5);
            int x_bit      = ((x_frame + frame) & 0x1f);
            int x_bit_mask = (1 << x_bit);

            int x = x_buf[x_node * x_frame_stride + x_unit];
            if ( x & x_bit_mask ) {
                sum += 1.0;
            }
        }
    }
    y_buf[y_node * y_frame_stride + y_frame] = sum * gain;
}


BBCU_DLL_EXPORT int bbcu_bit_fp32_BinaryToReal_Forward
        (
            int   const     *dev_x_buf,
            float           *dev_y_buf,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(y_frame_size, y_node_size);
    dim3    grid(1, 1);
    while ( block.y > 1 && block.x * block.y > 1024 ) {
        block.y = (block.y + 1) / 2;
    }
    grid.y = (y_node_size + (block.y - 1)) / block.y;
    while ( block.x > 1 && block.x * block.y > 1024 ) {
        block.x = (block.x + 1) / 2;
    }
    grid.x = (y_frame_size + (block.x - 1)) / block.x;
    
    kernal_bit_fp32_BinaryToReal_Forward<<<grid, block, 0, streamId>>>
        (
            dev_x_buf,
            dev_y_buf,
            1.0f / (node_mux_size * frame_mux_size),
            node_mux_size,
            frame_mux_size,
            y_node_size,
            x_frame_stride,
            y_frame_size,
            y_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_BinaryToReal_Backward(
            const float*    dy_buf,
            float*          dx_buf,
            float           gain,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride
        )
{
    int y_frame = blockDim.x * blockIdx.x + threadIdx.x;
    int y_node  = blockDim.y * blockIdx.y + threadIdx.y;

    if (y_frame >= y_frame_size || y_node >= y_node_size) {
        return;
    }

    float dy  = dy_buf[y_node * y_frame_stride + y_frame];
    float val = dy * gain;

    int x_frame = y_frame * frame_mux_size;
    for ( int node = 0;  node < node_mux_size; ++node ) {
        int x_node = y_node_size * node + y_node;
        for ( int frame = 0; frame < frame_mux_size; ++frame ) {
            dx_buf[x_node * x_frame_stride + x_frame + frame] = val;
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_BinaryToReal_Backward
        (
            const float*    dev_dy_buf,
            float*          dev_dx_buf,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(y_frame_size, y_node_size);
    dim3    grid(1, 1);
    while ( block.y > 1 && block.x * block.y > 1024 ) {
        block.y = (block.y + 1) / 2;
    }
    grid.y = (y_node_size + (block.y - 1)) / block.y;
    while ( block.x > 1 && block.x * block.y > 1024 ) {
        block.x = (block.x + 1) / 2;
    }
    grid.x = (y_frame_size + (block.x - 1)) / block.x;
    
    kernal_fp32_BinaryToReal_Backward<<<grid, block, 0, streamId>>>(
            dev_dy_buf,
            dev_dx_buf,
            1.0f / (node_mux_size * frame_mux_size),
            node_mux_size,
            frame_mux_size,
            y_node_size,
            x_frame_stride,
            y_frame_size,
            y_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// end of file
