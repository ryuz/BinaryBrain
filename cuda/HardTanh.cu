#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_HardTanh_Forward(
            float const *x_buf,
            float       *y_buf,
            float       limit_min,
            float       limit_max,
            int         frame_size,
            int         frame_stride
        )
{
    int node    = blockIdx.x;
    int id      = threadIdx.x;
    int id_step = blockDim.x;
    
    for ( int frame = id; frame < frame_size; frame += id_step ) {
        float x = x_buf[frame_stride*node + frame];
        if (x <= limit_min) { x = limit_min; }
        if (x >= limit_max) { x = limit_max; }
        y_buf[frame_stride*node + frame] = x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_HardTanh_Forward
        (
            float const *   dev_x_buf,
            float*          dev_y_buf,
            float           limit_min,
            float           limit_max,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    int     unit_x = 512;
    
    dim3    grid(node_size);
    dim3    block(unit_x);
    
    kernal_fp32_HardTanh_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            limit_min,
            limit_max,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_HardTanh_Backward
        (
            float const *x_buf,
            float const *dy_buf,
            float       *dx_buf,
            float       limit_min,
            float       limit_max,
            int         frame_size,
            int         frame_stride
        )
{
    int node    = blockIdx.x;
    int id      = threadIdx.x;
    int id_step = blockDim.x;
    
    float const *x_ptr  = &x_buf[frame_stride*node];
    float const *dy_ptr = &dy_buf[frame_stride*node];
    float       *dx_ptr = &dx_buf[frame_stride*node];

    for ( int frame = id; frame < frame_size; frame += id_step ) {
        float x  = x_ptr[frame];
        float dy = dy_ptr[frame];
        if (x <= limit_min) { dy = 0.0; }
        if (x >= limit_max) { dy = 0.0; }
        dx_ptr[frame] = dy;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_HardTanh_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           limit_min,
            float           limit_max,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    int     unit_x = 512;
    
    dim3    grid(node_size);
    dim3    block(unit_x);

    kernal_fp32_HardTanh_Backward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
            limit_min,
            limit_max,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

