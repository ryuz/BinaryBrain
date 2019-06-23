#include <iostream>
#include <algorithm>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



//////////////////////////////
// forward
//////////////////////////////

#define IM2COL_BORDER_CONSTANT      0
#define IM2COL_BORDER_REFLECT       1
#define IM2COL_BORDER_REFLECT_101   2
#define IM2COL_BORDER_REPLICATE     3
#define IM2COL_BORDER_WRAP          4

__device__ __forceinline__ bool device_Im2Col_Border(int mode, int &x, int &y, int w, int h)
{
    switch ( mode ) {
    case IM2COL_BORDER_REFLECT:
        if ( x < 0  ) { x = -x - 1; }
        if ( y < 0  ) { y = -y - 1; }
        if ( x >= w ) { x = (w - 1) - (x - w); }
        if ( y >= h ) { y = (h - 1) - (y - h); }
        return true;
    
    case IM2COL_BORDER_REFLECT_101:
        if ( x < 0  ) { x = -x; }
        if ( y < 0  ) { y = -y; }
        if ( x >= w ) { x = (w - 2) - (x - w); }
        if ( y >= h ) { y = (h - 2) - (y - h); }
        return true;

    case IM2COL_BORDER_REPLICATE:
        if ( x < 0  ) { x = 0; }
        if ( y < 0  ) { y = 0; }
        if ( x >= w ) { x = w - 1; }
        if ( y >= h ) { y = h - 1; }
        return true;

    case IM2COL_BORDER_WRAP:
        if ( x < 0  ) { x += w; }
        if ( y < 0  ) { y += h; }
        if ( x >= w ) { x -= w; }
        if ( y >= h ) { y -= h; }
        return true;
    }

    return false;
}



__global__ void kernal_fp32_Im2Col_Forward(
            float const     *x_buf,
            float           *y_buf,
            int             x_stride,
            int             y_stride,
            int             x_offset,
            int             y_offset,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             output_frame_size,
            int             output_frame_stride,
            int             output_w_size,
            int             output_size,
            int             border_mode,
            float           border_value
        )
{
    int filter_w_size = blockDim.y;
    int filter_h_size = blockDim.z;

    int output_frame = blockDim.x * blockIdx.x + threadIdx.x;

    if ( output_frame < output_frame_size ) {
        int fx           = threadIdx.y;
        int fy           = threadIdx.z;
        int c            = blockIdx.y;
    
        int input_frame = output_frame / output_size;
        int f           = output_frame % output_size;
        int iy = (f / output_w_size) * y_stride - y_offset + fy;
        int ix = (f % output_w_size) * x_stride - x_offset + fx;

        float x = border_value;
        if ( iy >= 0 && iy < input_h_size && ix >= 0 && ix < input_w_size ) {
            int input_node  = (c * input_h_size  + iy) * input_w_size  + ix;
            x = x_buf[input_node * input_frame_stride + input_frame];
        }
        else {
            if ( device_Im2Col_Border(border_mode, ix, iy, input_w_size, input_h_size) ) {
                int input_node  = (c * input_h_size  + iy) * input_w_size  + ix;
                x = x_buf[input_node * input_frame_stride + input_frame];
            }
        }

        int output_node = (c * filter_h_size + fy) * filter_w_size + fx;    
        y_buf[output_node * output_frame_stride + output_frame] = x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Im2Col_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int             x_stride,
            int             y_stride,
            int             x_offset,
            int             y_offset,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            int             output_w_size,
            int             output_h_size,
            int             output_frame_stride,
            int             filter_w_size,
            int             filter_h_size,
            int             border_mode,
            float           border_value,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    int output_c_size = input_c_size;
    int output_size   = output_w_size * output_h_size;
    
    int output_frame_size = input_frame_size * output_size;
    
    int     frame_unit = 1024;
    while ( frame_unit * filter_w_size * filter_h_size > 1024 ) { frame_unit /= 2; }
    BBCU_ASSERT(frame_unit > 0);

    dim3    block(frame_unit, filter_w_size, filter_h_size);
    dim3    grid((output_frame_size + (frame_unit-1))/frame_unit, output_c_size);
    
    kernal_fp32_Im2Col_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            x_stride,
            y_stride,
            x_offset,
            y_offset,
            input_frame_stride,
            input_w_size,
            input_h_size,          
            output_frame_size,
            output_frame_stride,
            output_w_size,
            output_size,
            border_mode,
            border_value
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


__global__ void kernal_bit_Im2Col_Forward(
            int const       *x_buf,
            int             *y_buf,
            int             x_stride,
            int             y_stride,
            int             x_offset,
            int             y_offset,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,           
            int             output_frame_size,
            int             output_frame_stride,
            int             output_w_size,
            int             output_size,
            int             border_mode
    )
{
    int output_frame_unit = blockDim.x * blockIdx.x + threadIdx.x;

    if ( output_frame_unit < output_frame_stride ) {
        int filter_w_size = blockDim.y;
        int filter_h_size = blockDim.z;

        int fx          = threadIdx.y;
        int fy          = threadIdx.z;
        int c           = blockIdx.y;

        int output_node = (c * filter_h_size + fy) * filter_w_size + fx;

        int y = 0;
        for ( int i = 0; i < 32; ++i ) {
            int output_frame = output_frame_unit * 32 + i;
            if ( output_frame < output_frame_size ) {
                int input_frame = output_frame / output_size;
                int f           = output_frame % output_size;
                int iy = (f / output_w_size) * y_stride - y_offset + fy;
                int ix = (f % output_w_size) * x_stride - x_offset + fx;

                if ( iy >= 0 && iy < input_h_size && ix >= 0 && ix < input_w_size ) {
                    int input_node  = (c * input_h_size  + iy) * input_w_size  + ix;
                    int const *x_ptr = &x_buf[input_node  * input_frame_stride];
                    int x = ((x_ptr[input_frame / 32] >> (input_frame % 32)) & 1);
                    y |= (x << i);
                }
                else {
                    if ( device_Im2Col_Border(border_mode, ix, iy, input_w_size, input_h_size) ) {
                        int input_node  = (c * input_h_size  + iy) * input_w_size  + ix;
                        int const *x_ptr = &x_buf[input_node  * input_frame_stride];
                        int x = ((x_ptr[input_frame / 32] >> (input_frame % 32)) & 1);
                        y |= (x << i);
                    }
                }
            }
        }

        int *y_ptr = &y_buf[output_node * output_frame_stride];
        y_ptr[output_frame_unit] = y;
    }
}

BBCU_DLL_EXPORT int bbcu_bit_Im2Col_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int             x_stride,
            int             y_stride,
            int             x_offset,
            int             y_offset,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            int             output_w_size,
            int             output_h_size,
            int             output_frame_stride,
            int             filter_w_size,
            int             filter_h_size,
            int             border_mode,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    int output_c_size = input_c_size;
    int output_size   = output_w_size * output_h_size;
    
    int output_frame_size = input_frame_size * output_size;
    int output_frame_unit = (output_frame_size + 31) / 32;

    int     frame_unit = 16;
    dim3    grid((output_frame_unit + (frame_unit-1))/frame_unit, output_c_size);
    dim3    block(frame_unit, filter_w_size, filter_h_size);
    
    kernal_bit_Im2Col_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            x_stride,
            y_stride,
            x_offset,
            y_offset,
            input_frame_stride,
            input_w_size,
            input_h_size,
            output_frame_size,
            output_frame_stride,
            output_w_size,
            output_size,
            border_mode
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}




//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_Im2Col_Backward(
            float const     *dy_buf,
            float           *dx_buf,

            int             x_stride,
            int             y_stride,
            int             x_offset,
            int             y_offset,

            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            
            int             output_frame_size,
            int             output_frame_stride,
            int             output_w_size,
            int             output_h_size,

            int             filter_w_size,
            int             filter_h_size
        )
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.z * blockIdx.z + threadIdx.z;
    
    if ( x < input_w_size && y < input_h_size && c < input_c_size ) {
        float const *dy_ptr = &dy_buf[c * filter_h_size * filter_w_size * output_frame_stride];

        int iy_limit = (output_h_size - 1) * y_stride;
        int ix_limit = (output_w_size - 1) * x_stride;

        int x_align = x % x_stride;
        int y_align = y % y_stride;

        for ( int input_frame = 0; input_frame < input_frame_size; ++input_frame ) {
            float dx = 0;
            for (int fy = y_align; fy < filter_h_size; fy += y_stride ) {
                int iy = y - fy + y_offset;
                if ( iy >= 0 && iy <= iy_limit ) {
                    for (int fx = x_align; fx < filter_w_size; fx += x_stride) {
                        int ix = x - fx + x_offset;
                        if (ix >= 0 && ix <= ix_limit ) {
                            int output_frame = (input_frame * output_h_size + (iy/y_stride)) * output_w_size + (ix/x_stride);
                            int output_node  = fy * filter_w_size + fx;
                            dx += dy_ptr[output_node * output_frame_stride + output_frame];
                        }
                    }
                }
            }
            dx_buf[((c * input_h_size + y) * input_w_size + x) * input_frame_stride + input_frame] = dx;
        }
    }
}



BBCU_DLL_EXPORT int bbcu_fp32_Im2Col_Backward
        (
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            int             x_stride,
            int             y_stride,
            int             x_offset,
            int             y_offset,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            int             output_w_size,
            int             output_h_size,
            int             output_frame_stride,            
            int             filter_w_size,
            int             filter_h_size,            
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

//  int output_c_size = input_c_size;
//    int output_w_size = input_w_size - filter_w_size + 1;
//    int output_h_size = input_h_size - filter_h_size + 1;
    int output_size   = output_w_size * output_h_size;
    
    int output_frame_size = input_frame_size * output_size;
    
    dim3    block(1024, 1, 1);
    while ( (int)block.x / 2 >= input_w_size ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= input_h_size ) { block.y /= 2; block.z *= 2; }
    while ( (int)block.z / 2 >= input_c_size ) { block.z /= 2; }
    block.z = std::min(64, (int)block.z);

    dim3    grid;
    grid.x = (input_w_size + block.x - 1) / block.x;
    grid.y = (input_h_size + block.y - 1) / block.y;
    grid.z = (input_c_size + block.z - 1) / block.z;

//  dim3    grid(input_w_size, input_h_size, 1);
//  dim3    block(1, 1, input_c_size);
    
    kernal_fp32_Im2Col_Backward<<<grid, block, 0, streamId>>>(
            dev_dy_buf,
            dev_dx_buf,
            x_stride,
            y_stride,
            x_offset,
            y_offset,
            input_frame_size,
            input_frame_stride,
            input_w_size,
            input_h_size,
            input_c_size,
            output_frame_size,
            output_frame_stride,
            output_w_size,
            output_h_size,
            filter_w_size,
            filter_h_size
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



