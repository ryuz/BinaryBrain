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

template <typename T=float, bool fill=true>
__global__ void kernal_UpSampling_Forward(
            T   const   *x_buf,
            T           *y_buf,
            int         c_size,
            int         input_h_size,
            int         input_w_size,
            int         output_h_size,
            int         output_w_size,
            int         filter_h_size,
            int         filter_w_size,
            int         frame_size,
            int         frame_stride
        )
{
    int id      = threadIdx.x;
    int id_step = blockDim.x;
    int ix      = blockIdx.y * blockDim.y + threadIdx.y;
    int iy      = blockIdx.z * blockDim.z + threadIdx.z;

    if ( iy < input_h_size && ix < input_w_size ) {
        for ( int c = 0; c < c_size; ++c ) {
            int input_node = (c * input_h_size + iy) * input_w_size + ix;
            T   const *x_ptr = &x_buf[input_node * frame_stride];
            for ( int frame = id; frame < frame_size; frame += id_step ) {
                T x_val = x_ptr[frame];
                for ( int fy = 0; fy < filter_h_size; ++fy) {
                    int oy = iy * filter_h_size + fy;
                    for (int fx = 0; fx < filter_w_size; ++fx) {
                        int ox = ix * filter_w_size + fx;
                        int output_node = (c * output_h_size + oy) * output_w_size + ox;
                        T *y_ptr = &y_buf[output_node * frame_stride];
                        if ( fill ) {
                            y_ptr[frame] = x_val;
                        }
                        else {
                            if ( fx == (filter_w_size / 2) && fy == (filter_h_size / 2) ) {
                                y_ptr[frame] = x_val;
                            }
                            else {
                                y_ptr[frame] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_UpSampling_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int             input_w_size,
            int             input_h_size,
            int             c_size,
            int             filter_w_size,
            int             filter_h_size,
            int             fill,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    int output_w_size = input_w_size * filter_w_size;
    int output_h_size = input_h_size * filter_h_size;


    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(1024, 1, 1);
    while ( (int)block.x / 2 >= frame_size   ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= input_w_size ) { block.y /= 2; block.z *= 2; }
    while ( (int)block.z / 2 >= input_h_size ) { block.z /= 2; }

    dim3    grid(1, (input_w_size + (block.y - 1)) / block.y, (input_h_size + (block.z - 1)) / block.z);
    
    if ( fill ) {
        kernal_UpSampling_Forward<float, true><<<grid, block, 0, streamId>>>(
                dev_x_buf,
                dev_y_buf,
                c_size,
                input_h_size,
                input_w_size,
                output_h_size,
                output_w_size,
                filter_h_size,
                filter_w_size,
                frame_size,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }
    else {
        kernal_UpSampling_Forward<float, false><<<grid, block, 0, streamId>>>(
                dev_x_buf,
                dev_y_buf,
                c_size,
                input_h_size,
                input_w_size,
                output_h_size,
                output_w_size,
                filter_h_size,
                filter_w_size,
                frame_size,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }

    return 0;
}


BBCU_DLL_EXPORT int bbcu_bit_UpSampling_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int             input_w_size,
            int             input_h_size,
            int             c_size,
            int             filter_w_size,
            int             filter_h_size,
            int             fill,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    int output_w_size = input_w_size * filter_w_size;
    int output_h_size = input_h_size * filter_h_size;


    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    frame_size = (frame_size + 31) / 32;

    dim3    block(1024, 1, 1);
    while ( (int)block.x / 2 >= frame_size   ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= input_w_size ) { block.y /= 2; block.z *= 2; }
    while ( (int)block.z / 2 >= input_h_size ) { block.z /= 2; }

    dim3    grid(1, (input_w_size + (block.y - 1)) / block.y, (input_h_size + (block.z - 1)) / block.z);
    
    if ( fill ) {
        kernal_UpSampling_Forward<int, true><<<grid, block, 0, streamId>>>(
                dev_x_buf,
                dev_y_buf,
                c_size,
                input_h_size,
                input_w_size,
                output_h_size,
                output_w_size,
                filter_h_size,
                filter_w_size,
                frame_size,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }
    else {
        kernal_UpSampling_Forward<int, false><<<grid, block, 0, streamId>>>(
                dev_x_buf,
                dev_y_buf,
                c_size,
                input_h_size,
                input_w_size,
                output_h_size,
                output_w_size,
                filter_h_size,
                filter_w_size,
                frame_size,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }

    return 0;
}



//////////////////////////////
// backward
//////////////////////////////

template <typename T=float, bool fill=true>
__global__ void kernal_UpSampling_Backward(
            T   const   *dy_buf,
            T           *dx_buf,
            int         c_size,
            int         input_h_size,
            int         input_w_size,
            int         output_h_size,
            int         output_w_size,
            int         filter_h_size,
            int         filter_w_size,
            int         frame_size,
            int         frame_stride
        )
{
    int id      = threadIdx.x;
    int id_step = blockDim.x;
    int ix      = blockIdx.y * blockDim.y + threadIdx.y;
    int iy      = blockIdx.z * blockDim.z + threadIdx.z;

    if ( iy < input_h_size && ix < input_w_size ) {
        for ( int c = 0; c < c_size; ++c ) {
            int input_node = (c * input_h_size + iy) * input_w_size + ix;
            T   *dx_ptr = &dx_buf[input_node * frame_stride];
            if ( fill ) {
                for ( int frame = id; frame < frame_size; frame += id_step ) {
                    T dx_val = 0;
                    for ( int fy = 0; fy < filter_h_size; ++fy) {
                        int oy = iy * filter_h_size + fy;
                        for (int fx = 0; fx < filter_w_size; ++fx) {
                            int ox = ix * filter_w_size + fx;
                            int output_node = (c * output_h_size + oy) * output_w_size + ox;
                            T   const   *dy_ptr = &dy_buf[output_node * frame_stride];
                            dx_val += dy_ptr[frame];
                        }
                    }
                    dx_ptr[frame] = dx_val;
                }
            }
            else {
                int fx = (filter_w_size / 2);
                int fy = (filter_h_size / 2);
                int oy = iy * filter_h_size + fy;
                int ox = ix * filter_w_size + fx;
                int output_node = (c * output_h_size + oy) * output_w_size + ox;
                T   const   *dy_ptr = &dy_buf[output_node * frame_stride];
                for ( int frame = id; frame < frame_size; frame += id_step ) {
                    dx_ptr[frame] = dy_ptr[frame];
                }
            }
        }
    }
}



BBCU_DLL_EXPORT int bbcu_fp32_UpSampling_Backward
        (
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            int             input_w_size,
            int             input_h_size,
            int             c_size,
            int             filter_w_size,
            int             filter_h_size,
            int             fill,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    int output_w_size = input_w_size * filter_w_size;
    int output_h_size = input_h_size * filter_h_size;


    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(1024, 1, 1);
    while ( (int)block.x / 2 >= frame_size   ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= input_w_size ) { block.y /= 2; block.z *= 2; }
    while ( (int)block.z / 2 >= input_h_size ) { block.z /= 2; }

    dim3    grid(1, (input_w_size + (block.y - 1)) / block.y, (input_h_size + (block.z - 1)) / block.z);
    
    if ( fill ) {
        kernal_UpSampling_Backward<float, true><<<grid, block, 0, streamId>>>(
                dev_dy_buf,
                dev_dx_buf,
                c_size,
                input_h_size,
                input_w_size,
                output_h_size,
                output_w_size,
                filter_h_size,
                filter_w_size,
                frame_size,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }
    else {
        kernal_UpSampling_Backward<float, false><<<grid, block, 0, streamId>>>(
                dev_dy_buf,
                dev_dx_buf,
                c_size,
                input_h_size,
                input_w_size,
                output_h_size,
                output_w_size,
                filter_h_size,
                filter_w_size,
                frame_size,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }

    return 0;
}


// end of file
