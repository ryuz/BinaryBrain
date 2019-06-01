#include <iostream>
#include <algorithm>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"

#include "Common.cuh"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_MaxPooling_Forward(
            float const *x_buf,
            float       *y_buf,
            int         filter_h_size,
            int         filter_w_size,
            int         input_w_size,
            int         input_h_size,
            int         output_w_size,
            int         output_h_size,
            int         c_size,
            int         frame_size,
            int         frame_stride
        )
{
    int frame_base = threadIdx.x;
    int frame_step = blockDim.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (y >= output_h_size || x >= output_w_size) {
        return;
    }

    for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
        // 最大値探索
        float max_val = -1.0e7f;
        for (int fy = 0; fy < filter_h_size; ++fy) {
            int iy = y * filter_h_size + fy;
            if ( iy < input_h_size ) {
                for (int fx = 0; fx < filter_w_size; ++fx) {
                    int ix = x * filter_w_size + fx;
                    if ( ix < input_w_size ) {
                        float sig = x_buf[((c * input_h_size + iy) * input_w_size + ix) * frame_stride + frame];
                        max_val = max(max_val, sig);
                    }
                }
            }
        }

        // 出力
        y_buf[((c * output_h_size + y) * output_w_size + x) * frame_stride + frame] = max_val;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_MaxPooling_Forward
        (
            float const *   dev_x_buf,
            float*          dev_y_buf,
            int             filter_h_size,
            int             filter_w_size,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(32, 32, 1);
    dim3    grid;
    grid.x = output_h_size;
    grid.y = (output_w_size + (block.y-1)) / block.y;
    grid.z = c_size;
    block.x = min(block.x, frame_size);
    block.y = min(block.y, output_w_size);

    kernal_fp32_MaxPooling_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            filter_h_size,
            filter_w_size,
            input_w_size,
            input_h_size,
            output_w_size,
            output_h_size,
            c_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


//////////////////


__global__ void kernal_bit_MaxPooling_Forward(
            int const   *x_buf,
            int         *y_buf,
            int         filter_h_size,
            int         filter_w_size,
            int         input_w_size,
            int         input_h_size,
            int         output_w_size,
            int         output_h_size,
            int         c_size,
            int         frame_size,
            int         frame_stride
        )
{
    int id      = threadIdx.x;
    int id_step = blockDim.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (y < output_h_size && x < output_w_size) {
        int loop_size = ((frame_size + 0x1f) & ~0x1f); 
        for ( int frame = id; frame < loop_size; frame += id_step ) {
            int unit     = (frame >> 5);
            int bit      = (frame & 0x1f);
            int bit_mask = (1 << bit);

            // 最大値探索
            int y_val = 0;
            if ( frame < frame_size ) {
                for (int fy = 0; fy < filter_h_size; ++fy) {
                    int iy = y * filter_h_size + fy;
                    if ( iy < input_h_size ) {
                        for (int fx = 0; fx < filter_w_size; ++fx) {
                            int ix = x * filter_w_size + fx;
                            if ( ix < input_w_size ) {
                                int x_val = x_buf[((c * input_h_size + iy) * input_w_size + ix) * frame_stride + unit];
                                y_val |= x_val;
                            }
                        }
                    }
                }
            }

            y_val = device_int_ShuffleOr(y_val & bit_mask);

            // 出力
            if ( bit == 0 ) {
                y_buf[((c * output_h_size + y) * output_w_size + x) * frame_stride + unit] = y_val;
            }
        }
    }
}


BBCU_DLL_EXPORT int bbcu_bit_MaxPooling_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int             filter_h_size,
            int             filter_w_size,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(32, 32, 1);
    dim3    grid;
    grid.x = output_h_size;
    grid.y = (output_w_size + (block.y-1)) / block.y;
    grid.z = c_size;
//  block.x = min(block.x, frame_size);
    block.y = min(block.y, output_w_size);

    kernal_bit_MaxPooling_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            filter_h_size,
            filter_w_size,
            input_w_size,
            input_h_size,
            output_w_size,
            output_h_size,
            c_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}




//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_MaxPooling_Backward(
            float const *x_buf,
            float const *y_buf,
            float const *dy_buf,
            float       *dx_buf,
            int         filter_h_size,
            int         filter_w_size,
            int         input_w_size,
            int         input_h_size,
            int         output_w_size,
            int         output_h_size,
            int         c_size,
            int         frame_size,
            int         frame_stride
        )
{
    int frame_base = threadIdx.x;
    int frame_step = blockDim.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (y >= output_h_size || x >= output_w_size) {
        return;
    }
    
    // 最大値箇所のみ伝播
    for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
        float out_sig = y_buf[((c * output_h_size + y) * output_w_size + x) * frame_stride + frame];
        float grad    = dy_buf[((c * output_h_size + y) * output_w_size + x) * frame_stride + frame];
        for (int fy = 0; fy < filter_h_size; ++fy) {
            int iy = y * filter_h_size + fy;
            if ( iy < input_h_size ) {
                for (int fx = 0; fx < filter_w_size; ++fx) {
                    int ix = x * filter_w_size + fx;
                    if ( ix < input_w_size ) {
                        float in_sig  = x_buf[((c * input_h_size + iy) * input_w_size + ix) * frame_stride + frame];
                        dx_buf[((c * input_h_size + iy) * input_w_size + ix) * frame_stride + frame] = (in_sig == out_sig) ? grad : 0;
                    }
                }
            }
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_MaxPooling_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_y_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            int             filter_h_size,
            int             filter_w_size,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(32, 32, 1);
    dim3    grid;
    grid.x = output_h_size;
    grid.y = (output_w_size + (block.y-1)) / block.y;
    grid.z = c_size;
    block.x = min(block.x, frame_size);
    block.y = min(block.y, output_w_size);

    kernal_fp32_MaxPooling_Backward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_dy_buf,
            dev_dx_buf,
            filter_h_size,
            filter_w_size,
            input_w_size,
            input_h_size,
            output_w_size,
            output_h_size,
            c_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



///////////


__global__ void kernal_bit_fp32_MaxPooling_Backward(
            int   const *x_buf,
            int   const *y_buf,
            float const *dy_buf,
            float       *dx_buf,
            int         filter_h_size,
            int         filter_w_size,
            int         input_w_size,
            int         input_h_size,
            int         output_w_size,
            int         output_h_size,
            int         c_size,
            int         frame_size,
            int         forward_frame_stride,
            int         backward_frame_stride
        )
{
    int id      = threadIdx.x;
    int id_step = blockDim.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (y < output_h_size && x < output_w_size) {
        // 最大値箇所のみ伝播
        for ( int frame = id; frame < frame_size; frame += id_step ) {
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);

            int   out_sig = (y_buf[((c * output_h_size + y) * output_w_size + x) * forward_frame_stride + unit] & bit);
            float grad    = dy_buf[((c * output_h_size + y) * output_w_size + x) * backward_frame_stride + frame];
            for (int fy = 0; fy < filter_h_size; ++fy) {
                int iy = y * filter_h_size + fy;
                if ( iy < input_h_size ) {
                    for (int fx = 0; fx < filter_w_size; ++fx) {
                        int ix = x * filter_w_size + fx;
                        if ( ix < input_w_size ) {
                            float in_sig  = (x_buf[((c * input_h_size + iy) * input_w_size + ix) * forward_frame_stride + unit] & bit);
                            dx_buf[((c * input_h_size + iy) * input_w_size + ix) * backward_frame_stride + frame] = (in_sig == out_sig) ? grad : 0;
                        }
                    }
                }
            }
        }
    }
}


BBCU_DLL_EXPORT int bbcu_bit_fp32_MaxPooling_Backward
        (
            int   const     *dev_x_buf,
            int   const     *dev_y_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            int             filter_h_size,
            int             filter_w_size,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             forward_frame_stride,
            int             backward_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(32, 32, 1);
    dim3    grid;
    grid.x = output_h_size;
    grid.y = (output_w_size + (block.y-1)) / block.y;
    grid.z = c_size;
    block.x = std::min((int)block.x, frame_size);
    block.y = std::min((int)block.y, output_w_size);

    kernal_bit_fp32_MaxPooling_Backward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_dy_buf,
            dev_dx_buf,
            filter_h_size,
            filter_w_size,
            input_w_size,
            input_h_size,
            output_w_size,
            output_h_size,
            c_size,
            frame_size,
            forward_frame_stride,
            backward_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// end of file
