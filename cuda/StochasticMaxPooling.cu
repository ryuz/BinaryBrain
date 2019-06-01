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

__global__ void kernal_fp32_StochasticMaxPooling_Forward(
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
    int x       = blockIdx.y * blockDim.y + threadIdx.y;
    int y       = blockIdx.x;
    int c       = blockIdx.z * blockDim.z + threadIdx.z;
    int id      = threadIdx.x;
    int id_step = blockDim.x;
    
    if ( x < output_w_size && y < output_h_size ) {
        for ( int frame = id; frame < frame_size; frame += id_step ) {
            /// OR演算を実施(反転してANDを取って、出力反転)
            float out_sig = 1.0;
            for (int fy = 0; fy < filter_h_size; ++fy) {
                int iy = y * filter_h_size + fy;
                if ( iy < input_h_size ) {
                    for (int fx = 0; fx < filter_w_size; ++fx) {
                        int ix = x * filter_w_size + fx;
                        if ( ix < input_w_size ) {
                            float in_sig = x_buf[((c * input_h_size + iy) * input_w_size + ix) * frame_stride + frame];
                            out_sig *= (1.0 - in_sig);
                        }
                    }
                }
            }

            // 出力
            y_buf[((c * output_h_size + y) * output_w_size + x) * frame_stride + frame] = (1.0 - out_sig);
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_StochasticMaxPooling_Forward
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
    block.x = std::min((int)block.x, frame_size);
    block.y = std::min((int)block.y, output_w_size);

    kernal_fp32_StochasticMaxPooling_Forward<<<grid, block, 0, streamId>>>(
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
// forward 2x2
//////////////////////////////

__global__ void kernal_fp32_StochasticMaxPooling2x2_Forward(
            float const *x_buf,
            float       *y_buf,
            int         input_w_size,
            int         input_h_size,
            int         output_w_size,
            int         output_h_size,
            int         c_size,
            int         frame_size,
            int         frame_stride
        )
{
    int id = threadIdx.x;
    int id_step = blockDim.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if ( x < output_w_size && y < output_h_size ) {
        for ( int frame = id; frame < frame_size; frame += id_step ) {
            // OR演算
            float out_sig = 1.0;
            for (int fy = 0; fy < 2; ++fy) {
                int iy = y * 2 + fy;
                if ( iy < input_h_size ) {
                    for (int fx = 0; fx < 2; ++fx) {
                        int ix = x * 2 + fx;
                        if ( ix < input_w_size ) {
                            float in_sig = x_buf[((c * input_h_size + iy) * input_w_size + ix) * frame_stride + frame];
                            out_sig *= (1.0 - in_sig);
                        }
                    }
                }
            }

            // 出力
            y_buf[((c * output_h_size + y) * output_w_size + x) * frame_stride + frame] = (1.0 - out_sig);
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_StochasticMaxPooling2x2_Forward
        (
            float const *   dev_x_buf,
            float*          dev_y_buf,
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
    block.x = std::min((int)block.x, frame_size);
    block.y = std::min((int)block.y, output_w_size);

    kernal_fp32_StochasticMaxPooling2x2_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
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

__global__ void kernal_fp32_StochasticMaxPooling2x2_Backward(
            float const *x_buf,
            float const *dy_buf,
            float       *dx_buf,
            int         input_w_size,
            int         input_h_size,
            int         output_w_size,
            int         output_h_size,
            int         c_size,
            int         frame_size,
            int         frame_stride
        )
{
    int id = threadIdx.x;
    int id_step = blockDim.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if ( x < output_w_size && y < output_h_size ) {
        for ( int frame = id; frame < frame_size; frame += id_step ) {
            
            float in_sig[2][2] = {{0, 0}, {0, 0}};
            for (int fy = 0; fy < 2; ++fy) {
                int iy = y * 2 + fy;
                if ( iy < input_h_size ) {
                    for (int fx = 0; fx < 2; ++fx) {
                        int ix = x * 2 + fx;
                        if ( ix < input_w_size ) {
                            in_sig[fy][fx] = 1.0 - x_buf[((c * input_h_size + iy) * input_w_size + ix) * frame_stride + frame];
                        }
                    }
                }
            }

            float out_grad = dy_buf[((c * output_h_size + y) * output_w_size + x) * frame_stride + frame];

            float in_grad[2][2];
            in_grad[0][0] = out_grad * in_sig[0][1] * in_sig[1][0] * in_sig[1][1];
            in_grad[0][1] = out_grad * in_sig[0][0] * in_sig[1][0] * in_sig[1][1];
            in_grad[1][0] = out_grad * in_sig[0][0] * in_sig[0][1] * in_sig[1][1];
            in_grad[1][1] = out_grad * in_sig[0][0] * in_sig[0][1] * in_sig[1][0];

            for (int fy = 0; fy < 2; ++fy) {
                int iy = y * 2 + fy;
                if ( iy < input_h_size ) {
                    for (int fx = 0; fx < 2; ++fx) {
                        int ix = x * 2 + fx;
                        if ( ix < input_w_size ) {
                            dx_buf[((c * input_h_size + iy) * input_w_size + ix) * frame_stride + frame] = in_grad[fy][fx];
                        }
                    }
                }
            }
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_StochasticMaxPooling2x2_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
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
    block.x = std::min((int)block.x, frame_size);
    block.y = std::min((int)block.y, output_w_size);

    kernal_fp32_StochasticMaxPooling2x2_Backward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
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

