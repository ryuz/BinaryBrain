#include <iostream>
#include <chrono>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#include "Common.cuh"
#include "StochasticLut.cuh"


// -------------------------------------------------
//  Forward
// -------------------------------------------------

// real type
template<int N=6, typename T=float, int MAX_NODE_UNIT=32>
__global__ void kernal_StochasticLut_Forward(
            T   const   *x_buf,
            T           *y_buf,
            int const   *input_index,
            T   const   *W_buf,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         input_binary,
            int         lut_binarize,
            T           unbinarize_bias
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T       W[(1<<N)][MAX_NODE_UNIT];
                T const *x_ptr[N];
                T       *y_ptr;
    
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < (1<<N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1<<N) + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > 0.5 ? 1.0 : 0.0;
            }
        }
        
        // read input index
        for ( int i = 0; i < N; ++i ) {
            x_ptr[i] = &x_buf[frame_stride * input_index[N*node + i]];
        }

        y_ptr = &y_buf[node * frame_stride];
    }

    __syncthreads();
    
    for (int frame = id; frame < frame_size; frame += id_step) {
        if ( node < node_size ) {
            T   x[N];
            if ( input_binary ) {
                for ( int i = 0; i < N; ++i) {
                    x[i] = 0.5 + ((x_ptr[i][frame] > 0.5) ? +unbinarize_bias : -unbinarize_bias);
                }
            }
            else {
                for ( int i = 0; i < N; ++i) {
                    x[i] = min(1.0, max(0.0, x_ptr[i][frame]));
                }
            }

            T   y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

#if 0
            T   xp[N], xn[N];
            for ( int i = 0; i < N; ++i) {
                T x_val = x_ptr[i][frame];
                if ( input_binary ) {
                    x_val = 0.5 + ((x_val > 0.5) ? +unbinarize_bias : -unbinarize_bias);
                }
                else {
                    x_val = min(1.0, max(0.0, x_val));
                }

                xp[i] = x_val;
                xn[i] = 1.0 - x_val;
            }

            T x0_00 = xn[1] * xn[0];
            T x0_01 = xn[1] * xp[0];
            T x0_10 = xp[1] * xn[0];
            T x0_11 = xp[1] * xp[0];
            T x1_00 = xn[3] * xn[2];
            T x1_01 = xn[3] * xp[2];
            T x1_10 = xp[3] * xn[2];
            T x1_11 = xp[3] * xp[2];
            T x2_00 = xn[5] * xn[4];
            T x2_01 = xn[5] * xp[4];
            T x2_10 = xp[5] * xn[4];
            T x2_11 = xp[5] * xp[4];

            T y = 0;
            T x2_00_x1_00 = x2_00 * x1_00;
            y += W[0 ][node_id] * x2_00_x1_00 * x0_00;
            y += W[1 ][node_id] * x2_00_x1_00 * x0_01;
            y += W[2 ][node_id] * x2_00_x1_00 * x0_10;
            y += W[3 ][node_id] * x2_00_x1_00 * x0_11;
            T x2_00_x1_01 = x2_00 * x1_01;
            y += W[4 ][node_id] * x2_00_x1_01 * x0_00;
            y += W[5 ][node_id] * x2_00_x1_01 * x0_01;
            y += W[6 ][node_id] * x2_00_x1_01 * x0_10;
            y += W[7 ][node_id] * x2_00_x1_01 * x0_11;
            T x2_00_x1_10 = x2_00 * x1_10;
            y += W[8 ][node_id] * x2_00_x1_10 * x0_00;
            y += W[9 ][node_id] * x2_00_x1_10 * x0_01;
            y += W[10][node_id] * x2_00_x1_10 * x0_10;
            y += W[11][node_id] * x2_00_x1_10 * x0_11;
            T x2_00_x1_11 = x2_00 * x1_11;
            y += W[12][node_id] * x2_00_x1_11 * x0_00;
            y += W[13][node_id] * x2_00_x1_11 * x0_01;
            y += W[14][node_id] * x2_00_x1_11 * x0_10;
            y += W[15][node_id] * x2_00_x1_11 * x0_11;
            T x2_01_x1_00 = x2_01 * x1_00;
            y += W[16][node_id] * x2_01_x1_00 * x0_00;
            y += W[17][node_id] * x2_01_x1_00 * x0_01;
            y += W[18][node_id] * x2_01_x1_00 * x0_10;
            y += W[19][node_id] * x2_01_x1_00 * x0_11;
            T x2_01_x1_01 = x2_01 * x1_01;
            y += W[20][node_id] * x2_01_x1_01 * x0_00;
            y += W[21][node_id] * x2_01_x1_01 * x0_01;
            y += W[22][node_id] * x2_01_x1_01 * x0_10;
            y += W[23][node_id] * x2_01_x1_01 * x0_11;
            T x2_01_x1_10 = x2_01 * x1_10;
            y += W[24][node_id] * x2_01_x1_10 * x0_00;
            y += W[25][node_id] * x2_01_x1_10 * x0_01;
            y += W[26][node_id] * x2_01_x1_10 * x0_10;
            y += W[27][node_id] * x2_01_x1_10 * x0_11;
            T x2_01_x1_11 = x2_01 * x1_11;
            y += W[28][node_id] * x2_01_x1_11 * x0_00;
            y += W[29][node_id] * x2_01_x1_11 * x0_01;
            y += W[30][node_id] * x2_01_x1_11 * x0_10;
            y += W[31][node_id] * x2_01_x1_11 * x0_11;
            T x2_10_x1_00 = x2_10 * x1_00;
            y += W[32][node_id] * x2_10_x1_00 * x0_00;
            y += W[33][node_id] * x2_10_x1_00 * x0_01;
            y += W[34][node_id] * x2_10_x1_00 * x0_10;
            y += W[35][node_id] * x2_10_x1_00 * x0_11;
            T x2_10_x1_01 = x2_10 * x1_01;
            y += W[36][node_id] * x2_10_x1_01 * x0_00;
            y += W[37][node_id] * x2_10_x1_01 * x0_01;
            y += W[38][node_id] * x2_10_x1_01 * x0_10;
            y += W[39][node_id] * x2_10_x1_01 * x0_11;
            T x2_10_x1_10 = x2_10 * x1_10;
            y += W[40][node_id] * x2_10_x1_10 * x0_00;
            y += W[41][node_id] * x2_10_x1_10 * x0_01;
            y += W[42][node_id] * x2_10_x1_10 * x0_10;
            y += W[43][node_id] * x2_10_x1_10 * x0_11;
            T x2_10_x1_11 = x2_10 * x1_11;
            y += W[44][node_id] * x2_10_x1_11 * x0_00;
            y += W[45][node_id] * x2_10_x1_11 * x0_01;
            y += W[46][node_id] * x2_10_x1_11 * x0_10;
            y += W[47][node_id] * x2_10_x1_11 * x0_11;
            T x2_11_x1_00 = x2_11 * x1_00;
            y += W[48][node_id] * x2_11_x1_00 * x0_00;
            y += W[49][node_id] * x2_11_x1_00 * x0_01;
            y += W[50][node_id] * x2_11_x1_00 * x0_10;
            y += W[51][node_id] * x2_11_x1_00 * x0_11;
            T x2_11_x1_01 = x2_11 * x1_01;
            y += W[52][node_id] * x2_11_x1_01 * x0_00;
            y += W[53][node_id] * x2_11_x1_01 * x0_01;
            y += W[54][node_id] * x2_11_x1_01 * x0_10;
            y += W[55][node_id] * x2_11_x1_01 * x0_11;
            T x2_11_x1_10 = x2_11 * x1_10;
            y += W[56][node_id] * x2_11_x1_10 * x0_00;
            y += W[57][node_id] * x2_11_x1_10 * x0_01;
            y += W[58][node_id] * x2_11_x1_10 * x0_10;
            y += W[59][node_id] * x2_11_x1_10 * x0_11;
            T x2_11_x1_11 = x2_11 * x1_11;
            y += W[60][node_id] * x2_11_x1_11 * x0_00;
            y += W[61][node_id] * x2_11_x1_11 * x0_01;
            y += W[62][node_id] * x2_11_x1_11 * x0_10;
            y += W[63][node_id] * x2_11_x1_11 * x0_11;
#endif

            // clamp
            y = max(0.0, y);
            y = min(1.0, y);
        
            y_ptr[frame] = y;
        }
    }
}


int bbcu_fp32_StochasticLut6_Forward
        (
            const float     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             input_binary,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 512;
    unsigned int const MAX_FRAME_UNIT = 512;
    unsigned int const MAX_NODE_UNIT  = 64;

#if 0
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size  ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size  ) { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size ) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (node_size + (block.y - 1)) / block.y);
    
    kernal_StochasticLut_Forward<6, float, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            node_size,
            frame_size,
            frame_stride,
            input_binary,
            lut_binarize,
            unbinarize_bias
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}



// bit packing
template<int N=6, typename T=float, int MAX_NODE_UNIT=32>
__global__ void kernal_bit_StochasticLut_Forward(
            int const   *x_buf,
            T           *y_buf,
            int const   *input_index,
            T   const   *W_buf,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         bin_frame_stride,
            int         binary_mode,
            T           unbinarize_bias
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__ T    W[(1 << N)][MAX_NODE_UNIT];
    int   const     *x_ptr[N];
    T               *y_ptr;
    
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < (1 << N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1 << N) + i];
            if ( binary_mode ) {
                W[i][node_id] = W[i][node_id] > 0.5 ? 1.0 : 0.0;
            }
        }
        
        // read input index
        for ( int i = 0; i < N; ++i ) {
            x_ptr[i] = &x_buf[bin_frame_stride * input_index[N*node + i]];
        }

        y_ptr = &y_buf[node * frame_stride];
    }

    __syncthreads();
    
    for (int frame = id; frame < frame_size; frame += id_step) {
        if ( node < node_size ) {
            int bit_mask = (1 << (frame & 0x1f));
            int unit     = (frame >> 5);
            
            T   x[N];
            for ( int i = 0; i < N; ++i) {
                x[i] = 0.5 + ((x_ptr[i][unit] & bit_mask) ? +unbinarize_bias : -unbinarize_bias);
            }

            T   y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

#if 0
            T   xp[N], xn[N];
            for ( int i = 0; i < N; ++i) {
                T x_val =  0.5 + ((x_ptr[i][unit] & bit) ? +unbinarize_bias : -unbinarize_bias);
                xp[i] = x_val;
                xn[i] = 1.0 - x_val;
            }

            T x0_00 = xn[1] * xn[0];
            T x0_01 = xn[1] * xp[0];
            T x0_10 = xp[1] * xn[0];
            T x0_11 = xp[1] * xp[0];
            T x1_00 = xn[3] * xn[2];
            T x1_01 = xn[3] * xp[2];
            T x1_10 = xp[3] * xn[2];
            T x1_11 = xp[3] * xp[2];
            T x2_00 = xn[5] * xn[4];
            T x2_01 = xn[5] * xp[4];
            T x2_10 = xp[5] * xn[4];
            T x2_11 = xp[5] * xp[4];

            T y = 0;
            T x2_00_x1_00 = x2_00 * x1_00;
            y += W[0 ][node_id] * x2_00_x1_00 * x0_00;
            y += W[1 ][node_id] * x2_00_x1_00 * x0_01;
            y += W[2 ][node_id] * x2_00_x1_00 * x0_10;
            y += W[3 ][node_id] * x2_00_x1_00 * x0_11;
            T x2_00_x1_01 = x2_00 * x1_01;
            y += W[4 ][node_id] * x2_00_x1_01 * x0_00;
            y += W[5 ][node_id] * x2_00_x1_01 * x0_01;
            y += W[6 ][node_id] * x2_00_x1_01 * x0_10;
            y += W[7 ][node_id] * x2_00_x1_01 * x0_11;
            T x2_00_x1_10 = x2_00 * x1_10;
            y += W[8 ][node_id] * x2_00_x1_10 * x0_00;
            y += W[9 ][node_id] * x2_00_x1_10 * x0_01;
            y += W[10][node_id] * x2_00_x1_10 * x0_10;
            y += W[11][node_id] * x2_00_x1_10 * x0_11;
            T x2_00_x1_11 = x2_00 * x1_11;
            y += W[12][node_id] * x2_00_x1_11 * x0_00;
            y += W[13][node_id] * x2_00_x1_11 * x0_01;
            y += W[14][node_id] * x2_00_x1_11 * x0_10;
            y += W[15][node_id] * x2_00_x1_11 * x0_11;
            T x2_01_x1_00 = x2_01 * x1_00;
            y += W[16][node_id] * x2_01_x1_00 * x0_00;
            y += W[17][node_id] * x2_01_x1_00 * x0_01;
            y += W[18][node_id] * x2_01_x1_00 * x0_10;
            y += W[19][node_id] * x2_01_x1_00 * x0_11;
            T x2_01_x1_01 = x2_01 * x1_01;
            y += W[20][node_id] * x2_01_x1_01 * x0_00;
            y += W[21][node_id] * x2_01_x1_01 * x0_01;
            y += W[22][node_id] * x2_01_x1_01 * x0_10;
            y += W[23][node_id] * x2_01_x1_01 * x0_11;
            T x2_01_x1_10 = x2_01 * x1_10;
            y += W[24][node_id] * x2_01_x1_10 * x0_00;
            y += W[25][node_id] * x2_01_x1_10 * x0_01;
            y += W[26][node_id] * x2_01_x1_10 * x0_10;
            y += W[27][node_id] * x2_01_x1_10 * x0_11;
            T x2_01_x1_11 = x2_01 * x1_11;
            y += W[28][node_id] * x2_01_x1_11 * x0_00;
            y += W[29][node_id] * x2_01_x1_11 * x0_01;
            y += W[30][node_id] * x2_01_x1_11 * x0_10;
            y += W[31][node_id] * x2_01_x1_11 * x0_11;
            T x2_10_x1_00 = x2_10 * x1_00;
            y += W[32][node_id] * x2_10_x1_00 * x0_00;
            y += W[33][node_id] * x2_10_x1_00 * x0_01;
            y += W[34][node_id] * x2_10_x1_00 * x0_10;
            y += W[35][node_id] * x2_10_x1_00 * x0_11;
            T x2_10_x1_01 = x2_10 * x1_01;
            y += W[36][node_id] * x2_10_x1_01 * x0_00;
            y += W[37][node_id] * x2_10_x1_01 * x0_01;
            y += W[38][node_id] * x2_10_x1_01 * x0_10;
            y += W[39][node_id] * x2_10_x1_01 * x0_11;
            T x2_10_x1_10 = x2_10 * x1_10;
            y += W[40][node_id] * x2_10_x1_10 * x0_00;
            y += W[41][node_id] * x2_10_x1_10 * x0_01;
            y += W[42][node_id] * x2_10_x1_10 * x0_10;
            y += W[43][node_id] * x2_10_x1_10 * x0_11;
            T x2_10_x1_11 = x2_10 * x1_11;
            y += W[44][node_id] * x2_10_x1_11 * x0_00;
            y += W[45][node_id] * x2_10_x1_11 * x0_01;
            y += W[46][node_id] * x2_10_x1_11 * x0_10;
            y += W[47][node_id] * x2_10_x1_11 * x0_11;
            T x2_11_x1_00 = x2_11 * x1_00;
            y += W[48][node_id] * x2_11_x1_00 * x0_00;
            y += W[49][node_id] * x2_11_x1_00 * x0_01;
            y += W[50][node_id] * x2_11_x1_00 * x0_10;
            y += W[51][node_id] * x2_11_x1_00 * x0_11;
            T x2_11_x1_01 = x2_11 * x1_01;
            y += W[52][node_id] * x2_11_x1_01 * x0_00;
            y += W[53][node_id] * x2_11_x1_01 * x0_01;
            y += W[54][node_id] * x2_11_x1_01 * x0_10;
            y += W[55][node_id] * x2_11_x1_01 * x0_11;
            T x2_11_x1_10 = x2_11 * x1_10;
            y += W[56][node_id] * x2_11_x1_10 * x0_00;
            y += W[57][node_id] * x2_11_x1_10 * x0_01;
            y += W[58][node_id] * x2_11_x1_10 * x0_10;
            y += W[59][node_id] * x2_11_x1_10 * x0_11;
            T x2_11_x1_11 = x2_11 * x1_11;
            y += W[60][node_id] * x2_11_x1_11 * x0_00;
            y += W[61][node_id] * x2_11_x1_11 * x0_01;
            y += W[62][node_id] * x2_11_x1_11 * x0_10;
            y += W[63][node_id] * x2_11_x1_11 * x0_11;
#endif

            // clamp
            y = max(0.0, y);
            y = min(1.0, y);

            y_ptr[frame] = y;
        }
    }
}


int bbcu_bit_fp32_StochasticLut6_Forward
        (
            int   const     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             bin_frame_stride,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 512;
    unsigned int const MAX_FRAME_UNIT = 512;
    unsigned int const MAX_NODE_UNIT  = 64;

#if 0
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size  ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size  ) { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size ) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (node_size + (block.y - 1)) / block.y);
    
    kernal_bit_StochasticLut_Forward<6, float, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            node_size,
            frame_size,
            frame_stride,
            bin_frame_stride,
            lut_binarize,
            unbinarize_bias
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}




// -------------------------------------------------
//  Backward
// -------------------------------------------------

// real type
template<int N=6, typename T=float, int MAX_FRAME_UNIT=256, int MAX_NODE_UNIT=16>
__global__ void kernal_StochasticLut_Backward
        (
            T   const   *x_buf,
            T   const   *dy_buf,
            T           *dx_buf,
            int const   *input_index,
            T   const   *W_buf,
            T           *dW_buf,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         input_binary,
            int         lut_binarize,
            T           unbinarize_bias
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T       sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  T       dW_prev[(1 << N)][MAX_NODE_UNIT];
    __shared__  T       W[(1 << N)][MAX_NODE_UNIT];
                T       dW[(1 << N)];
                T const *x_ptr[6];
                T const *dy_ptr;
    
    // initialize dW
    if ( node < node_size ) {
        for ( int i = 0; i < (1 << N); ++i) {
            dW[i] = 0;
        }

        for ( int i = id; i < (1 << N); i += id_step ) {
            dW_prev[i][node_id] = dW_buf[node * (1 << N) + i];
        }

        // read W
        for ( int i = id; i < (1 << N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1 << N) + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > 0.5 ? 1.0 : 0.0;
            }
        }
    
        // init pointer
        for ( int i = 0; i < 6; ++i ) {
            int input_node = input_index[6*node + i];
            x_ptr[i]  = &x_buf[input_node * frame_stride];
        }

        dy_ptr = &dy_buf[node * frame_stride];
    }

    __syncthreads();

    for ( int frame = id; frame < frame_size; frame += id_step ) {
        if ( node < node_size ) {
            // read x
            T   x[N];
            if ( input_binary ) {
                for ( int i = 0; i < N; ++i) {
                    x[i] = 0.5 +((x_ptr[i][frame] > 0.5)  ? +unbinarize_bias : -unbinarize_bias);
                }
            }
            else {
                for ( int i = 0; i < N; ++i) {
                    x[i] = max(0.0, min(1.0, x_ptr[i][frame]));
                }
            }

            // read dy
            T   dy = dy_ptr[frame];

            // calc
            StochasticLut<N, T, MAX_NODE_UNIT>::NodeBackward(node_id, x, dy, &dx_buf[node*N*frame_stride + frame], W, dW, frame_stride);
#if 0
            T xp[6], xn[6];
            for ( int i = 0; i < 6; ++i) {
                T x_val = x_ptr[i][frame];
                if ( input_binary ) {
                    x_val = 0.5 + ((x_val > 0.5) ? +unbinarize_bias : -unbinarize_bias);
                }
                else {
                    x_val = min(1.0, max(0.0, x_val));
                }

                xp[i] = x_val;
                xn[i] = 1.0 - x_val;
            }

            T x0_00 = xn[1] * xn[0];
            T x0_01 = xn[1] * xp[0];
            T x0_10 = xp[1] * xn[0];
            T x0_11 = xp[1] * xp[0];
            T x1_00 = xn[3] * xn[2];
            T x1_01 = xn[3] * xp[2];
            T x1_10 = xp[3] * xn[2];
            T x1_11 = xp[3] * xp[2];
            T x2_00 = xn[5] * xn[4];
            T x2_01 = xn[5] * xp[4];
            T x2_10 = xp[5] * xn[4];
            T x2_11 = xp[5] * xp[4];

            T grad = dy_ptr[frame];

            T  x2_00_x1_00 =  x2_00 * x1_00;
            T  x2_00_x1_01 =  x2_00 * x1_01;
            T  x2_00_x1_10 =  x2_00 * x1_10;
            T  x2_00_x1_11 =  x2_00 * x1_11;
            T  x2_01_x1_00 =  x2_01 * x1_00;
            T  x2_01_x1_01 =  x2_01 * x1_01;
            T  x2_01_x1_10 =  x2_01 * x1_10;
            T  x2_01_x1_11 =  x2_01 * x1_11;
            T  x2_10_x1_00 =  x2_10 * x1_00;
            T  x2_10_x1_01 =  x2_10 * x1_01;
            T  x2_10_x1_10 =  x2_10 * x1_10;
            T  x2_10_x1_11 =  x2_10 * x1_11;
            T  x2_11_x1_00 =  x2_11 * x1_00;
            T  x2_11_x1_01 =  x2_11 * x1_01;
            T  x2_11_x1_10 =  x2_11 * x1_10;
            T  x2_11_x1_11 =  x2_11 * x1_11;

            dW[ 0] += x2_00_x1_00 * x0_00 * grad;
            dW[ 1] += x2_00_x1_00 * x0_01 * grad;
            dW[ 2] += x2_00_x1_00 * x0_10 * grad;
            dW[ 3] += x2_00_x1_00 * x0_11 * grad;
            dW[ 4] += x2_00_x1_01 * x0_00 * grad;
            dW[ 5] += x2_00_x1_01 * x0_01 * grad;
            dW[ 6] += x2_00_x1_01 * x0_10 * grad;
            dW[ 7] += x2_00_x1_01 * x0_11 * grad;
            dW[ 8] += x2_00_x1_10 * x0_00 * grad;
            dW[ 9] += x2_00_x1_10 * x0_01 * grad;
            dW[10] += x2_00_x1_10 * x0_10 * grad;
            dW[11] += x2_00_x1_10 * x0_11 * grad;
            dW[12] += x2_00_x1_11 * x0_00 * grad;
            dW[13] += x2_00_x1_11 * x0_01 * grad;
            dW[14] += x2_00_x1_11 * x0_10 * grad;
            dW[15] += x2_00_x1_11 * x0_11 * grad;
            dW[16] += x2_01_x1_00 * x0_00 * grad;
            dW[17] += x2_01_x1_00 * x0_01 * grad;
            dW[18] += x2_01_x1_00 * x0_10 * grad;
            dW[19] += x2_01_x1_00 * x0_11 * grad;
            dW[20] += x2_01_x1_01 * x0_00 * grad;
            dW[21] += x2_01_x1_01 * x0_01 * grad;
            dW[22] += x2_01_x1_01 * x0_10 * grad;
            dW[23] += x2_01_x1_01 * x0_11 * grad;
            dW[24] += x2_01_x1_10 * x0_00 * grad;
            dW[25] += x2_01_x1_10 * x0_01 * grad;
            dW[26] += x2_01_x1_10 * x0_10 * grad;
            dW[27] += x2_01_x1_10 * x0_11 * grad;
            dW[28] += x2_01_x1_11 * x0_00 * grad;
            dW[29] += x2_01_x1_11 * x0_01 * grad;
            dW[30] += x2_01_x1_11 * x0_10 * grad;
            dW[31] += x2_01_x1_11 * x0_11 * grad;
            dW[32] += x2_10_x1_00 * x0_00 * grad;
            dW[33] += x2_10_x1_00 * x0_01 * grad;
            dW[34] += x2_10_x1_00 * x0_10 * grad;
            dW[35] += x2_10_x1_00 * x0_11 * grad;
            dW[36] += x2_10_x1_01 * x0_00 * grad;
            dW[37] += x2_10_x1_01 * x0_01 * grad;
            dW[38] += x2_10_x1_01 * x0_10 * grad;
            dW[39] += x2_10_x1_01 * x0_11 * grad;
            dW[40] += x2_10_x1_10 * x0_00 * grad;
            dW[41] += x2_10_x1_10 * x0_01 * grad;
            dW[42] += x2_10_x1_10 * x0_10 * grad;
            dW[43] += x2_10_x1_10 * x0_11 * grad;
            dW[44] += x2_10_x1_11 * x0_00 * grad;
            dW[45] += x2_10_x1_11 * x0_01 * grad;
            dW[46] += x2_10_x1_11 * x0_10 * grad;
            dW[47] += x2_10_x1_11 * x0_11 * grad;
            dW[48] += x2_11_x1_00 * x0_00 * grad;
            dW[49] += x2_11_x1_00 * x0_01 * grad;
            dW[50] += x2_11_x1_00 * x0_10 * grad;
            dW[51] += x2_11_x1_00 * x0_11 * grad;
            dW[52] += x2_11_x1_01 * x0_00 * grad;
            dW[53] += x2_11_x1_01 * x0_01 * grad;
            dW[54] += x2_11_x1_01 * x0_10 * grad;
            dW[55] += x2_11_x1_01 * x0_11 * grad;
            dW[56] += x2_11_x1_10 * x0_00 * grad;
            dW[57] += x2_11_x1_10 * x0_01 * grad;
            dW[58] += x2_11_x1_10 * x0_10 * grad;
            dW[59] += x2_11_x1_10 * x0_11 * grad;
            dW[60] += x2_11_x1_11 * x0_00 * grad;
            dW[61] += x2_11_x1_11 * x0_01 * grad;
            dW[62] += x2_11_x1_11 * x0_10 * grad;
            dW[63] += x2_11_x1_11 * x0_11 * grad;

            T  x2_00_x0_00 =  x2_00 * x0_00;
            T  x2_00_x0_01 =  x2_00 * x0_01;
            T  x2_00_x0_10 =  x2_00 * x0_10;
            T  x2_00_x0_11 =  x2_00 * x0_11;
            T  x2_01_x0_00 =  x2_01 * x0_00;
            T  x2_01_x0_01 =  x2_01 * x0_01;
            T  x2_01_x0_10 =  x2_01 * x0_10;
            T  x2_01_x0_11 =  x2_01 * x0_11;
            T  x2_10_x0_00 =  x2_10 * x0_00;
            T  x2_10_x0_01 =  x2_10 * x0_01;
            T  x2_10_x0_10 =  x2_10 * x0_10;
            T  x2_10_x0_11 =  x2_10 * x0_11;
            T  x2_11_x0_00 =  x2_11 * x0_00;
            T  x2_11_x0_01 =  x2_11 * x0_01;
            T  x2_11_x0_10 =  x2_11 * x0_10;
            T  x2_11_x0_11 =  x2_11 * x0_11;

            T  x1_00_x0_00 =  x1_00 * x0_00;
            T  x1_00_x0_01 =  x1_00 * x0_01;
            T  x1_00_x0_10 =  x1_00 * x0_10;
            T  x1_00_x0_11 =  x1_00 * x0_11;
            T  x1_01_x0_00 =  x1_01 * x0_00;
            T  x1_01_x0_01 =  x1_01 * x0_01;
            T  x1_01_x0_10 =  x1_01 * x0_10;
            T  x1_01_x0_11 =  x1_01 * x0_11;
            T  x1_10_x0_00 =  x1_10 * x0_00;
            T  x1_10_x0_01 =  x1_10 * x0_01;
            T  x1_10_x0_10 =  x1_10 * x0_10;
            T  x1_10_x0_11 =  x1_10 * x0_11;
            T  x1_11_x0_00 =  x1_11 * x0_00;
            T  x1_11_x0_01 =  x1_11 * x0_01;
            T  x1_11_x0_10 =  x1_11 * x0_10;
            T  x1_11_x0_11 =  x1_11 * x0_11;


            T dxi;
            T dx0_00 = 0;
            T dx0_01 = 0;
            T dx0_10 = 0;
            T dx0_11 = 0;
            T dx1_00 = 0;
            T dx1_01 = 0;
            T dx1_10 = 0;
            T dx1_11 = 0;
            T dx2_00 = 0;
            T dx2_01 = 0;
            T dx2_10 = 0;
            T dx2_11 = 0;
            dxi = W[ 0][node_id];  dx0_00 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_00_x0_00;
            dxi = W[ 1][node_id];  dx0_01 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_00_x0_01;
            dxi = W[ 2][node_id];  dx0_10 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_00_x0_10;
            dxi = W[ 3][node_id];  dx0_11 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_00_x0_11;
            dxi = W[ 4][node_id];  dx0_00 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_01_x0_00;
            dxi = W[ 5][node_id];  dx0_01 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_01_x0_01;
            dxi = W[ 6][node_id];  dx0_10 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_01_x0_10;
            dxi = W[ 7][node_id];  dx0_11 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_01_x0_11;
            dxi = W[ 8][node_id];  dx0_00 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_10_x0_00;
            dxi = W[ 9][node_id];  dx0_01 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_10_x0_01;
            dxi = W[10][node_id];  dx0_10 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_10_x0_10;
            dxi = W[11][node_id];  dx0_11 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_10_x0_11;
            dxi = W[12][node_id];  dx0_00 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_11_x0_00;
            dxi = W[13][node_id];  dx0_01 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_11_x0_01;
            dxi = W[14][node_id];  dx0_10 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_11_x0_10;
            dxi = W[15][node_id];  dx0_11 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_11_x0_11;
            dxi = W[16][node_id];  dx0_00 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_00_x0_00;
            dxi = W[17][node_id];  dx0_01 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_00_x0_01;
            dxi = W[18][node_id];  dx0_10 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_00_x0_10;
            dxi = W[19][node_id];  dx0_11 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_00_x0_11;
            dxi = W[20][node_id];  dx0_00 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_01_x0_00;
            dxi = W[21][node_id];  dx0_01 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_01_x0_01;
            dxi = W[22][node_id];  dx0_10 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_01_x0_10;
            dxi = W[23][node_id];  dx0_11 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_01_x0_11;
            dxi = W[24][node_id];  dx0_00 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_10_x0_00;
            dxi = W[25][node_id];  dx0_01 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_10_x0_01;
            dxi = W[26][node_id];  dx0_10 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_10_x0_10;
            dxi = W[27][node_id];  dx0_11 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_10_x0_11;
            dxi = W[28][node_id];  dx0_00 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_11_x0_00;
            dxi = W[29][node_id];  dx0_01 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_11_x0_01;
            dxi = W[30][node_id];  dx0_10 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_11_x0_10;
            dxi = W[31][node_id];  dx0_11 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_11_x0_11;
            dxi = W[32][node_id];  dx0_00 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_00_x0_00;
            dxi = W[33][node_id];  dx0_01 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_00_x0_01;
            dxi = W[34][node_id];  dx0_10 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_00_x0_10;
            dxi = W[35][node_id];  dx0_11 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_00_x0_11;
            dxi = W[36][node_id];  dx0_00 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_01_x0_00;
            dxi = W[37][node_id];  dx0_01 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_01_x0_01;
            dxi = W[38][node_id];  dx0_10 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_01_x0_10;
            dxi = W[39][node_id];  dx0_11 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_01_x0_11;
            dxi = W[40][node_id];  dx0_00 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_10_x0_00;
            dxi = W[41][node_id];  dx0_01 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_10_x0_01;
            dxi = W[42][node_id];  dx0_10 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_10_x0_10;
            dxi = W[43][node_id];  dx0_11 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_10_x0_11;
            dxi = W[44][node_id];  dx0_00 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_11_x0_00;
            dxi = W[45][node_id];  dx0_01 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_11_x0_01;
            dxi = W[46][node_id];  dx0_10 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_11_x0_10;
            dxi = W[47][node_id];  dx0_11 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_11_x0_11;
            dxi = W[48][node_id];  dx0_00 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_00_x0_00;
            dxi = W[49][node_id];  dx0_01 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_00_x0_01;
            dxi = W[50][node_id];  dx0_10 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_00_x0_10;
            dxi = W[51][node_id];  dx0_11 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_00_x0_11;
            dxi = W[52][node_id];  dx0_00 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_01_x0_00;
            dxi = W[53][node_id];  dx0_01 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_01_x0_01;
            dxi = W[54][node_id];  dx0_10 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_01_x0_10;
            dxi = W[55][node_id];  dx0_11 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_01_x0_11;
            dxi = W[56][node_id];  dx0_00 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_10_x0_00;
            dxi = W[57][node_id];  dx0_01 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_10_x0_01;
            dxi = W[58][node_id];  dx0_10 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_10_x0_10;
            dxi = W[59][node_id];  dx0_11 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_10_x0_11;
            dxi = W[60][node_id];  dx0_00 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_11_x0_00;
            dxi = W[61][node_id];  dx0_01 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_11_x0_01;
            dxi = W[62][node_id];  dx0_10 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_11_x0_10;
            dxi = W[63][node_id];  dx0_11 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_11_x0_11;
        
            T *dx_ptr = &dx_buf[(node*6)*frame_stride + frame];
            T dxn;
            T dxp;
            T dx;
            dxn  = dx0_00 * xn[1];    dxn += dx0_10 * xp[1];
            dxp  = dx0_01 * xn[1];    dxp += dx0_11 * xp[1];
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[0 * frame_stride] = dx;

            dxn  = dx0_00 * xn[0];
            dxn += dx0_01 * xp[0];
            dxp  = dx0_10 * xn[0];
            dxp += dx0_11 * xp[0];
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[1 * frame_stride] = dx;

            dxn  = dx1_00 * xn[3];     
            dxp  = dx1_01 * xn[3];     
            dxn += dx1_10 * xp[3];     
            dxp += dx1_11 * xp[3];     
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[2 * frame_stride] = dx;

            dxn  = dx1_00 * xn[2];
            dxn += dx1_01 * xp[2];
            dxp  = dx1_10 * xn[2];
            dxp += dx1_11 * xp[2];
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[3 * frame_stride] = dx;

            dxn  = dx2_00 * xn[5];     
            dxp  = dx2_01 * xn[5];     
            dxn += dx2_10 * xp[5];     
            dxp += dx2_11 * xp[5];     
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[4 * frame_stride] = dx;

            dxn  = dx2_00 * xn[4];
            dxn += dx2_01 * xp[4];
            dxp  = dx2_10 * xn[4];
            dxp += dx2_11 * xp[4];
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[5 * frame_stride] = dx;
#endif
        }
    }

    for ( int i = 0; i < (1 << N); ++i ) {
        dW[i] = device_fp32_LocalSum(dW[i], sbuf[node_id]);
    }

    if ( node < node_size ) {
        if ( id == 0 ) {
            for ( int i = 0; i < (1 << N); ++i) {
                dW_buf[node*(1 << N) + i] = dW[i] + dW_prev[i][node_id];
            }
        }
    }
}

/*
__global__ void kernal_fp32_StochasticLut6_BackwardMarge(
            const float*    src_buf,
            float*          dst_buf,
            const int*      input_index,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;

    for ( int node = 0; node < node_size; ++node ) {
        if ( frame < frame_size ) {
            for ( int n = 0; n < 6; ++n ) {
                int in_idx = input_index[node*6 + n];
                float*       dst_buf_ptr = &dst_buf[frame_stride * in_idx];
                float        prev_data   = dst_buf_ptr[frame];
                const float* src_buf_ptr = &src_buf[(6 * node + n) * frame_stride];
                
                dst_buf_ptr[frame] = prev_data + src_buf_ptr[frame];
            }
        }
        __syncthreads();
    }
}
*/

int bbcu_fp32_StochasticLut6_Backward(
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *dev_dW,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             input_binary,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId
    )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    {
        unsigned int const THREAD_SIZE    = 256;
        unsigned int const MAX_FRAME_UNIT = 256;
        unsigned int const MAX_NODE_UNIT  = 16;

#if 0
        dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
        while ( (int)block.x / 2 >= frame_size )       { block.x /= 2; block.y *= 2; }
        while ( (int)block.y / 2 >= output_node_size ) { block.y /= 2; }
#else
        dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
        while ( (int)block.y / 2 >= output_node_size) { block.y /= 2; block.x *= 2;}
        while ( (int)block.x / 2 >= frame_size      ) { block.x /= 2; }
#endif

        block.x = std::min(block.x, MAX_FRAME_UNIT);
        block.y = std::min(block.y, MAX_NODE_UNIT);
        dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);

        kernal_StochasticLut_Backward<6, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
                dev_x_buf,
                dev_dy_buf,
                dev_dx_tmp,
                dev_input_index,
                dev_W,
                dev_dW,
                output_node_size,
                frame_size,
                frame_stride,
                input_binary,
                lut_binarize,
                unbinarize_bias
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }
    

    {
        BB_CUDA_SAFE_CALL(cudaMemset(dev_dx_buf, 0, input_node_size * frame_stride * sizeof(float)));

        int block_x = frame_size;
        while ( block_x > 1024 ) { block_x /= 2; }

        dim3    grid((frame_size + block_x - 1) /block_x, 1);
        dim3    block(block_x, 1, 1);

        /*
        kernal_fp32_StochasticLut6_BackwardMarge<<<grid, block>>>(
                dev_dx_tmp,
                dev_dx_buf,
                dev_input_index,
                output_node_size,
                frame_size,
                frame_stride
            );
        */

        kernal_NodeIntegrate<6, float><<<grid, block>>>(
                dev_dx_tmp,
                dev_dx_buf,
                dev_input_index,
                output_node_size,
                frame_size,
                frame_stride,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }

    return 0;
}



// bit packing
template<int N=6, typename T=float, int MAX_FRAME_UNIT=256, int MAX_NODE_UNIT=16>
__global__ void kernal_bit_StochasticLut_Backward
        (
            int const   *x_buf,
            T   const   *dy_buf,
            T           *dx_buf,
            int const   *input_index,
            T   const   *W_buf,
            T           *dW_buf,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         bin_frame_stride,
            int         lut_binarize,
            T           unbinarize_bias
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T       sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  T       dW_prev[(1 << N)][MAX_NODE_UNIT];
    __shared__  T       W[(1 << N)][MAX_NODE_UNIT];
                T       dW[(1 << N)];
                int   const *x_ptr[N];
                T const *dy_ptr;
    
    // initialize dW
    if ( node < node_size ) {
        for ( int i = 0; i < (1 << N); ++i) {
            dW[i] = 0;
        }

        for ( int i = id; i < (1 << N); i += id_step ) {
            dW_prev[i][node_id] = dW_buf[node * (1 << N) + i];
        }

        // read W
        for ( int i = id; i < (1 << N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1 << N) + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > 0.5 ? 1.0 : 0.0;
            }
        }
        
        // init pointer
        for ( int i = 0; i < N; ++i ) {
            int input_node = input_index[N*node + i];
            x_ptr[i]  = &x_buf[input_node * bin_frame_stride];
        }

        dy_ptr = &dy_buf[node * frame_stride];
    }

    __syncthreads();

    for ( int frame = id; frame < frame_size; frame += id_step ) {
        if ( node < node_size ) {
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);

            // read x
            T   x[N];
            for ( int i = 0; i < N; ++i) {
                x[i] = 0.5 +((x_ptr[i][unit] & bit) ? +unbinarize_bias : -unbinarize_bias);
            }

            // read dy
            T   dy = dy_ptr[frame];

            // calc
            StochasticLut<N, T, MAX_NODE_UNIT>::NodeBackward(node_id, x, dy, &dx_buf[node*N*frame_stride + frame], W, dW, frame_stride);

#if 0
            T   xp[6], xn[6];
            for ( int i = 0; i < 6; ++i) {
                T x_val = 0.5 + ((x_ptr[i][unit] & bit) ? +unbinarize_bias : -unbinarize_bias);
                xp[i] = x_val;
                xn[i] = 1.0 - x_val;
            }

            T x0_00 = xn[1] * xn[0];
            T x0_01 = xn[1] * xp[0];
            T x0_10 = xp[1] * xn[0];
            T x0_11 = xp[1] * xp[0];
            T x1_00 = xn[3] * xn[2];
            T x1_01 = xn[3] * xp[2];
            T x1_10 = xp[3] * xn[2];
            T x1_11 = xp[3] * xp[2];
            T x2_00 = xn[5] * xn[4];
            T x2_01 = xn[5] * xp[4];
            T x2_10 = xp[5] * xn[4];
            T x2_11 = xp[5] * xp[4];

            T grad = dy_ptr[frame];

            T  x2_00_x1_00 =  x2_00 * x1_00;
            T  x2_00_x1_01 =  x2_00 * x1_01;
            T  x2_00_x1_10 =  x2_00 * x1_10;
            T  x2_00_x1_11 =  x2_00 * x1_11;
            T  x2_01_x1_00 =  x2_01 * x1_00;
            T  x2_01_x1_01 =  x2_01 * x1_01;
            T  x2_01_x1_10 =  x2_01 * x1_10;
            T  x2_01_x1_11 =  x2_01 * x1_11;
            T  x2_10_x1_00 =  x2_10 * x1_00;
            T  x2_10_x1_01 =  x2_10 * x1_01;
            T  x2_10_x1_10 =  x2_10 * x1_10;
            T  x2_10_x1_11 =  x2_10 * x1_11;
            T  x2_11_x1_00 =  x2_11 * x1_00;
            T  x2_11_x1_01 =  x2_11 * x1_01;
            T  x2_11_x1_10 =  x2_11 * x1_10;
            T  x2_11_x1_11 =  x2_11 * x1_11;

            dW[ 0] += x2_00_x1_00 * x0_00 * grad;
            dW[ 1] += x2_00_x1_00 * x0_01 * grad;
            dW[ 2] += x2_00_x1_00 * x0_10 * grad;
            dW[ 3] += x2_00_x1_00 * x0_11 * grad;
            dW[ 4] += x2_00_x1_01 * x0_00 * grad;
            dW[ 5] += x2_00_x1_01 * x0_01 * grad;
            dW[ 6] += x2_00_x1_01 * x0_10 * grad;
            dW[ 7] += x2_00_x1_01 * x0_11 * grad;
            dW[ 8] += x2_00_x1_10 * x0_00 * grad;
            dW[ 9] += x2_00_x1_10 * x0_01 * grad;
            dW[10] += x2_00_x1_10 * x0_10 * grad;
            dW[11] += x2_00_x1_10 * x0_11 * grad;
            dW[12] += x2_00_x1_11 * x0_00 * grad;
            dW[13] += x2_00_x1_11 * x0_01 * grad;
            dW[14] += x2_00_x1_11 * x0_10 * grad;
            dW[15] += x2_00_x1_11 * x0_11 * grad;
            dW[16] += x2_01_x1_00 * x0_00 * grad;
            dW[17] += x2_01_x1_00 * x0_01 * grad;
            dW[18] += x2_01_x1_00 * x0_10 * grad;
            dW[19] += x2_01_x1_00 * x0_11 * grad;
            dW[20] += x2_01_x1_01 * x0_00 * grad;
            dW[21] += x2_01_x1_01 * x0_01 * grad;
            dW[22] += x2_01_x1_01 * x0_10 * grad;
            dW[23] += x2_01_x1_01 * x0_11 * grad;
            dW[24] += x2_01_x1_10 * x0_00 * grad;
            dW[25] += x2_01_x1_10 * x0_01 * grad;
            dW[26] += x2_01_x1_10 * x0_10 * grad;
            dW[27] += x2_01_x1_10 * x0_11 * grad;
            dW[28] += x2_01_x1_11 * x0_00 * grad;
            dW[29] += x2_01_x1_11 * x0_01 * grad;
            dW[30] += x2_01_x1_11 * x0_10 * grad;
            dW[31] += x2_01_x1_11 * x0_11 * grad;
            dW[32] += x2_10_x1_00 * x0_00 * grad;
            dW[33] += x2_10_x1_00 * x0_01 * grad;
            dW[34] += x2_10_x1_00 * x0_10 * grad;
            dW[35] += x2_10_x1_00 * x0_11 * grad;
            dW[36] += x2_10_x1_01 * x0_00 * grad;
            dW[37] += x2_10_x1_01 * x0_01 * grad;
            dW[38] += x2_10_x1_01 * x0_10 * grad;
            dW[39] += x2_10_x1_01 * x0_11 * grad;
            dW[40] += x2_10_x1_10 * x0_00 * grad;
            dW[41] += x2_10_x1_10 * x0_01 * grad;
            dW[42] += x2_10_x1_10 * x0_10 * grad;
            dW[43] += x2_10_x1_10 * x0_11 * grad;
            dW[44] += x2_10_x1_11 * x0_00 * grad;
            dW[45] += x2_10_x1_11 * x0_01 * grad;
            dW[46] += x2_10_x1_11 * x0_10 * grad;
            dW[47] += x2_10_x1_11 * x0_11 * grad;
            dW[48] += x2_11_x1_00 * x0_00 * grad;
            dW[49] += x2_11_x1_00 * x0_01 * grad;
            dW[50] += x2_11_x1_00 * x0_10 * grad;
            dW[51] += x2_11_x1_00 * x0_11 * grad;
            dW[52] += x2_11_x1_01 * x0_00 * grad;
            dW[53] += x2_11_x1_01 * x0_01 * grad;
            dW[54] += x2_11_x1_01 * x0_10 * grad;
            dW[55] += x2_11_x1_01 * x0_11 * grad;
            dW[56] += x2_11_x1_10 * x0_00 * grad;
            dW[57] += x2_11_x1_10 * x0_01 * grad;
            dW[58] += x2_11_x1_10 * x0_10 * grad;
            dW[59] += x2_11_x1_10 * x0_11 * grad;
            dW[60] += x2_11_x1_11 * x0_00 * grad;
            dW[61] += x2_11_x1_11 * x0_01 * grad;
            dW[62] += x2_11_x1_11 * x0_10 * grad;
            dW[63] += x2_11_x1_11 * x0_11 * grad;

            T  x2_00_x0_00 =  x2_00 * x0_00;
            T  x2_00_x0_01 =  x2_00 * x0_01;
            T  x2_00_x0_10 =  x2_00 * x0_10;
            T  x2_00_x0_11 =  x2_00 * x0_11;
            T  x2_01_x0_00 =  x2_01 * x0_00;
            T  x2_01_x0_01 =  x2_01 * x0_01;
            T  x2_01_x0_10 =  x2_01 * x0_10;
            T  x2_01_x0_11 =  x2_01 * x0_11;
            T  x2_10_x0_00 =  x2_10 * x0_00;
            T  x2_10_x0_01 =  x2_10 * x0_01;
            T  x2_10_x0_10 =  x2_10 * x0_10;
            T  x2_10_x0_11 =  x2_10 * x0_11;
            T  x2_11_x0_00 =  x2_11 * x0_00;
            T  x2_11_x0_01 =  x2_11 * x0_01;
            T  x2_11_x0_10 =  x2_11 * x0_10;
            T  x2_11_x0_11 =  x2_11 * x0_11;

            T  x1_00_x0_00 =  x1_00 * x0_00;
            T  x1_00_x0_01 =  x1_00 * x0_01;
            T  x1_00_x0_10 =  x1_00 * x0_10;
            T  x1_00_x0_11 =  x1_00 * x0_11;
            T  x1_01_x0_00 =  x1_01 * x0_00;
            T  x1_01_x0_01 =  x1_01 * x0_01;
            T  x1_01_x0_10 =  x1_01 * x0_10;
            T  x1_01_x0_11 =  x1_01 * x0_11;
            T  x1_10_x0_00 =  x1_10 * x0_00;
            T  x1_10_x0_01 =  x1_10 * x0_01;
            T  x1_10_x0_10 =  x1_10 * x0_10;
            T  x1_10_x0_11 =  x1_10 * x0_11;
            T  x1_11_x0_00 =  x1_11 * x0_00;
            T  x1_11_x0_01 =  x1_11 * x0_01;
            T  x1_11_x0_10 =  x1_11 * x0_10;
            T  x1_11_x0_11 =  x1_11 * x0_11;


            T dxi;
            T dx0_00 = 0;
            T dx0_01 = 0;
            T dx0_10 = 0;
            T dx0_11 = 0;
            T dx1_00 = 0;
            T dx1_01 = 0;
            T dx1_10 = 0;
            T dx1_11 = 0;
            T dx2_00 = 0;
            T dx2_01 = 0;
            T dx2_10 = 0;
            T dx2_11 = 0;
            dxi = W[ 0][node_id];  dx0_00 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_00_x0_00;
            dxi = W[ 1][node_id];  dx0_01 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_00_x0_01;
            dxi = W[ 2][node_id];  dx0_10 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_00_x0_10;
            dxi = W[ 3][node_id];  dx0_11 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_00_x0_11;
            dxi = W[ 4][node_id];  dx0_00 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_01_x0_00;
            dxi = W[ 5][node_id];  dx0_01 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_01_x0_01;
            dxi = W[ 6][node_id];  dx0_10 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_01_x0_10;
            dxi = W[ 7][node_id];  dx0_11 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_01_x0_11;
            dxi = W[ 8][node_id];  dx0_00 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_10_x0_00;
            dxi = W[ 9][node_id];  dx0_01 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_10_x0_01;
            dxi = W[10][node_id];  dx0_10 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_10_x0_10;
            dxi = W[11][node_id];  dx0_11 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_10_x0_11;
            dxi = W[12][node_id];  dx0_00 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_11_x0_00;
            dxi = W[13][node_id];  dx0_01 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_11_x0_01;
            dxi = W[14][node_id];  dx0_10 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_11_x0_10;
            dxi = W[15][node_id];  dx0_11 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_11_x0_11;
            dxi = W[16][node_id];  dx0_00 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_00_x0_00;
            dxi = W[17][node_id];  dx0_01 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_00_x0_01;
            dxi = W[18][node_id];  dx0_10 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_00_x0_10;
            dxi = W[19][node_id];  dx0_11 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_00_x0_11;
            dxi = W[20][node_id];  dx0_00 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_01_x0_00;
            dxi = W[21][node_id];  dx0_01 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_01_x0_01;
            dxi = W[22][node_id];  dx0_10 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_01_x0_10;
            dxi = W[23][node_id];  dx0_11 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_01_x0_11;
            dxi = W[24][node_id];  dx0_00 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_10_x0_00;
            dxi = W[25][node_id];  dx0_01 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_10_x0_01;
            dxi = W[26][node_id];  dx0_10 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_10_x0_10;
            dxi = W[27][node_id];  dx0_11 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_10_x0_11;
            dxi = W[28][node_id];  dx0_00 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_11_x0_00;
            dxi = W[29][node_id];  dx0_01 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_11_x0_01;
            dxi = W[30][node_id];  dx0_10 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_11_x0_10;
            dxi = W[31][node_id];  dx0_11 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_11_x0_11;
            dxi = W[32][node_id];  dx0_00 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_00_x0_00;
            dxi = W[33][node_id];  dx0_01 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_00_x0_01;
            dxi = W[34][node_id];  dx0_10 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_00_x0_10;
            dxi = W[35][node_id];  dx0_11 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_00_x0_11;
            dxi = W[36][node_id];  dx0_00 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_01_x0_00;
            dxi = W[37][node_id];  dx0_01 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_01_x0_01;
            dxi = W[38][node_id];  dx0_10 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_01_x0_10;
            dxi = W[39][node_id];  dx0_11 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_01_x0_11;
            dxi = W[40][node_id];  dx0_00 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_10_x0_00;
            dxi = W[41][node_id];  dx0_01 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_10_x0_01;
            dxi = W[42][node_id];  dx0_10 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_10_x0_10;
            dxi = W[43][node_id];  dx0_11 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_10_x0_11;
            dxi = W[44][node_id];  dx0_00 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_11_x0_00;
            dxi = W[45][node_id];  dx0_01 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_11_x0_01;
            dxi = W[46][node_id];  dx0_10 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_11_x0_10;
            dxi = W[47][node_id];  dx0_11 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_11_x0_11;
            dxi = W[48][node_id];  dx0_00 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_00_x0_00;
            dxi = W[49][node_id];  dx0_01 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_00_x0_01;
            dxi = W[50][node_id];  dx0_10 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_00_x0_10;
            dxi = W[51][node_id];  dx0_11 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_00_x0_11;
            dxi = W[52][node_id];  dx0_00 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_01_x0_00;
            dxi = W[53][node_id];  dx0_01 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_01_x0_01;
            dxi = W[54][node_id];  dx0_10 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_01_x0_10;
            dxi = W[55][node_id];  dx0_11 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_01_x0_11;
            dxi = W[56][node_id];  dx0_00 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_10_x0_00;
            dxi = W[57][node_id];  dx0_01 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_10_x0_01;
            dxi = W[58][node_id];  dx0_10 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_10_x0_10;
            dxi = W[59][node_id];  dx0_11 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_10_x0_11;
            dxi = W[60][node_id];  dx0_00 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_11_x0_00;
            dxi = W[61][node_id];  dx0_01 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_11_x0_01;
            dxi = W[62][node_id];  dx0_10 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_11_x0_10;
            dxi = W[63][node_id];  dx0_11 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_11_x0_11;
        
            T *dx_ptr = &dx_buf[(node*6)*frame_stride + frame];
            T dxn;
            T dxp;
            T dx;
            dxn  = dx0_00 * xn[1];    dxn += dx0_10 * xp[1];
            dxp  = dx0_01 * xn[1];    dxp += dx0_11 * xp[1];
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[0 * frame_stride] = dx;

            dxn  = dx0_00 * xn[0];
            dxn += dx0_01 * xp[0];
            dxp  = dx0_10 * xn[0];
            dxp += dx0_11 * xp[0];
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[1 * frame_stride] = dx;

            dxn  = dx1_00 * xn[3];     
            dxp  = dx1_01 * xn[3];     
            dxn += dx1_10 * xp[3];     
            dxp += dx1_11 * xp[3];     
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[2 * frame_stride] = dx;

            dxn  = dx1_00 * xn[2];
            dxn += dx1_01 * xp[2];
            dxp  = dx1_10 * xn[2];
            dxp += dx1_11 * xp[2];
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[3 * frame_stride] = dx;

            dxn  = dx2_00 * xn[5];     
            dxp  = dx2_01 * xn[5];     
            dxn += dx2_10 * xp[5];     
            dxp += dx2_11 * xp[5];     
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[4 * frame_stride] = dx;

            dxn  = dx2_00 * xn[4];
            dxn += dx2_01 * xp[4];
            dxp  = dx2_10 * xn[4];
            dxp += dx2_11 * xp[4];
            dx = (dxp - dxn) * grad;
            if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
            dx_ptr[5 * frame_stride] = dx;
#endif
        }
    }

    for ( int i = 0; i < (1 << N); ++i ) {
        dW[i] = device_fp32_LocalSum(dW[i], sbuf[node_id]);
    }

    if ( node < node_size ) {
        if ( id == 0 ) {
            for ( int i = 0; i < (1 << N); ++i) {
                dW_buf[node*(1 << N) + i] = dW[i] + dW_prev[i][node_id];
            }
        }
    }
}

int bbcu_bit_fp32_StochasticLut6_Backward(
            int   const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *dev_dW,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             bin_frame_stride,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId
    )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    {
        unsigned int const THREAD_SIZE    = 256;
        unsigned int const MAX_FRAME_UNIT = 256;
        unsigned int const MAX_NODE_UNIT  = 16;

#if 0
        dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
        while ( (int)block.x / 2 >= frame_size )       { block.x /= 2; block.y *= 2; }
        while ( (int)block.y / 2 >= output_node_size ) { block.y /= 2; }
#else
        dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
        while ( (int)block.y / 2 >= output_node_size) { block.y /= 2; block.x *= 2;}
        while ( (int)block.x / 2 >= frame_size      ) { block.x /= 2; }
#endif

        block.x = std::min(block.x, MAX_FRAME_UNIT);
        block.y = std::min(block.y, MAX_NODE_UNIT);
        dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);

        kernal_bit_StochasticLut_Backward<6, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
                dev_x_buf,
                dev_dy_buf,
                dev_dx_tmp,
                dev_input_index,
                dev_W,
                dev_dW,
                output_node_size,
                frame_size,
                frame_stride,
                bin_frame_stride,
                lut_binarize,
                unbinarize_bias
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }
    

    {
        BB_CUDA_SAFE_CALL(cudaMemset(dev_dx_buf, 0, input_node_size * frame_stride * sizeof(float)));

        int block_x = frame_size;
        while ( block_x > 1024 ) { block_x /= 2; }

        dim3    grid((frame_size + block_x - 1) /block_x, 1);
        dim3    block(block_x, 1, 1);
        /*
        kernal_fp32_StochasticLut6_BackwardMarge<<<grid, block>>>(
                dev_dx_tmp,
                dev_dx_buf,
                dev_input_index,
                output_node_size,
                frame_size,
                frame_stride
            );
        */
        
        kernal_NodeIntegrate<6, float><<<grid, block>>>(
                dev_dx_tmp,
                dev_dx_buf,
                dev_input_index,
                output_node_size,
                frame_size,
                frame_stride,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }

    return 0;
}




// end of file
