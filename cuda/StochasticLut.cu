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

            // clamp
            y = max(0.0, y);
            y = min(1.0, y);
        
            y_ptr[frame] = y;
        }
    }
}


template <int N>
BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Forward
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
    
    kernal_StochasticLut_Forward<N, float, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
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
            
            // read x
            T   x[N];
            for ( int i = 0; i < N; ++i) {
                x[i] = 0.5 + ((x_ptr[i][unit] & bit_mask) ? +unbinarize_bias : -unbinarize_bias);
            }

            // calculate
            T   y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

            // clamp
            y = max(0.0, y);
            y = min(1.0, y);

            y_ptr[frame] = y;
        }
    }
}


template <int N>
BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Forward
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
    
    kernal_bit_StochasticLut_Forward<N, float, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
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



// bit packing and binarize
template<int N=6, typename T=float, int MAX_NODE_UNIT=32>
__global__ void kernal_bit_bit_StochasticLut_Forward(
            int const   *x_buf,
            int         *y_buf,
            int const   *input_index,
            T   const   *W_buf,
            int         node_size,
            int         frame_size,
            int         frame_stride,
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
    int             *y_ptr;
    
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
            x_ptr[i] = &x_buf[frame_stride * input_index[N*node + i]];
        }

        y_ptr = &y_buf[node * frame_stride];
    }

    __syncthreads();
    
    T   unbinarize_hi = 0.5 + unbinarize_bias;
    T   unbinarize_lo = 0.5 - unbinarize_bias;

    if ( node < node_size ) {
        int frame_unit_size = ((frame_size + 0x1f) & ~0x1f);
        for (int frame = id; frame < frame_unit_size; frame += id_step) {
            int y_mask   = 0;
            int unit     = (frame >> 5);
            int bit      = (frame & 0x1f);
            int bit_mask = (1 << bit);
            if ( frame < frame_size ) {
                // read x
                T   x[N];
                for ( int i = 0; i < N; ++i) {
                    x[i] = ((x_ptr[i][unit] & bit_mask) ? unbinarize_hi : unbinarize_lo);
                }

                // calculate
                T   y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

                // binarize
                if ( y > 0.5 ) {
                    y_mask = bit_mask;
                }
            }

            // OR
            y_mask = device_int_ShuffleOr(y_mask);

            if ( bit == 0 ) {
                y_ptr[unit] = y_mask;
            }
        }
    }
}


template <int N>
BBCU_DLL_EXPORT int bbcu_bit_bit_fp32_StochasticLut_Forward
        (
            int   const     *dev_x_buf,
            int             *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 512;
    unsigned int const MAX_FRAME_UNIT = 512;
    unsigned int const MAX_NODE_UNIT  = THREAD_SIZE / 32;

#if 0
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size && block.x > 32 ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size                  ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size                  ) { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size && block.x > 32 ) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (node_size + (block.y - 1)) / block.y);
    
    kernal_bit_bit_StochasticLut_Forward<N, float, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            node_size,
            frame_size,
            frame_stride,
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
            int         dx_frame_stride,
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
                T const *x_ptr[N];
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

            // calculate
            StochasticLut<N, T, MAX_NODE_UNIT>::NodeBackward(node_id, x, dy, &dx_buf[node*N*dx_frame_stride + frame], W, dW, dx_frame_stride);
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


template <int N>
BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            int   const     *dev_reverse_index,
            float const     *dev_W,
            float           *dev_dW,
            int             reverse_index_stride,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             tmp_frame_size,
            int             tmp_frame_stride,
            int             input_binary,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    int frame_offset = 0;
    do {
        int unit_frame_size = frame_size - frame_offset;
        if (unit_frame_size > tmp_frame_size) {
            unit_frame_size = tmp_frame_size;
        }

        {
            unsigned int const THREAD_SIZE    = 256;
            unsigned int const MAX_FRAME_UNIT = 256;
            unsigned int const MAX_NODE_UNIT  = 16;

    #if 0
            dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
            while ( (int)block.x / 2 >= unit_frame_size  ) { block.x /= 2; block.y *= 2; }
            while ( (int)block.y / 2 >= output_node_size ) { block.y /= 2; }
    #else
            dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
            while ( (int)block.y / 2 >= output_node_size ) { block.y /= 2; block.x *= 2;}
            while ( (int)block.x / 2 >= unit_frame_size  ) { block.x /= 2; }
    #endif

            block.x = std::min(block.x, MAX_FRAME_UNIT);
            block.y = std::min(block.y, MAX_NODE_UNIT);
            dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);

            kernal_StochasticLut_Backward<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
                    dev_x_buf  + frame_offset,
                    dev_dy_buf + frame_offset,
                    dev_dx_tmp,
                    dev_input_index,
                    dev_W,
                    dev_dW,
                    output_node_size,
                    unit_frame_size,
                    frame_stride,
                    tmp_frame_stride,
                    input_binary,
                    lut_binarize,
                    unbinarize_bias
                );
            BB_CUDA_CHECK_LAST_ERROR();
        }
    

        {
            unsigned int const THREAD_SIZE    = 1024;
            unsigned int const MAX_FRAME_UNIT = 1024;
            unsigned int const MAX_NODE_UNIT  = 1024;

    #if 1
            dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
            while ( (int)block.x / 2 >= unit_frame_size ) { block.x /= 2; block.y *= 2; }
            while ( (int)block.y / 2 >= input_node_size ) { block.y /= 2; }
    #else
            dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
            while ( (int)block.y / 2 >= input_node_size ) { block.y /= 2; block.x *= 2;}
            while ( (int)block.x / 2 >= unit_frame_size ) { block.x /= 2; }
    #endif

            block.x = std::min(block.x, MAX_FRAME_UNIT);
            block.y = std::min(block.y, MAX_NODE_UNIT);
            dim3    grid((unit_frame_size + (block.x - 1)) / block.x, (input_node_size + (block.y - 1)) / block.y);

            kernal_NodeIntegrateWithTable<float><<<grid, block>>>
                (
                    dev_dx_tmp,
                    dev_dx_buf + frame_offset,
                    dev_reverse_index,
                    reverse_index_stride,
                    input_node_size,
                    unit_frame_size,
                    tmp_frame_stride,
                    frame_stride
                );
            BB_CUDA_CHECK_LAST_ERROR();
        }

        frame_offset += unit_frame_size;
    } while ( frame_offset < frame_size );

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
            int         x_frame_stride,
            int         dy_frame_stride,
            int         dx_frame_stride,
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
            x_ptr[i]  = &x_buf[input_node * x_frame_stride];
        }

        dy_ptr = &dy_buf[node * dy_frame_stride];
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

            // calculate
            StochasticLut<N, T, MAX_NODE_UNIT>::NodeBackward(node_id, x, dy, &dx_buf[node*N*dx_frame_stride + frame], W, dW, dx_frame_stride);
        }
    }

    // write dW
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


template <int N>
BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Backward
        (
            int   const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            int   const     *dev_reverse_index,
            float const     *dev_W,
            float           *dev_dW,
            int             reverse_index_stride,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             bin_frame_stride,
            int             tmp_frame_size,
            int             tmp_frame_stride,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    int frame_offset = 0;
    do {
        int unit_frame_size = frame_size - frame_offset;
        if (unit_frame_size > tmp_frame_size) {
            unit_frame_size = tmp_frame_size;
        }

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

            kernal_bit_StochasticLut_Backward<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
                    dev_x_buf  + (frame_offset / 32),
                    dev_dy_buf + frame_offset,
                    dev_dx_tmp,
                    dev_input_index,
                    dev_W,
                    dev_dW,
                    output_node_size,
                    unit_frame_size,
                    bin_frame_stride,
                    frame_stride,
                    tmp_frame_stride,
                    lut_binarize,
                    unbinarize_bias
                );
            BB_CUDA_CHECK_LAST_ERROR();
        }
    

        {
            unsigned int const THREAD_SIZE    = 1024;
            unsigned int const MAX_FRAME_UNIT = 1024;
            unsigned int const MAX_NODE_UNIT  = 1024;

    #if 1
            dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
            while ( (int)block.x / 2 >= unit_frame_size ) { block.x /= 2; block.y *= 2; }
            while ( (int)block.y / 2 >= input_node_size ) { block.y /= 2; }
    #else
            dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
            while ( (int)block.y / 2 >= input_node_size ) { block.y /= 2; block.x *= 2;}
            while ( (int)block.x / 2 >= unit_frame_size ) { block.x /= 2; }
    #endif

            block.x = std::min(block.x, MAX_FRAME_UNIT);
            block.y = std::min(block.y, MAX_NODE_UNIT);
            dim3    grid((unit_frame_size + (block.x - 1)) / block.x, (input_node_size + (block.y - 1)) / block.y);

            kernal_NodeIntegrateWithTable<float><<<grid, block>>>
                (
                    dev_dx_tmp,
                    dev_dx_buf + frame_offset,
                    dev_reverse_index,
                    reverse_index_stride,
                    input_node_size,
                    unit_frame_size,
                    tmp_frame_stride,
                    frame_stride
                );
            BB_CUDA_CHECK_LAST_ERROR();
        }

        frame_offset += unit_frame_size;
    } while ( frame_offset < frame_size );
    

    return 0;
}



// 実体化
template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Forward<6>(const float *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Forward<5>(const float *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Forward<4>(const float *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Forward<3>(const float *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Forward<2>(const float *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);

template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Forward<6>(int const *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int bin_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Forward<5>(int const *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int bin_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Forward<4>(int const *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int bin_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Forward<3>(int const *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int bin_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Forward<2>(int const *dev_x_buf, float *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int bin_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);

template BBCU_DLL_EXPORT int bbcu_bit_bit_fp32_StochasticLut_Forward<6>(int const *dev_x_buf, int *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_bit_fp32_StochasticLut_Forward<5>(int const *dev_x_buf, int *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_bit_fp32_StochasticLut_Forward<4>(int const *dev_x_buf, int *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_bit_fp32_StochasticLut_Forward<3>(int const *dev_x_buf, int *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_bit_fp32_StochasticLut_Forward<2>(int const *dev_x_buf, int *dev_y_buf, int const *dev_input_index, float const *dev_W, int node_size, int frame_size, int frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);

template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Backward<6>(float const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int tmp_frame_size, int tmp_frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Backward<5>(float const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int tmp_frame_size, int tmp_frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Backward<4>(float const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int tmp_frame_size, int tmp_frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Backward<3>(float const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int tmp_frame_size, int tmp_frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Backward<2>(float const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int tmp_frame_size, int tmp_frame_stride, int input_binary, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);

template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Backward<6>(int const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int bin_frame_stride, int tmp_frame_size, int tmp_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Backward<5>(int const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int bin_frame_stride, int tmp_frame_size, int tmp_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Backward<4>(int const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int bin_frame_stride, int tmp_frame_size, int tmp_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Backward<3>(int const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int bin_frame_stride, int tmp_frame_size, int tmp_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Backward<2>(int const *dev_x_buf, float const *dev_dy_buf, float *dev_dx_buf, float *dev_dx_tmp, int const *dev_input_index, int const *dev_reverse_index, float const *dev_W, float *dev_dW, int reverse_index_stride, int input_node_size, int output_node_size, int frame_size, int frame_stride, int bin_frame_stride, int tmp_frame_size, int tmp_frame_stride, int lut_binarize, float unbinarize_bias, cudaStream_t streamId);


// end of file
