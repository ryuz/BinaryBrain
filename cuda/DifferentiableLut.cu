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
template<int N=6, typename T=float, int MAX_FRAME_UNIT=32, int MAX_NODE_UNIT=32>
__global__ void kernal_DifferentiableLut_ForwardTraining
        (
            T   const   *x_buf,
            T           *y_buf,
            int const   *input_index,
            T   const   *W_buf,
            T           *mean_buf,
            T           *rstd_buf,
            T           *running_mean_buf,
            T           *running_var_buf,
            T           gamma,
            T           beta,
            T           momentum,
            T           unbinarize_bias,
            T           reciprocal_frame_size,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         lut_binarize,
            int         binary_mode
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T           sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];

    __shared__  T           W[(1 << N)][MAX_NODE_UNIT];
                T   const   *x_ptr[N];
                T           *y_ptr;
    
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < (1 << N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1 << N) + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > (T)0.5 ? (T)1.0 : (T)0.0;
            }
        }
        
        // read input index
        for ( int i = 0; i < N; ++i ) {
            x_ptr[i] = &x_buf[frame_stride * input_index[N*node + i]];
        }
        
        y_ptr = &y_buf[node * frame_stride];
    }

    __syncthreads();
    
    // 平均と分散計測
    T s1 = 0, c1 = 0, y1, t1;
    T s2 = 0, c2 = 0, y2, t2;
    for (int frame = id; frame < frame_size; frame += id_step) {
        if ( node < node_size ) {
            // Forward計算
            T x[N];
            if ( binary_mode ) {
                for ( int i = 0; i < N; ++i) {
                    x[i] = (T)0.5 + ((x_ptr[i][frame] > (T)0.5) ? +unbinarize_bias : -unbinarize_bias);
                }
            }
            else {
                for ( int i = 0; i < N; ++i) {
                    x[i] = max(0.0, min((T)1.0, x_ptr[i][frame]));
                }
            }

            T y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

            // 集計
            y1 = y - c1;
            t1 = s1 + y1;
            c1 = (t1 - s1) - y1;
            s1 = t1;

            y2 = (y * y) - c2;
            t2 = s2 + y2;
            c2 = (t2 - s2) - y2;
            s2 = t2;
        }
    }

    s1 = device_fp32_LocalSum(s1, sbuf[node_id]);
    s2 = device_fp32_LocalSum(s2, sbuf[node_id]);
    T mean = s1 * reciprocal_frame_size;
    T var = max(1.0e-5f, (s2 * reciprocal_frame_size) - (mean * mean));
  
    T rstd = rsqrt(var);

    // 書き込み
    if (id == 0) {
        if ( node < node_size ) {
            running_mean_buf[node] = running_mean_buf[node] * momentum + mean * ((T)1.0 - momentum);
            running_var_buf[node]  = running_var_buf[node] * momentum + var * ((T)1.0 - momentum);
            mean_buf[node] = mean;
            rstd_buf[node] = rstd;
        }
    }

    // 正規化
    for ( int frame = id; frame < frame_size; frame += id_step) {
        if ( node < node_size ) {
            // Forward計算
            T x[N];
            if ( binary_mode ) {
                for ( int i = 0; i < N; ++i) {
                    x[i] = 0.5 + ((x_ptr[i][frame] > (T)0.5) ? +unbinarize_bias : -unbinarize_bias);
                }
            }
            else {
                for ( int i = 0; i < N; ++i) {
                    x[i] = max(0.0, min(1.0, x_ptr[i][frame]));
                }
            }

            T y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

            y = (y - mean) * rstd;
            y = y * gamma + beta;

            if ( binary_mode ) {
                // binarize
                y = (y > (T)0.5) ? (T)1.0 : (T)0.0;
            }
            else {
                // hard-tanh
                y = min(y, (T)1.0);
                y = max(y, (T)0.0);
            }

            y_ptr[frame] = y;
        }
    }
}


template <int N>
BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardTraining
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *dev_mean_buf,
            float           *dev_rstd_buf,
            float           *dev_running_mean_buf,
            float           *dev_running_var_buf,
            float           gamma,
            float           beta,
            float           momentum,
            float           unbinarize_bias,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            int             binary_mode,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 256;
    unsigned int const MAX_FRAME_UNIT = 256;
    unsigned int const MAX_NODE_UNIT  = 16;

#if 0
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size && block.x > 32 ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size                  ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size  )                { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size && block.x > 32) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (node_size + (block.y - 1)) / block.y);
    
    kernal_DifferentiableLut_ForwardTraining<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            dev_mean_buf,
            dev_rstd_buf,
            dev_running_mean_buf,
            dev_running_var_buf,
            gamma,
            beta,
            momentum,
            unbinarize_bias,
            1.0f / (float)frame_size,
            node_size,
            frame_size,
            frame_stride,
            lut_binarize,
            binary_mode
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}



// bit packing binary
template<int N=6, typename T=float, int MAX_FRAME_UNIT=32, int MAX_NODE_UNIT=32>
__global__ void kernal_bit_DifferentiableLut_ForwardTraining
        (
            int const   *x_buf,
            int         *y_buf,
            int const   *input_index,
            T   const   *W_buf,
            T           *mean_buf,
            T           *rstd_buf,
            T           *running_mean_buf,
            T           *running_var_buf,
            T           gamma,
            T           beta,
            T           momentum,
            T           unbinarize_bias,
            T           reciprocal_frame_size,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         lut_binarize
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T           sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];

    __shared__  T           W[(1 << N)][MAX_NODE_UNIT];
                int   const *x_ptr[N];
                int         *y_ptr;
    
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < (1 << N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1 << N) + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > (T)0.5 ? (T)1.0 : (T)0.0;
            }
        }
        
        // read input index
        for ( int i = 0; i < N; ++i ) {
            x_ptr[i] = &x_buf[frame_stride * input_index[N*node + i]];
        }
                     
        y_ptr = &y_buf[node * frame_stride];
    }

    __syncthreads();
    
    // 平均と分散計測
    T s1 = 0, c1 = 0, y1, t1;
    T s2 = 0, c2 = 0, y2, t2;
    for (int frame = id; frame < frame_size; frame += id_step) {
        if ( node < node_size ) {
            // Forward計算
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);
            T x[N];
            for ( int i = 0; i < N; ++i) {
                x[i] = (T)0.5 + ((x_ptr[i][unit] & bit) ? +unbinarize_bias : -unbinarize_bias);
            }
            T y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);
//          printf("[0] n=%3d f=%3d y=%10f\n", node, frame, y);

            // 集計
            y1 = y - c1;
            t1 = s1 + y1;
            c1 = (t1 - s1) - y1;
            s1 = t1;

            y2 = (y * y) - c2;
            t2 = s2 + y2;
            c2 = (t2 - s2) - y2;
            s2 = t2;
        }
    }

    s1 = device_LocalSumX<float>(s1, sbuf[node_id]);
    s2 = device_LocalSumX<float>(s2, sbuf[node_id]);

    T mean = s1 * reciprocal_frame_size;
    T var = max(1.0e-5f, (s2 * reciprocal_frame_size) - (mean * mean));
    T rstd = rsqrt(var);

//  if ( node < node_size && id == 0 ) {
////      printf("[0] n=%3d s1=%10f s2=%10f mean=%10f var=%10f rstd=%10f\n", node, s1, s2, mean, var, rstd);
//      printf("0\t%3d\t%.20e\t%.20e\t%.20e\t%.20e\t%.20e\n", node, s1, s2, mean, var, rstd);
//  }

    // 書き込み
    if (id == 0) {
        if ( node < node_size ) {
            running_mean_buf[node] = running_mean_buf[node] * momentum + mean * ((T)1.0 - momentum);
            running_var_buf[node]  = running_var_buf[node] * momentum + var * ((T)1.0 - momentum);
            mean_buf[node] = mean;
            rstd_buf[node] = rstd;
        }
    }

    // 正規化
    int loop_size = ((frame_size + blockDim.x - 1) & ~(blockDim.x - 1));
    for ( int frame = id; frame < loop_size; frame += id_step) {
        int unit     = (frame >> 5);
        int bit      = (frame & 0x1f);
        int bit_mask = (1 << bit);

        int y_mask = 0;
        if ( node < node_size && frame < frame_size) {
            // Forward計算
            T x[N];
            for ( int i = 0; i < N; ++i) {
                x[i] = (T)0.5 + ((x_ptr[i][unit] & bit_mask)  ? +unbinarize_bias : -unbinarize_bias);
            }
            T y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

            y = (y - mean) * rstd;
            y = y * gamma + beta;

            if ( y > (T)0.5 ) {
                y_mask = bit_mask;
            }
        }

        y_mask = device_int_ShuffleOr(y_mask);

        if ( bit == 0 ) {
            if ( node < node_size && frame < frame_size ) {
                y_ptr[unit] = y_mask;
            }
        }
    }
}


template <int N>
BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardTraining
        (
            int   const     *dev_x_buf,
            int             *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *dev_mean_buf,
            float           *dev_rstd_buf,
            float           *dev_running_mean_buf,
            float           *dev_running_var_buf,
            float           gamma,
            float           beta,
            float           momentum,
            float           unbinarize_bias,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 256;
    unsigned int const MAX_FRAME_UNIT = 256;
    unsigned int const MAX_NODE_UNIT  = 8;  // THREAD_SIZE/32 より小さくすること

#if 0
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size && block.x > 32 ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size                  ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size  )                { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size && block.x > 32) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (node_size + (block.y - 1)) / block.y);
    
    kernal_bit_DifferentiableLut_ForwardTraining<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            dev_mean_buf,
            dev_rstd_buf,
            dev_running_mean_buf,
            dev_running_var_buf,
            gamma,
            beta,
            momentum,
            unbinarize_bias,
            1.0f / (float)frame_size,
            node_size,
            frame_size,
            frame_stride,
            lut_binarize
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}



// -------------------------------------------------
//  Forward Inference
// -------------------------------------------------

// real type
template<int N=6, typename T=float, int MAX_FRAME_UNIT=32, int MAX_NODE_UNIT=32>
__global__ void kernal_DifferentiableLut_ForwardInference
        (
            T   const   *x_buf,
            T           *y_buf,
            int const   *input_index,
            T   const   *W_buf,
            T   const   *running_mean_buf,
            T   const   *running_var_buf,
            T           gamma,
            T           beta,
            T           unbinarize_bias,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         lut_binarize,
            int         binary_mode
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T           W[(1 << N)][MAX_NODE_UNIT];
                T   const   *x_ptr[N];
                T           *y_ptr;
    
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < (1 << N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1 << N) + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > (T)0.5 ? (T)1.0 : (T)0.0;
            }
        }
        
        // read input index
        for ( int i = 0; i < N; ++i ) {
            x_ptr[i] = &x_buf[frame_stride * input_index[N * node + i]];
        }
        
        y_ptr = &y_buf[node * frame_stride];
    }

    __syncthreads();
    
    if ( node < node_size ) {
        T   mean  = running_mean_buf[node];
        T   var   = running_var_buf[node];
        T   rstd = (T)1.0 / (sqrt(var) + (T)1.0e-7);

        for ( int frame = id; frame < frame_size; frame += id_step) {
            // Forward計算
            T   x[N];
            if ( binary_mode ) {
                for ( int i = 0; i < N; ++i) {
                    x[i] = (T)0.5 + ((x_ptr[i][frame] > (T)0.5) ? +unbinarize_bias : -unbinarize_bias);
                }
            }
            else {
                for ( int i = 0; i < N; ++i) {
                    x[i] = max((T)0.0, min((T)1.0, x_ptr[i][frame]));
                }
            }

            T   y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

            y = ((y - mean) * rstd) * gamma + beta;

            if ( binary_mode ) {
                y = (y > (T)0.5) ? (T)1.0 : (T)0.0;
            }

            y_ptr[frame] = y;
        }
    }
}


template <int N>
BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardInference
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float const     *running_mean_buf,
            float const     *running_var_buf,
            float           gamma,
            float           beta,
            float           unbinarize_bias,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            int             binary_mode,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 256;
    unsigned int const MAX_FRAME_UNIT = 256;
    unsigned int const MAX_NODE_UNIT  = 8;  // THREAD_SIZE/32 より小さくすること

#if 0
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size && block.x > 32 ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size                  ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size  )                { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size && block.x > 32) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (node_size + (block.y - 1)) / block.y);
    
    kernal_DifferentiableLut_ForwardInference<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            running_mean_buf,
            running_var_buf,
            gamma,
            beta,
            unbinarize_bias,
            node_size,
            frame_size,
            frame_stride,
            lut_binarize,
            binary_mode
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}



// bit packing binary
template<int N=6, typename T=float, int MAX_FRAME_UNIT=32, int MAX_NODE_UNIT=32>
__global__ void kernal_bit_DifferentiableLut_ForwardInference
        (
            int const   *x_buf,
            int         *y_buf,
            int const   *input_index,
            T   const   *W_buf,
            T   const   *running_mean_buf,
            T   const   *running_var_buf,
            T           gamma,
            T           beta,
            T           unbinarize_bias,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         lut_binarize
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T           W[(1 << N)][MAX_NODE_UNIT];
                int const   *x_ptr[N];
                int         *y_ptr;
    
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < (1 << N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1 << N) + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > (T)0.5 ? (T)1.0 : (T)0.0;
            }
        }
        
        // read input index
        for ( int i = 0; i < N; ++i ) {
            x_ptr[i] = &x_buf[frame_stride * input_index[N * node + i]];
        }
                     
        y_ptr = &y_buf[node * frame_stride];
    }

    __syncthreads();
    
    if ( node < node_size ) {
        T   mean  = running_mean_buf[node];
        T   var   = running_var_buf[node];
        T   rstd = (T)1.0 / (sqrt(var) + (T)1.0e-7);

        int loop_size = ((frame_size + blockDim.x - 1) & ~(blockDim.x - 1));
        for ( int frame = id; frame < loop_size; frame += id_step) {
            int unit     = (frame >> 5);
            int bit      = (frame & 0x1f);
            int bit_mask = (1 << bit);

            int y_mask = 0;
            if ( node < node_size && frame < frame_size) {
                // Forward計算
                T   x[N];
                for ( int i = 0; i < N; ++i) {
                    x[i] = (T)0.5 + ((x_ptr[i][unit] & bit_mask) ? +unbinarize_bias : -unbinarize_bias);
                }
                T   y = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

                y = ((y - mean) * rstd) * gamma + beta;

                if ( y > (T)0.5 ) {
                    y_mask = bit_mask;
                }
            }

            y_mask = device_int_ShuffleOr(y_mask);

            if ( bit == 0 ) {
                if ( node < node_size && frame < frame_size ) {
                    y_ptr[unit] = y_mask;
                }
            }
        }
    }
}


template <int N>
BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardInference
        (
            int   const     *dev_x_buf,
            int             *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float const     *running_mean_buf,
            float const     *running_var_buf,
            float           gamma,
            float           beta,
            float           unbinarize_bias,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 256;
    unsigned int const MAX_FRAME_UNIT = 256;
    unsigned int const MAX_NODE_UNIT  = 8;  // THREAD_SIZE/32 より小さくすること

#if 0
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size && block.x > 32 ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size                  ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size  )                { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size && block.x > 32) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (node_size + (block.y - 1)) / block.y);
    
    kernal_bit_DifferentiableLut_ForwardInference<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            running_mean_buf,
            running_var_buf,
            gamma,
            beta,
            unbinarize_bias,
            node_size,
            frame_size,
            frame_stride,
            lut_binarize
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}



// -------------------------------------------------
//  Backward
// -------------------------------------------------


// real type
template<int N=6, typename T=float, int MAX_FRAME_UNIT=256, int MAX_NODE_UNIT=16>
__global__ void kernal_DifferentiableLut_BackwardPhase0
        (
            T   const   *x_buf,
            T   const   *dy_buf,
            int const   *input_index,
            T   const   *W_buf,
            T           *dW_buf,
            T   const   *mean_buf,
            T   const   *rstd_buf,
            T           *dmean_buf,
            T           *dvar_buf,
            T           gamma,
            T           beta,
            T           unbinarize_bias,
            T           reciprocal_frame_size,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         lut_binarize,
            int         binary_mode
        )
{

    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T           sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  T           W[(1 << N)][MAX_NODE_UNIT];
                T   const   *x_ptr[N];
                T   const   *dy_ptr;
    
    // initialize dW
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < (1 << N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1 << N) + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > (T)0.5 ? (T)1.0 : (T)0.0;
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
    

    T   mean;
    T   rstd;
    if ( node < node_size ) {
        mean = mean_buf[node];
        rstd = rstd_buf[node];
    }
    T   rstd2 = rstd * rstd;

    T   dmeanx = 0;
    T   dstd   = 0;
    for ( int frame = id; frame < frame_size; frame += id_step ) {
        if ( node < node_size ) {
            // x を再計算
            T   x_vec[N];
            if ( binary_mode ) {
                for ( int i = 0; i < N; ++i) {
                    x_vec[i] = (T)0.5 +((x_ptr[i][frame] > (T)0.5)  ? +unbinarize_bias : -unbinarize_bias);
                }
            }
            else {
                for ( int i = 0; i < N; ++i) {
                    x_vec[i] = max((T)0.0, min((T)1.0, x_ptr[i][frame]));
                }
            }
            T   x = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x_vec, W);
            T   tanh_x = ((x - mean) * rstd) * gamma + beta;
            
            // hard-tanh
            T   dy = dy_ptr[frame];
            if (tanh_x <= (T)0.0) { dy = (T)0.0; }
            if (tanh_x >= (T)1.0) { dy = (T)0.0; }

            // BatchNorm
            T   xc = x - mean;
    //      T   xn = xc * rstd;
            T   dxn = gamma * dy;

            dstd   += -(dxn * xc * rstd2);
            dmeanx += -(dxn * rstd);
        }
    }

    // block内でX軸で集計
    dstd   = device_LocalSumX<T>(dstd,   sbuf[node_id]);
    dmeanx = device_LocalSumX<T>(dmeanx, sbuf[node_id]);

    T   dvar  = dstd * rstd;
    T   dmean = (dmeanx - (mean * dvar)) * reciprocal_frame_size;

    if ( node < node_size ) {
        if ( id == 0 ) {
            dvar_buf[node]  = dvar;
            dmean_buf[node] = dmean;
        }
    }  
}


template<int N=6, typename T=float, int MAX_FRAME_UNIT=256, int MAX_NODE_UNIT=16>
__global__ void kernal_DifferentiableLut_BackwardPhase1
        (
            T   const   *x_buf,
            T   const   *dy_buf,
            T           *dx_buf,
            int const   *input_index,
            T   const   *W_buf,
            T           *dW_buf,
            T   const   *mean_buf,
            T   const   *rstd_buf,
            T   const   *dmean_buf,
            T   const   *dvar_buf,
            T           gamma,
            T           beta,
            T           unbinarize_bias,
            T           reciprocal_frame_size,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         dx_frame_stride,
            int         lut_binarize,
            int         binary_mode
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T           sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  T           dW_prev[(1 << N)][MAX_NODE_UNIT];
    __shared__  T           W[(1 << N)][MAX_NODE_UNIT];
                T           dW[(1 << N)];
                T   const   *x_ptr[N];
                T   const   *dy_ptr;
    
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
                W[i][node_id] = W[i][node_id] > (T)0.5 ? (T)1.0 : (T)0.0;
            }
        }
        
        // init pointer
        for ( int i = 0; i < N; ++i ) {
            int input_node = input_index[N*node + i];
            x_ptr[i]  = &x_buf[input_node * frame_stride];
        }

        dy_ptr = &dy_buf[node * frame_stride];
    }
    
    T   mean;
    T   rstd;
    T   dmean;
    T   dvar;
    if ( node < node_size ) {
        mean  = mean_buf[node];
        rstd  = rstd_buf[node];
        dmean = dmean_buf[node];
        dvar  = dvar_buf[node];
    }

    for ( int frame = id; frame < frame_size; frame += id_step ) {
        if ( node < node_size ) {
            // x を再計算
            T   x_vec[N];
            if ( binary_mode ) {
                for ( int i = 0; i < N; ++i) {
                    x_vec[i] = (T)0.5 +((x_ptr[i][frame] > (T)0.5)  ? +unbinarize_bias : -unbinarize_bias);
                }
            }
            else {
                for ( int i = 0; i < N; ++i) {
                    x_vec[i] = max((T)0.0, min((T)1.0, x_ptr[i][frame]));
                }
            }
            T   x = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x_vec, W);
            T   tanh_x = ((x - mean) * rstd) * gamma + beta;

            // hard-tanh
            T   dy = dy_ptr[frame];
            if (tanh_x <= (T)0.0) { dy = (T)0.0; }
            if (tanh_x >= (T)1.0) { dy = (T)0.0; }

            T   dxn = dy * gamma;
            T   dxc = dxn * rstd;
            T   dx  = dxc + dmean + (x * dvar * reciprocal_frame_size);

            StochasticLut<N, T, MAX_NODE_UNIT>::NodeBackward(node_id, x_vec, dx, &dx_buf[node*N*dx_frame_stride + frame], W, dW, dx_frame_stride);
        }
    }

    for ( int i = 0; i < (1 << N); ++i ) {
        dW[i] = device_LocalSumX<T>(dW[i], sbuf[node_id]);
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
BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            int   const     *dev_reverse_index,
            float const     *dev_W,
            float           *dev_dW,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            float           *dev_dmean_tmp,
            float           *dev_dvar_tmp,
            float           gamma,
            float           beta,
            float           unbinarize_bias,
            int             reverse_index_stride,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             tmp_frame_size,
            int             tmp_frame_stride,
            int             lut_binarize,
            int             binary_mode,
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
        while ( (int)block.x / 2 >= frame_size && frame_size > 32 ) { block.x /= 2; block.y *= 2; }
        while ( (int)block.y / 2 >= output_node_size              ) { block.y /= 2; }
#else
        dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
        while ( (int)block.y / 2 >= output_node_size              ) { block.y /= 2; block.x *= 2;}
        while ( (int)block.x / 2 >= frame_size && frame_size > 32 ) { block.x /= 2; }
#endif

        block.x = std::min(block.x, MAX_FRAME_UNIT);
        block.y = std::min(block.y, MAX_NODE_UNIT);
        dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);
        kernal_DifferentiableLut_BackwardPhase0<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>
            (
                dev_x_buf,
                dev_dy_buf,
                dev_input_index,
                dev_W,
                dev_dW,
                dev_mean_buf,
                dev_rstd_buf,
                dev_dmean_tmp,
                dev_dvar_tmp,
                gamma,
                beta,
                unbinarize_bias,
                1.0f / frame_size,
                output_node_size,
                frame_size,
                frame_stride,
                lut_binarize,
                binary_mode
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }
    
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
            while ( (int)block.x / 2 >= unit_frame_size && unit_frame_size > 32 ) { block.x /= 2; block.y *= 2; }
            while ( (int)block.y / 2 >= output_node_size                        ) { block.y /= 2; }
    #else
            dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
            while ( (int)block.y / 2 >= output_node_size                        ) { block.y /= 2; block.x *= 2;}
            while ( (int)block.x / 2 >= unit_frame_size && unit_frame_size > 32 ) { block.x /= 2; }
    #endif

            block.x = std::min(block.x, MAX_FRAME_UNIT);
            block.y = std::min(block.y, MAX_NODE_UNIT);
            dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);
            kernal_DifferentiableLut_BackwardPhase1<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>
                (
                    dev_x_buf  + frame_offset,
                    dev_dy_buf + frame_offset,
                    dev_dx_tmp,
                    dev_input_index,
                    dev_W,
                    dev_dW,
                    dev_mean_buf,
                    dev_rstd_buf,
                    dev_dmean_tmp,
                    dev_dvar_tmp,
                    gamma,
                    beta,
                    unbinarize_bias,
                    1.0f / frame_size,
                    output_node_size,
                    unit_frame_size,
                    frame_stride,
                    tmp_frame_stride,
                    lut_binarize,
                    binary_mode
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



// bit packing binary
template<int N=6, typename T=float, int MAX_FRAME_UNIT=256, int MAX_NODE_UNIT=16>
__global__ void kernal_bit_DifferentiableLut_BackwardPhase0
        (
            int const   *x_buf,
            T   const   *dy_buf,
            int const   *input_index,
            T   const   *W_buf,
            T           *dW_buf,
            T   const   *mean_buf,
            T   const   *rstd_buf,
            T           *dmean_buf,
            T           *dvar_buf,
            T           gamma,
            T           beta,
            T           unbinarize_bias,
            T           reciprocal_frame_size,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            int         bin_frame_stride,
            int         lut_binarize
        )
{

    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T           sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  T           W[(1 << N)][MAX_NODE_UNIT];
                int const   *x_ptr[N];
                T   const   *dy_ptr;
    
    // initialize dW
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < (1 << N); i += id_step ) {
            W[i][node_id] = W_buf[node * (1 << N) + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > (T)0.5 ? (T)1.0 : (T)0.0;
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
    

    T   mean;
    T   rstd;
    if ( node < node_size ) {
        mean = mean_buf[node];
        rstd = rstd_buf[node];
    }
    T   rstd2 = rstd * rstd;

    T   dmeanx = 0;
    T   dstd   = 0;
    int loop_size = ((frame_size + blockDim.x - 1) & ~(blockDim.x - 1));
    for ( int frame = id; frame < loop_size; frame += id_step ) {
        if ( node < node_size && frame < frame_size ) {
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);
            
            // x を再計算
            T   x_vec[N];
            for ( int i = 0; i < N; ++i) {
                x_vec[i] = (T)0.5 +((x_ptr[i][unit] & bit)  ? +unbinarize_bias : -unbinarize_bias);
            }
            T   x = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x_vec, W);
            T   tanh_x = ((x - mean) * rstd) * gamma + beta;
            
            // hard-tanh
            T   dy = dy_ptr[frame];
            if (tanh_x <= (T)0.0) { dy = (T)0.0; }
            if (tanh_x >= (T)1.0) { dy = (T)0.0; }

            // BatchNorm
            T   xc = x - mean;
    //      T   xn = xc * rstd;
            T   dxn = gamma * dy;

            dstd   += -(dxn * xc * rstd2);
            dmeanx += -(dxn * rstd);
        }
    }

    dstd   = device_fp32_LocalSum(dstd,   sbuf[node_id]);
    dmeanx = device_fp32_LocalSum(dmeanx, sbuf[node_id]);

    T   dvar  = dstd * rstd;
    T   dmean = (dmeanx - (mean * dvar)) * reciprocal_frame_size;

    if ( node < node_size ) {
        if ( id == 0 ) {
            dvar_buf[node]  = dvar;
            dmean_buf[node] = dmean;
        }
    }  
}


template<int N=6, typename T=float, int MAX_FRAME_UNIT=256, int MAX_NODE_UNIT=16>
__global__ void kernal_bit_DifferentiableLut_BackwardPhase1
        (
            int const   *x_buf,
            T   const   *dy_buf,
            T           *dx_buf,
            int const   *input_index,
            T   const   *W_buf,
            T           *dW_buf,
            T   const   *mean_buf,
            T   const   *rstd_buf,
            T   const   *dmean_buf,
            T   const   *dvar_buf,
            T           gamma,
            T           beta,
            T           unbinarize_bias,
            T           reciprocal_frame_size,
            int         node_size,
            int         frame_size,
            int         x_frame_stride,
            int         dy_frame_stride,
            int         dx_frame_stride,
            int         lut_binarize
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  T           sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  T           dW_prev[(1 << N)][MAX_NODE_UNIT];
    __shared__  T           W[(1 << N)][MAX_NODE_UNIT];
                T           dW[(1 << N)];
                int const   *x_ptr[N];
                T   const   *dy_ptr;
    
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
                W[i][node_id] = W[i][node_id] > (T)0.5 ? (T)1.0 : (T)0.0;
            }
        }
        
        // init pointer
        for ( int i = 0; i < N; ++i ) {
            int input_node = input_index[N*node + i];
            x_ptr[i]  = &x_buf[input_node * x_frame_stride];
        }

        dy_ptr = &dy_buf[node * dy_frame_stride];
    }
    
    T   mean;
    T   rstd;
    T   dmean;
    T   dvar;
    if ( node < node_size ) {
        mean  = mean_buf[node];
        rstd  = rstd_buf[node];
        dmean = dmean_buf[node];
        dvar  = dvar_buf[node];
    }

    int loop_size = ((frame_size + blockDim.x - 1) & ~(blockDim.x - 1));
    for ( int frame = id; frame < loop_size; frame += id_step ) {
        if ( node < node_size && frame < frame_size ) {
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);
            
            // x を再計算
            T   x_vec[N];
            for ( int i = 0; i < N; ++i) {
                x_vec[i] = (T)0.5 + ((x_ptr[i][unit] & bit) ? +unbinarize_bias : -unbinarize_bias);
            }
            T   x = StochasticLut<N, T, MAX_NODE_UNIT>::NodeForward(node_id, x_vec, W);
            T   tanh_x = ((x - mean) * rstd) * gamma + beta;

            // hard-tanh
            T   dy = dy_ptr[frame];
            if (tanh_x <= (T)0.0) { dy = (T)0.0; }
            if (tanh_x >= (T)1.0) { dy = (T)0.0; }

            T   dxn = dy * gamma;
            T   dxc = dxn * rstd;
            T   dx  = dxc + dmean + (x * dvar * reciprocal_frame_size);

            StochasticLut<N, T, MAX_NODE_UNIT>::NodeBackward(node_id, x_vec, dx, &dx_buf[node*N*dx_frame_stride + frame], W, dW, dx_frame_stride);
        }
    }

    for ( int i = 0; i < (1 << N); ++i ) {
        dW[i] = device_LocalSumX<T>(dW[i], sbuf[node_id]);
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
BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_Backward
        (
            int   const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            int   const     *dev_reverse_index,
            float const     *dev_W,
            float           *dev_dW,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            float           *dev_dmean_tmp,
            float           *dev_dvar_tmp,
            float           gamma,
            float           beta,
            float           unbinarize_bias,
            int             reverse_index_stride,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             x_frame_stride,
            int             tmp_frame_size,
            int             tmp_frame_stride,
            int             lut_binarize,
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
        while ( (int)block.x / 2 >= frame_size && frame_size > 32 ) { block.x /= 2; block.y *= 2; }
        while ( (int)block.y / 2 >= output_node_size              ) { block.y /= 2; }
#else
        dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
        while ( (int)block.y / 2 >= output_node_size              ) { block.y /= 2; block.x *= 2;}
        while ( (int)block.x / 2 >= frame_size && frame_size > 32 ) { block.x /= 2; }
#endif

        block.x = std::min(block.x, MAX_FRAME_UNIT);
        block.y = std::min(block.y, MAX_NODE_UNIT);
        dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);
        kernal_bit_DifferentiableLut_BackwardPhase0<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>
            (
                dev_x_buf,
                dev_dy_buf,
                dev_input_index,
                dev_W,
                dev_dW,
                dev_mean_buf,
                dev_rstd_buf,
                dev_dmean_tmp,
                dev_dvar_tmp,
                gamma,
                beta,
                unbinarize_bias,
                1.0f / frame_size,
                output_node_size,
                frame_size,
                frame_stride,
                x_frame_stride,
                lut_binarize
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }
    
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
            while ( (int)block.x / 2 >= unit_frame_size && unit_frame_size > 32 ) { block.x /= 2; block.y *= 2; }
            while ( (int)block.y / 2 >= output_node_size                        ) { block.y /= 2; }
    #else
            dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
            while ( (int)block.y / 2 >= output_node_size                        ) { block.y /= 2; block.x *= 2;}
            while ( (int)block.x / 2 >= unit_frame_size && unit_frame_size > 32 ) { block.x /= 2; }
    #endif

            block.x = std::min(block.x, MAX_FRAME_UNIT);
            block.y = std::min(block.y, MAX_NODE_UNIT);
            dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);
            kernal_bit_DifferentiableLut_BackwardPhase1<N, float, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>
                (
                    dev_x_buf  + (frame_offset / 32),
                    dev_dy_buf + frame_offset,
                    dev_dx_tmp,
                    dev_input_index,
                    dev_W,
                    dev_dW,
                    dev_mean_buf,
                    dev_rstd_buf,
                    dev_dmean_tmp,
                    dev_dvar_tmp,
                    gamma,
                    beta,
                    unbinarize_bias,
                    1.0f / frame_size,
                    output_node_size,
                    unit_frame_size,
                    x_frame_stride,
                    frame_stride,
                    tmp_frame_stride,
                    lut_binarize
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
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardTraining<6>(float const *, float *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardTraining<5>(float const *, float *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardTraining<4>(float const *, float *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardTraining<3>(float const *, float *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardTraining<2>(float const *, float *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, int, cudaStream_t);

template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardTraining<6>(int const *, int *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardTraining<5>(int const *, int *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardTraining<4>(int const *, int *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardTraining<3>(int const *, int *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardTraining<2>(int const *, int *, int const *, float const *, float *, float *, float *, float *, float, float, float, float, int, int, int, int, cudaStream_t);

template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardInference<6>(float const *, float *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardInference<5>(float const *, float *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardInference<4>(float const *, float *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardInference<3>(float const *, float *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardInference<2>(float const *, float *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, int, cudaStream_t);

template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardInference<6>(int const *, int *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardInference<5>(int const *, int *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardInference<4>(int const *, int *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardInference<3>(int const *, int *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardInference<2>(int const *, int *, int const *, float const *, float const *, float const *, float, float, float, int, int, int, int, cudaStream_t);

template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_Backward<6>(float const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_Backward<5>(float const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_Backward<4>(float const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_Backward<3>(float const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_Backward<2>(float const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);

template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_Backward<6>(int const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_Backward<5>(int const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_Backward<4>(int const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_Backward<3>(int const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_Backward<2>(int const *, float const *, float *, float *, int const *, int const *, float const *, float *, float const *, float const *, float *, float *, float, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);


// end of file
