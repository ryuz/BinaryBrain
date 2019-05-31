#include <iostream>
#include <chrono>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


#include "Common.cuh"
#include "StochasticLut.cuh"


//#define BINARY_BIAS     (0.125/2)
//#define BINARY_BIAS     0.125
#define BINARY_BIAS     0.2
#define BINARY_ZERO     (0.5 - BINARY_BIAS)
#define BINARY_ONE      (0.5 + BINARY_BIAS)


// -------------------------------------------------
//  Forward
// -------------------------------------------------

template<int MAX_FRAME_UNIT=32, int MAX_NODE_UNIT=32>
__global__ void kernal_bit_fp32_SparseBinaryLut6_ForwardTraining
        (
            int   const     *x_buf,
            int             *y_buf,
            int   const     *input_index,
            float const     *W_buf,
            float           *mean_buf,
            float           *rstd_buf,
            float           *running_mean_buf,
            float           *running_var_buf,
            float           gamma,
            float           beta,
            float           momentum,
            float           reciprocal_frame_size,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  float       sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];

    __shared__  float       W[64][MAX_NODE_UNIT];
                int   const *x_ptr[6];
                int         *y_ptr;
    
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < 64; i += id_step ) {
            W[i][node_id] = W_buf[node * 64 + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > 0.5 ? 1.0 : 0.0;
            }
        }
        
        // read input index
        for ( int i = 0; i < 6; ++i ) {
            x_ptr[i] = &x_buf[frame_stride * input_index[6*node + i]];
        }
                     
        y_ptr = &y_buf[node * frame_stride];
    }

    __syncthreads();
    
    // ïΩãœÇ∆ï™éUåvë™
    float s1 = 0, c1 = 0, y1, t1;
    float s2 = 0, c2 = 0, y2, t2;
    for (int frame = id; frame < frame_size; frame += id_step) {
        if ( node < node_size ) {
            // ForwardåvéZ
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);
            float x[6];
            for ( int i = 0; i < 6; ++i) {
                x[i] = (x_ptr[i][unit] & bit) ? BINARY_ONE : BINARY_ZERO;
            }
            float y = StochasticLut<6, float, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

            // èWåv
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
    float mean = s1 * reciprocal_frame_size;
    float var = max(1.0e-7f, (s2 * reciprocal_frame_size) - (mean * mean));
  
    float rstd = rsqrt(var);

    // èëÇ´çûÇ›
    if (id == 0) {
        if ( node < node_size ) {
            running_mean_buf[node] = running_mean_buf[node] * momentum + mean * (1.0f - momentum);
            running_var_buf[node]  = running_var_buf[node] * momentum + var * (1.0f - momentum);
            mean_buf[node] = mean;
            rstd_buf[node] = rstd;
        }
    }

    // ê≥ãKâª
    int loop_size = ((frame_size + blockDim.x - 1) & ~(blockDim.x - 1));
    for ( int frame = id; frame < loop_size; frame += id_step) {
        int unit     = (frame >> 5);
        int bit      = (frame & 0x1f);
        int bit_mask = (1 << bit);

        int y_mask = 0;
        if ( node < node_size && frame < frame_size) {
            // ForwardåvéZ
            float x[6];
            for ( int i = 0; i < 6; ++i) {
                x[i] = (x_ptr[i][unit] & bit_mask) ? BINARY_ONE : BINARY_ZERO;
            }
            float y = StochasticLut<6, float, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

            y = (y - mean) * rstd;
            y = y * gamma + beta;

            if ( y > 0.5 ) {
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


BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseBinaryLut6_ForwardTraining
        (
            int   const     *dev_x_buf,
            int             *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *mean_buf,
            float           *rstd_buf,
            float           *running_mean_buf,
            float           *running_var_buf,
            float           gamma,
            float           beta,
            float           momentum,
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
    unsigned int const MAX_NODE_UNIT  = 8;  // THREAD_SIZE/32 ÇÊÇËè¨Ç≥Ç≠Ç∑ÇÈÇ±Ç∆

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
    
    kernal_bit_fp32_SparseBinaryLut6_ForwardTraining<MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            mean_buf,
            rstd_buf,
            running_mean_buf,
            running_var_buf,
            gamma,
            beta,
            momentum,
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


template<int MAX_FRAME_UNIT=32, int MAX_NODE_UNIT=32>
__global__ void kernal_bit_fp32_SparseBinaryLut6_ForwardInference
        (
            int   const     *x_buf,
            int             *y_buf,
            int   const     *input_index,
            float const     *W_buf,
            float const     *running_mean_buf,
            float const     *running_var_buf,
            float           gamma,
            float           beta,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  float       W[64][MAX_NODE_UNIT];
                int   const *x_ptr[6];
                int         *y_ptr;
    
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < 64; i += id_step ) {
            W[i][node_id] = W_buf[node * 64 + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > 0.5 ? 1.0 : 0.0;
            }
        }
        
        // read input index
        for ( int i = 0; i < 6; ++i ) {
            x_ptr[i] = &x_buf[frame_stride * input_index[6*node + i]];
        }
                     
        y_ptr = &y_buf[node * frame_stride];
    }

    __syncthreads();
    
    if ( node < node_size ) {
        float mean  = running_mean_buf[node];
        float var   = running_var_buf[node];
        float rstd = 1.0 / (sqrt(var) + 1.0e-7);

        int loop_size = ((frame_size + blockDim.x - 1) & ~(blockDim.x - 1));
        for ( int frame = id; frame < loop_size; frame += id_step) {
            int unit     = (frame >> 5);
            int bit      = (frame & 0x1f);
            int bit_mask = (1 << bit);

            int y_mask = 0;
            if ( node < node_size && frame < frame_size) {
                // ForwardåvéZ
                float x[6];
                for ( int i = 0; i < 6; ++i) {
                    x[i] = (x_ptr[i][unit] & bit_mask) ? BINARY_ONE : BINARY_ZERO;
                }
                float y = StochasticLut<6, float, MAX_NODE_UNIT>::NodeForward(node_id, x, W);

                y = ((y - mean) * rstd) * gamma + beta;

                if ( y > 0.5 ) {
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


BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseBinaryLut6_ForwardInference
        (
            int   const     *dev_x_buf,
            int             *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float const     *running_mean_buf,
            float const     *running_var_buf,
            float           gamma,
            float           beta,
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
    unsigned int const MAX_NODE_UNIT  = 8;  // THREAD_SIZE/32 ÇÊÇËè¨Ç≥Ç≠Ç∑ÇÈÇ±Ç∆

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
    
    kernal_bit_fp32_SparseBinaryLut6_ForwardInference<MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            running_mean_buf,
            running_var_buf,
            gamma,
            beta,
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


template<int MAX_FRAME_UNIT=256, int MAX_NODE_UNIT=16>
__global__ void kernal_bit_fp32_SparseBinaryLut6_BackwardPhase0
        (
            int   const     *x_buf,
            float const     *dy_buf,
            int   const     *input_index,
            float const     *W_buf,
            float           *dW_buf,
            float const     *mean_buf,
            float const     *rstd_buf,
            float           *dmean_buf,
            float           *dvar_buf,
            float           gamma,
            float           beta,
            float           reciprocal_frame_size,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             bin_frame_stride,
            int             lut_binarize
        )
{

    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  float       sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  float       W[64][MAX_NODE_UNIT];
                int   const *x_ptr[6];
                float const *dy_ptr;
    
    // initialize dW
    if ( node < node_size ) {
        // read W
        for ( int i = id; i < 64; i += id_step ) {
            W[i][node_id] = W_buf[node * 64 + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > 0.5 ? 1.0 : 0.0;
            }
        }
        
        // init pointer
        for ( int i = 0; i < 6; ++i ) {
            int input_node = input_index[6*node + i];
            x_ptr[i]  = &x_buf[input_node * bin_frame_stride];
        }

        dy_ptr = &dy_buf[node * frame_stride];
    }

    __syncthreads();
    

    float mean;
    float rstd;
    if ( node < node_size ) {
        mean = mean_buf[node];
        rstd = rstd_buf[node];
    }
    float rstd2 = rstd * rstd;

    float dmeanx = 0;
    float dstd   = 0;
    for ( int frame = id; frame < frame_size; frame += id_step ) {
        if ( node < node_size ) {
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);
            
            // x ÇçƒåvéZ
            float x_vec[6];
            for ( int i = 0; i < 6; ++i) {
                x_vec[i] = (x_ptr[i][unit] & bit) ? BINARY_ONE : BINARY_ZERO;
            }
            float x = StochasticLut<6, float, MAX_NODE_UNIT>::NodeForward(node_id, x_vec, W);
            float tanh_x = ((x - mean) * rstd) * gamma + beta;
            
            // hard-tanh
            float dy = dy_ptr[frame];
            if (tanh_x <= 0.0) { dy = 0.0; }
            if (tanh_x >= 1.0) { dy = 0.0; }

            // BatchNorm
            float xc = x - mean;
    //      float xn = xc * rstd;
            float dxn = gamma * dy;

            dstd   += -(dxn * xc * rstd2);
            dmeanx += -(dxn * rstd);
        }
    }

    dstd   = device_fp32_LocalSum(dstd,   sbuf[node_id]);
    dmeanx = device_fp32_LocalSum(dmeanx, sbuf[node_id]);

    float dvar  = dstd * rstd;
    float dmean = (dmeanx - (mean * dvar)) * reciprocal_frame_size;

    if ( node < node_size ) {
        if ( id == 0 ) {
            dvar_buf[node]  = dvar;
            dmean_buf[node] = dmean;
        }
    }  
}

template<int MAX_FRAME_UNIT=256, int MAX_NODE_UNIT=16>
__global__ void kernal_bit_fp32_SparseBinaryLut6_BackwardPhase1
        (
            int   const     *x_buf,
            float const     *dy_buf,
            float           *dx_buf,
            int   const     *input_index,
            float const     *W_buf,
            float           *dW_buf,
            float const     *mean_buf,
            float const     *rstd_buf,
            float const     *dmean_buf,
            float const     *dvar_buf,
            float           gamma,
            float           beta,
            float           reciprocal_frame_size,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             dy_frame_stride,
            int             dx_frame_stride,
            int             lut_binarize
        )
{
    int node_id = threadIdx.y;
    int node    = blockIdx.y * blockDim.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    __shared__  float       sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  float       dW_prev[64][MAX_NODE_UNIT];
    __shared__  float       W[64][MAX_NODE_UNIT];
                float       dW[64];
                int   const *x_ptr[6];
                float const *dy_ptr;
    
    // initialize dW
    if ( node < node_size ) {
        for ( int i = 0; i < 64; ++i) {
            dW[i] = 0;
        }

        for ( int i = id; i < 64; i += id_step ) {
            dW_prev[i][node_id] = dW_buf[node * 64 + i];
        }

        // read W
        for ( int i = id; i < 64; i += id_step ) {
            W[i][node_id] = W_buf[node * 64 + i];
            if ( lut_binarize ) {
                W[i][node_id] = W[i][node_id] > 0.5 ? 1.0 : 0.0;
            }
        }
        
        // init pointer
        for ( int i = 0; i < 6; ++i ) {
            int input_node = input_index[6*node + i];
            x_ptr[i]  = &x_buf[input_node * x_frame_stride];
        }

        dy_ptr = &dy_buf[node * dy_frame_stride];
    }
    
    float   mean;
    float   rstd;
    float   dmean;
    float   dvar;
    if ( node < node_size ) {
        mean  = mean_buf[node];
        rstd  = rstd_buf[node];
        dmean = dmean_buf[node];
        dvar  = dvar_buf[node];
    }

    for ( int frame = id; frame < frame_size; frame += id_step ) {
        if ( node < node_size ) {
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);
            
            // x ÇçƒåvéZ
            float x_vec[6];
            for ( int i = 0; i < 6; ++i) {
                x_vec[i] = (x_ptr[i][unit] & bit) ? BINARY_ONE : BINARY_ZERO;
            }
            float x = StochasticLut<6, float, MAX_NODE_UNIT>::NodeForward(node_id, x_vec, W);
            float tanh_x = ((x - mean) * rstd) * gamma + beta;

            // hard-tanh
            float dy = dy_ptr[frame];
            if (tanh_x <= 0.0) { dy = 0.0; }
            if (tanh_x >= 1.0) { dy = 0.0; }

            float dxn = dy * gamma;
            float dxc = dxn * rstd;
            float dx  = dxc + dmean + (x * dvar * reciprocal_frame_size);

            StochasticLut<6, float, MAX_NODE_UNIT>::NodeBackward(node_id, x_vec, dx, &dx_buf[node*6*dx_frame_stride + frame], W, dW, dx_frame_stride);
        }
    }

    for ( int i = 0; i < 64; ++i ) {
        dW[i] = device_fp32_LocalSum(dW[i], sbuf[node_id]);
    }

    if ( node < node_size ) {
        if ( id == 0 ) {
            for ( int i = 0; i < 64; ++i) {
                dW_buf[node*64 + i] = dW[i] + dW_prev[i][node_id];
            }
        }
    }
}


BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseBinaryLut6_Backward
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
        kernal_bit_fp32_SparseBinaryLut6_BackwardPhase0<MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>
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
            kernal_bit_fp32_SparseBinaryLut6_BackwardPhase1<MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>
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

            kernal_BackwardMarge<float><<<grid, block>>>
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



// end of file
