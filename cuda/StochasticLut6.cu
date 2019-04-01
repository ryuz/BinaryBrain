#include <iostream>
#include <chrono>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



// -------------------------------------------------
//  Forward
// -------------------------------------------------


__global__ void kernal_fp32_StochasticLut6_Forward(
            float const     *x_buf,
            float           *y_buf,
            int   const     *input_index,
            float const     *W_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             binary_mode
        )
{
    int node       = blockIdx.y * blockDim.y + threadIdx.y;
    int frame_step = blockDim.x;
    int frame_base = threadIdx.x;
    
    if (node >= node_size) {
        return;
    }

    float W[64];
    for ( int i = 0; i < 64; ++i) {
        W[i] = W_buf[node * 64 + i];
        if ( binary_mode ) {
            W[i] = W[i] > 0.5 ? 1.0 : 0.0;
        }
    }
    
    float const *x_ptr[6];
    for ( int i = 0; i < 6; ++i) {
        x_ptr[i] = &x_buf[frame_stride * input_index[6*node + i]];
    }
    float *y_ptr = &y_buf[node*frame_stride];
    
    for (int frame = frame_base; frame < frame_size; frame += frame_step) {
	    float   xp[6], xn[6];
        for ( int i = 0; i < 6; ++i) {
            xp[i] = x_ptr[i][frame];
            xn[i] = 1.0 - xp[i];
        }

        float x0_00 = xn[1] * xn[0];
        float x0_01 = xn[1] * xp[0];
        float x0_10 = xp[1] * xn[0];
        float x0_11 = xp[1] * xp[0];
        float x1_00 = xn[3] * xn[2];
        float x1_01 = xn[3] * xp[2];
        float x1_10 = xp[3] * xn[2];
        float x1_11 = xp[3] * xp[2];
        float x2_00 = xn[5] * xn[4];
        float x2_01 = xn[5] * xp[4];
        float x2_10 = xp[5] * xn[4];
        float x2_11 = xp[5] * xp[4];

        float xi[64];
        xi[0]  = x2_00 * x1_00 * x0_00;
        xi[1]  = x2_00 * x1_00 * x0_01;
        xi[2]  = x2_00 * x1_00 * x0_10;
        xi[3]  = x2_00 * x1_00 * x0_11;
        xi[4]  = x2_00 * x1_01 * x0_00;
        xi[5]  = x2_00 * x1_01 * x0_01;
        xi[6]  = x2_00 * x1_01 * x0_10;
        xi[7]  = x2_00 * x1_01 * x0_11;
        xi[8]  = x2_00 * x1_10 * x0_00;
        xi[9]  = x2_00 * x1_10 * x0_01;
        xi[10] = x2_00 * x1_10 * x0_10;
        xi[11] = x2_00 * x1_10 * x0_11;
        xi[12] = x2_00 * x1_11 * x0_00;
        xi[13] = x2_00 * x1_11 * x0_01;
        xi[14] = x2_00 * x1_11 * x0_10;
        xi[15] = x2_00 * x1_11 * x0_11;
        xi[16] = x2_01 * x1_00 * x0_00;
        xi[17] = x2_01 * x1_00 * x0_01;
        xi[18] = x2_01 * x1_00 * x0_10;
        xi[19] = x2_01 * x1_00 * x0_11;
        xi[20] = x2_01 * x1_01 * x0_00;
        xi[21] = x2_01 * x1_01 * x0_01;
        xi[22] = x2_01 * x1_01 * x0_10;
        xi[23] = x2_01 * x1_01 * x0_11;
        xi[24] = x2_01 * x1_10 * x0_00;
        xi[25] = x2_01 * x1_10 * x0_01;
        xi[26] = x2_01 * x1_10 * x0_10;
        xi[27] = x2_01 * x1_10 * x0_11;
        xi[28] = x2_01 * x1_11 * x0_00;
        xi[29] = x2_01 * x1_11 * x0_01;
        xi[30] = x2_01 * x1_11 * x0_10;
        xi[31] = x2_01 * x1_11 * x0_11;
        xi[32] = x2_10 * x1_00 * x0_00;
        xi[33] = x2_10 * x1_00 * x0_01;
        xi[34] = x2_10 * x1_00 * x0_10;
        xi[35] = x2_10 * x1_00 * x0_11;
        xi[36] = x2_10 * x1_01 * x0_00;
        xi[37] = x2_10 * x1_01 * x0_01;
        xi[38] = x2_10 * x1_01 * x0_10;
        xi[39] = x2_10 * x1_01 * x0_11;
        xi[40] = x2_10 * x1_10 * x0_00;
        xi[41] = x2_10 * x1_10 * x0_01;
        xi[42] = x2_10 * x1_10 * x0_10;
        xi[43] = x2_10 * x1_10 * x0_11;
        xi[44] = x2_10 * x1_11 * x0_00;
        xi[45] = x2_10 * x1_11 * x0_01;
        xi[46] = x2_10 * x1_11 * x0_10;
        xi[47] = x2_10 * x1_11 * x0_11;
        xi[48] = x2_11 * x1_00 * x0_00;
        xi[49] = x2_11 * x1_00 * x0_01;
        xi[50] = x2_11 * x1_00 * x0_10;
        xi[51] = x2_11 * x1_00 * x0_11;
        xi[52] = x2_11 * x1_01 * x0_00;
        xi[53] = x2_11 * x1_01 * x0_01;
        xi[54] = x2_11 * x1_01 * x0_10;
        xi[55] = x2_11 * x1_01 * x0_11;
        xi[56] = x2_11 * x1_10 * x0_00;
        xi[57] = x2_11 * x1_10 * x0_01;
        xi[58] = x2_11 * x1_10 * x0_10;
        xi[59] = x2_11 * x1_10 * x0_11;
        xi[60] = x2_11 * x1_11 * x0_00;
        xi[61] = x2_11 * x1_11 * x0_01;
        xi[62] = x2_11 * x1_11 * x0_10;
        xi[63] = x2_11 * x1_11 * x0_11;

        float sig = 0;
		for ( int i = 0; i < 64; ++i) {
		    sig += W[i] * xi[i];
		}

        // clamp
        sig = max(0.0, sig);
        sig = min(1.0, sig);
        
        y_ptr[frame] = sig;
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
            int             binary_mode,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block;
    block.x = std::min(192, frame_size);
    block.y = std::min(192, node_size);
    while (block.x * block.y > 192) {
        block.y = (block.y + 1) / block.y;
    }
    dim3    grid;
    grid.x = 1;
    grid.y = (node_size + (block.y - 1)) / block.y;

    kernal_fp32_StochasticLut6_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            node_size,
            frame_size,
            frame_stride,
            binary_mode
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}



// -------------------------------------------------
//  Backward
// -------------------------------------------------

#if 0

// kernel
__global__ void kernal_fp32_StochasticLut6_Backward
        (
            float const     *x_buf,
            float const     *dy_buf,
            float           *dx_buf,
            int   const     *input_index,
            float const     *W,
            float           *dW,
            int             frame_size,
            int             frame_stride,
            int             binary_mode
        )
{
}


__global__ void kernal_fp32_StochasticLut6_BackwardMarge(
            const float*    src_buf,
            float*          dst_buf,
            const int*      input_index,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int n          = blockDim.y * blockIdx.y + threadIdx.y;
    int frame_base = threadIdx.x;
    int frame_step = blockDim.x;
    
    for ( int node = 0; node < node_size; ++node ) {
        int in_idx = input_index[node*6 + n];
        float*       dst_buf_ptr = &dst_buf[frame_size * in_idx];
        const float* src_buf_ptr = &src_buf[(6 * node + n) * frame_size];
        for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
            dst_buf_ptr[frame] += src_buf_ptr[frame];
        }
    }
}


int bbcu_fp32_StochasticLut6_Backward(
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *dev_dW,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             binary_mode,
            cudaStream_t    streamId = 0
    )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    {
        const int x_size = 128; // (8192 / (N*M));

        dim3    grid(output_node_size);
        dim3    block(x_size, 1, 1);
        
        kernal_fp32_StochasticLut6_Backward<<<grid, block, 0, streamId>>>(
                dev_x_buf,
                dev_dy_buf,
                dev_dx_tmp,
                dev_input_index,
                dev_W,
                dev_dW,
                output_node_size,
                frame_size,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }

    {
        BB_CUDA_SAFE_CALL(cudaMemset(dev_dx_buf, 0, input_node_size * frame_stride * sizeof(float)));

        int block_x = frame_size;
        while ( block_x > 1024 ) { block_x /= 2; }

        dim3    grid((frame_size + block_x - 1) /block_x, 6);
        dim3    block(block_x, 1, 1);
        kernal_fp32_StochasticLut6_BackwardMarge<<<grid, block>>>(
                dev_dx_tmp,
                dev_dx_buf,
                dev_input_index,
                output_node_size,
                frame_size,
                frame_stride
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }

    return 0;
}


BBCU_DLL_EXPORT int bbcu_fp32_MicroMlp6x16_Backward(
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            float const     *dev_hidden_W,
            float const     *dev_hidden_b,
            float           *dev_hidden_dW,
            float           *dev_hidden_db,
            float const     *dev_output_W,
            float const     *dev_output_b,
            float           *dev_output_dW,
            float           *dev_output_db,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    return bbcu_fp32_MicroMlp_Backward<6, 16>(
            dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
            dev_dx_tmp,
            dev_input_index,
            dev_hidden_W,
            dev_hidden_b,
            dev_hidden_dW,
            dev_hidden_db,
            dev_output_W,
            dev_output_b,
            dev_output_dW,
            dev_output_db,
            input_node_size,
            output_node_size,
            frame_size,
            frame_stride,
            streamId
        );
}

#endif


// end of file
