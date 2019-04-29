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
            int             frame_size,
            int             frame_stride,
            int             binary_mode
        )
{
    int node    = blockIdx.x;
    int id      = threadIdx.x;
    int id_step = blockDim.x;
    
    // read W
    __shared__ float    W[64];
    for ( int i = id; i < 64; i += id_step ) {
        W[i] = W_buf[node * 64 + i];
        if ( binary_mode ) {
            W[i] = W[i] > 0.5 ? 1.0 : 0.0;
        }
    }
    
    // read input index
    __shared__ float const  *x_ptr[6];
    for ( int i = id; i < 6; i += id_step ) {
        x_ptr[i] = &x_buf[frame_stride * input_index[6*node + i]];
    }
    float        *y_ptr = &y_buf[node * frame_stride];

    __syncthreads();

    for (int frame = id; frame < frame_size; frame += id_step) {
        float   xp[6], xn[6];
        for ( int i = 0; i < 6; ++i) {
            xp[i] = x_ptr[i][frame];
            xp[i] = min(1.0, max(0.0, xp[i]));
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

        float y = 0;
        y += W[0 ] * x2_00 * x1_00 * x0_00;
        y += W[1 ] * x2_00 * x1_00 * x0_01;
        y += W[2 ] * x2_00 * x1_00 * x0_10;
        y += W[3 ] * x2_00 * x1_00 * x0_11;
        y += W[4 ] * x2_00 * x1_01 * x0_00;
        y += W[5 ] * x2_00 * x1_01 * x0_01;
        y += W[6 ] * x2_00 * x1_01 * x0_10;
        y += W[7 ] * x2_00 * x1_01 * x0_11;
        y += W[8 ] * x2_00 * x1_10 * x0_00;
        y += W[9 ] * x2_00 * x1_10 * x0_01;
        y += W[10] * x2_00 * x1_10 * x0_10;
        y += W[11] * x2_00 * x1_10 * x0_11;
        y += W[12] * x2_00 * x1_11 * x0_00;
        y += W[13] * x2_00 * x1_11 * x0_01;
        y += W[14] * x2_00 * x1_11 * x0_10;
        y += W[15] * x2_00 * x1_11 * x0_11;
        y += W[16] * x2_01 * x1_00 * x0_00;
        y += W[17] * x2_01 * x1_00 * x0_01;
        y += W[18] * x2_01 * x1_00 * x0_10;
        y += W[19] * x2_01 * x1_00 * x0_11;
        y += W[20] * x2_01 * x1_01 * x0_00;
        y += W[21] * x2_01 * x1_01 * x0_01;
        y += W[22] * x2_01 * x1_01 * x0_10;
        y += W[23] * x2_01 * x1_01 * x0_11;
        y += W[24] * x2_01 * x1_10 * x0_00;
        y += W[25] * x2_01 * x1_10 * x0_01;
        y += W[26] * x2_01 * x1_10 * x0_10;
        y += W[27] * x2_01 * x1_10 * x0_11;
        y += W[28] * x2_01 * x1_11 * x0_00;
        y += W[29] * x2_01 * x1_11 * x0_01;
        y += W[30] * x2_01 * x1_11 * x0_10;
        y += W[31] * x2_01 * x1_11 * x0_11;
        y += W[32] * x2_10 * x1_00 * x0_00;
        y += W[33] * x2_10 * x1_00 * x0_01;
        y += W[34] * x2_10 * x1_00 * x0_10;
        y += W[35] * x2_10 * x1_00 * x0_11;
        y += W[36] * x2_10 * x1_01 * x0_00;
        y += W[37] * x2_10 * x1_01 * x0_01;
        y += W[38] * x2_10 * x1_01 * x0_10;
        y += W[39] * x2_10 * x1_01 * x0_11;
        y += W[40] * x2_10 * x1_10 * x0_00;
        y += W[41] * x2_10 * x1_10 * x0_01;
        y += W[42] * x2_10 * x1_10 * x0_10;
        y += W[43] * x2_10 * x1_10 * x0_11;
        y += W[44] * x2_10 * x1_11 * x0_00;
        y += W[45] * x2_10 * x1_11 * x0_01;
        y += W[46] * x2_10 * x1_11 * x0_10;
        y += W[47] * x2_10 * x1_11 * x0_11;
        y += W[48] * x2_11 * x1_00 * x0_00;
        y += W[49] * x2_11 * x1_00 * x0_01;
        y += W[50] * x2_11 * x1_00 * x0_10;
        y += W[51] * x2_11 * x1_00 * x0_11;
        y += W[52] * x2_11 * x1_01 * x0_00;
        y += W[53] * x2_11 * x1_01 * x0_01;
        y += W[54] * x2_11 * x1_01 * x0_10;
        y += W[55] * x2_11 * x1_01 * x0_11;
        y += W[56] * x2_11 * x1_10 * x0_00;
        y += W[57] * x2_11 * x1_10 * x0_01;
        y += W[58] * x2_11 * x1_10 * x0_10;
        y += W[59] * x2_11 * x1_10 * x0_11;
        y += W[60] * x2_11 * x1_11 * x0_00;
        y += W[61] * x2_11 * x1_11 * x0_01;
        y += W[62] * x2_11 * x1_11 * x0_10;
        y += W[63] * x2_11 * x1_11 * x0_11;
        
        // clamp
        y = max(0.0, y);
        y = min(1.0, y);
        
        y_ptr[frame] = y;
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

    dim3    block(512);
    dim3    grid(node_size);
    while ( frame_size < (int)block.x / 2 ) {
        block.x /= 2;
    }
    
    kernal_fp32_StochasticLut6_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
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


__device__ __forceinline__ float device_fp32_LocalSum(float v, float *buf)
{
    buf[threadIdx.x] = v;
    __syncthreads();

    // スレッド間集計
    int comb = 1;
    while (comb < blockDim.x) {
        int next = comb * 2;
        int mask = next - 1;
        if ((threadIdx.x & mask) == 0) {
            buf[threadIdx.x] += buf[threadIdx.x + comb];
        }
        comb = next;
        __syncthreads();
    }

    float sum = buf[0];
    __syncthreads();
    
    return sum;
}



// kernel
template<int THREAD_SIZE=256>
__global__ void kernal_fp32_StochasticLut6_Backward
        (
            float const     *x_buf,
            float const     *dy_buf,
            float           *dx_buf,
            int   const     *input_index,
            float const     *W_buf,
            float           *dW_buf,
            int             frame_size,
            int             frame_stride,
            int             binary_mode
        )
{
    __shared__ float buf[THREAD_SIZE];

    int node    = blockIdx.x;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    // initialize dW
    float dW[64];
    for ( int i = 0; i < 64; ++i) {
        dW[i] = 0;
    }

    __shared__ float    dW_prev[64];
    for ( int i = id; i < 64; i += id_step ) {
        dW_prev[i] = dW_buf[node * 64 + i];
    }

    // read W
    __shared__ float    W[64];
    for ( int i = id; i < 64; i += id_step ) {
        W[i] = W_buf[node * 64 + i];
        if ( binary_mode ) {
            W[i] = W[i] > 0.5 ? 1.0 : 0.0;
        }
    }
    
    // init pointer
    __shared__  float const *x_ptr[6];
    for ( int i = id; i < 6; i += id_step ) {
        int input_node = input_index[6*node + i];
        x_ptr[i]  = &x_buf[frame_stride * input_node];
    }
    float const *dy_ptr = &dy_buf[node*frame_stride];

    __syncthreads();

    for ( int frame = id; frame < frame_size; frame += id_step ) {
        float xp[6], xn[6];
        for ( int i = 0; i < 6; ++i) {
            xp[i] = x_ptr[i][frame];
            xp[i] = min(1.0, max(0.0, xp[i]));
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

        float grad = dy_ptr[frame];

        dW[0]  += x2_00 * x1_00 * x0_00 * grad;
        dW[1]  += x2_00 * x1_00 * x0_01 * grad;
        dW[2]  += x2_00 * x1_00 * x0_10 * grad;
        dW[3]  += x2_00 * x1_00 * x0_11 * grad;
        dW[4]  += x2_00 * x1_01 * x0_00 * grad;
        dW[5]  += x2_00 * x1_01 * x0_01 * grad;
        dW[6]  += x2_00 * x1_01 * x0_10 * grad;
        dW[7]  += x2_00 * x1_01 * x0_11 * grad;
        dW[8]  += x2_00 * x1_10 * x0_00 * grad;
        dW[9]  += x2_00 * x1_10 * x0_01 * grad;
        dW[10] += x2_00 * x1_10 * x0_10 * grad;
        dW[11] += x2_00 * x1_10 * x0_11 * grad;
        dW[12] += x2_00 * x1_11 * x0_00 * grad;
        dW[13] += x2_00 * x1_11 * x0_01 * grad;
        dW[14] += x2_00 * x1_11 * x0_10 * grad;
        dW[15] += x2_00 * x1_11 * x0_11 * grad;
        dW[16] += x2_01 * x1_00 * x0_00 * grad;
        dW[17] += x2_01 * x1_00 * x0_01 * grad;
        dW[18] += x2_01 * x1_00 * x0_10 * grad;
        dW[19] += x2_01 * x1_00 * x0_11 * grad;
        dW[20] += x2_01 * x1_01 * x0_00 * grad;
        dW[21] += x2_01 * x1_01 * x0_01 * grad;
        dW[22] += x2_01 * x1_01 * x0_10 * grad;
        dW[23] += x2_01 * x1_01 * x0_11 * grad;
        dW[24] += x2_01 * x1_10 * x0_00 * grad;
        dW[25] += x2_01 * x1_10 * x0_01 * grad;
        dW[26] += x2_01 * x1_10 * x0_10 * grad;
        dW[27] += x2_01 * x1_10 * x0_11 * grad;
        dW[28] += x2_01 * x1_11 * x0_00 * grad;
        dW[29] += x2_01 * x1_11 * x0_01 * grad;
        dW[30] += x2_01 * x1_11 * x0_10 * grad;
        dW[31] += x2_01 * x1_11 * x0_11 * grad;
        dW[32] += x2_10 * x1_00 * x0_00 * grad;
        dW[33] += x2_10 * x1_00 * x0_01 * grad;
        dW[34] += x2_10 * x1_00 * x0_10 * grad;
        dW[35] += x2_10 * x1_00 * x0_11 * grad;
        dW[36] += x2_10 * x1_01 * x0_00 * grad;
        dW[37] += x2_10 * x1_01 * x0_01 * grad;
        dW[38] += x2_10 * x1_01 * x0_10 * grad;
        dW[39] += x2_10 * x1_01 * x0_11 * grad;
        dW[40] += x2_10 * x1_10 * x0_00 * grad;
        dW[41] += x2_10 * x1_10 * x0_01 * grad;
        dW[42] += x2_10 * x1_10 * x0_10 * grad;
        dW[43] += x2_10 * x1_10 * x0_11 * grad;
        dW[44] += x2_10 * x1_11 * x0_00 * grad;
        dW[45] += x2_10 * x1_11 * x0_01 * grad;
        dW[46] += x2_10 * x1_11 * x0_10 * grad;
        dW[47] += x2_10 * x1_11 * x0_11 * grad;
        dW[48] += x2_11 * x1_00 * x0_00 * grad;
        dW[49] += x2_11 * x1_00 * x0_01 * grad;
        dW[50] += x2_11 * x1_00 * x0_10 * grad;
        dW[51] += x2_11 * x1_00 * x0_11 * grad;
        dW[52] += x2_11 * x1_01 * x0_00 * grad;
        dW[53] += x2_11 * x1_01 * x0_01 * grad;
        dW[54] += x2_11 * x1_01 * x0_10 * grad;
        dW[55] += x2_11 * x1_01 * x0_11 * grad;
        dW[56] += x2_11 * x1_10 * x0_00 * grad;
        dW[57] += x2_11 * x1_10 * x0_01 * grad;
        dW[58] += x2_11 * x1_10 * x0_10 * grad;
        dW[59] += x2_11 * x1_10 * x0_11 * grad;
        dW[60] += x2_11 * x1_11 * x0_00 * grad;
        dW[61] += x2_11 * x1_11 * x0_01 * grad;
        dW[62] += x2_11 * x1_11 * x0_10 * grad;
        dW[63] += x2_11 * x1_11 * x0_11 * grad;

        float dxi;
        float dx0_00 = 0;
        float dx0_01 = 0;
        float dx0_10 = 0;
        float dx0_11 = 0;
        float dx1_00 = 0;
        float dx1_01 = 0;
        float dx1_10 = 0;
        float dx1_11 = 0;
        float dx2_00 = 0;
        float dx2_01 = 0;
        float dx2_10 = 0;
        float dx2_11 = 0;
        dxi = W[ 0] * grad;  dx0_00 += dxi * x2_00 * x1_00;  dx1_00 += dxi * x2_00 * x0_00;  dx2_00 += dxi * x1_00 * x0_00;
        dxi = W[ 1] * grad;  dx0_01 += dxi * x2_00 * x1_00;  dx1_00 += dxi * x2_00 * x0_01;  dx2_00 += dxi * x1_00 * x0_01;
        dxi = W[ 2] * grad;  dx0_10 += dxi * x2_00 * x1_00;  dx1_00 += dxi * x2_00 * x0_10;  dx2_00 += dxi * x1_00 * x0_10;
        dxi = W[ 3] * grad;  dx0_11 += dxi * x2_00 * x1_00;  dx1_00 += dxi * x2_00 * x0_11;  dx2_00 += dxi * x1_00 * x0_11;
        dxi = W[ 4] * grad;  dx0_00 += dxi * x2_00 * x1_01;  dx1_01 += dxi * x2_00 * x0_00;  dx2_00 += dxi * x1_01 * x0_00;
        dxi = W[ 5] * grad;  dx0_01 += dxi * x2_00 * x1_01;  dx1_01 += dxi * x2_00 * x0_01;  dx2_00 += dxi * x1_01 * x0_01;
        dxi = W[ 6] * grad;  dx0_10 += dxi * x2_00 * x1_01;  dx1_01 += dxi * x2_00 * x0_10;  dx2_00 += dxi * x1_01 * x0_10;
        dxi = W[ 7] * grad;  dx0_11 += dxi * x2_00 * x1_01;  dx1_01 += dxi * x2_00 * x0_11;  dx2_00 += dxi * x1_01 * x0_11;
        dxi = W[ 8] * grad;  dx0_00 += dxi * x2_00 * x1_10;  dx1_10 += dxi * x2_00 * x0_00;  dx2_00 += dxi * x1_10 * x0_00;
        dxi = W[ 9] * grad;  dx0_01 += dxi * x2_00 * x1_10;  dx1_10 += dxi * x2_00 * x0_01;  dx2_00 += dxi * x1_10 * x0_01;
        dxi = W[10] * grad;  dx0_10 += dxi * x2_00 * x1_10;  dx1_10 += dxi * x2_00 * x0_10;  dx2_00 += dxi * x1_10 * x0_10;
        dxi = W[11] * grad;  dx0_11 += dxi * x2_00 * x1_10;  dx1_10 += dxi * x2_00 * x0_11;  dx2_00 += dxi * x1_10 * x0_11;
        dxi = W[12] * grad;  dx0_00 += dxi * x2_00 * x1_11;  dx1_11 += dxi * x2_00 * x0_00;  dx2_00 += dxi * x1_11 * x0_00;
        dxi = W[13] * grad;  dx0_01 += dxi * x2_00 * x1_11;  dx1_11 += dxi * x2_00 * x0_01;  dx2_00 += dxi * x1_11 * x0_01;
        dxi = W[14] * grad;  dx0_10 += dxi * x2_00 * x1_11;  dx1_11 += dxi * x2_00 * x0_10;  dx2_00 += dxi * x1_11 * x0_10;
        dxi = W[15] * grad;  dx0_11 += dxi * x2_00 * x1_11;  dx1_11 += dxi * x2_00 * x0_11;  dx2_00 += dxi * x1_11 * x0_11;
        dxi = W[16] * grad;  dx0_00 += dxi * x2_01 * x1_00;  dx1_00 += dxi * x2_01 * x0_00;  dx2_01 += dxi * x1_00 * x0_00;
        dxi = W[17] * grad;  dx0_01 += dxi * x2_01 * x1_00;  dx1_00 += dxi * x2_01 * x0_01;  dx2_01 += dxi * x1_00 * x0_01;
        dxi = W[18] * grad;  dx0_10 += dxi * x2_01 * x1_00;  dx1_00 += dxi * x2_01 * x0_10;  dx2_01 += dxi * x1_00 * x0_10;
        dxi = W[19] * grad;  dx0_11 += dxi * x2_01 * x1_00;  dx1_00 += dxi * x2_01 * x0_11;  dx2_01 += dxi * x1_00 * x0_11;
        dxi = W[20] * grad;  dx0_00 += dxi * x2_01 * x1_01;  dx1_01 += dxi * x2_01 * x0_00;  dx2_01 += dxi * x1_01 * x0_00;
        dxi = W[21] * grad;  dx0_01 += dxi * x2_01 * x1_01;  dx1_01 += dxi * x2_01 * x0_01;  dx2_01 += dxi * x1_01 * x0_01;
        dxi = W[22] * grad;  dx0_10 += dxi * x2_01 * x1_01;  dx1_01 += dxi * x2_01 * x0_10;  dx2_01 += dxi * x1_01 * x0_10;
        dxi = W[23] * grad;  dx0_11 += dxi * x2_01 * x1_01;  dx1_01 += dxi * x2_01 * x0_11;  dx2_01 += dxi * x1_01 * x0_11;
        dxi = W[24] * grad;  dx0_00 += dxi * x2_01 * x1_10;  dx1_10 += dxi * x2_01 * x0_00;  dx2_01 += dxi * x1_10 * x0_00;
        dxi = W[25] * grad;  dx0_01 += dxi * x2_01 * x1_10;  dx1_10 += dxi * x2_01 * x0_01;  dx2_01 += dxi * x1_10 * x0_01;
        dxi = W[26] * grad;  dx0_10 += dxi * x2_01 * x1_10;  dx1_10 += dxi * x2_01 * x0_10;  dx2_01 += dxi * x1_10 * x0_10;
        dxi = W[27] * grad;  dx0_11 += dxi * x2_01 * x1_10;  dx1_10 += dxi * x2_01 * x0_11;  dx2_01 += dxi * x1_10 * x0_11;
        dxi = W[28] * grad;  dx0_00 += dxi * x2_01 * x1_11;  dx1_11 += dxi * x2_01 * x0_00;  dx2_01 += dxi * x1_11 * x0_00;
        dxi = W[29] * grad;  dx0_01 += dxi * x2_01 * x1_11;  dx1_11 += dxi * x2_01 * x0_01;  dx2_01 += dxi * x1_11 * x0_01;
        dxi = W[30] * grad;  dx0_10 += dxi * x2_01 * x1_11;  dx1_11 += dxi * x2_01 * x0_10;  dx2_01 += dxi * x1_11 * x0_10;
        dxi = W[31] * grad;  dx0_11 += dxi * x2_01 * x1_11;  dx1_11 += dxi * x2_01 * x0_11;  dx2_01 += dxi * x1_11 * x0_11;
        dxi = W[32] * grad;  dx0_00 += dxi * x2_10 * x1_00;  dx1_00 += dxi * x2_10 * x0_00;  dx2_10 += dxi * x1_00 * x0_00;
        dxi = W[33] * grad;  dx0_01 += dxi * x2_10 * x1_00;  dx1_00 += dxi * x2_10 * x0_01;  dx2_10 += dxi * x1_00 * x0_01;
        dxi = W[34] * grad;  dx0_10 += dxi * x2_10 * x1_00;  dx1_00 += dxi * x2_10 * x0_10;  dx2_10 += dxi * x1_00 * x0_10;
        dxi = W[35] * grad;  dx0_11 += dxi * x2_10 * x1_00;  dx1_00 += dxi * x2_10 * x0_11;  dx2_10 += dxi * x1_00 * x0_11;
        dxi = W[36] * grad;  dx0_00 += dxi * x2_10 * x1_01;  dx1_01 += dxi * x2_10 * x0_00;  dx2_10 += dxi * x1_01 * x0_00;
        dxi = W[37] * grad;  dx0_01 += dxi * x2_10 * x1_01;  dx1_01 += dxi * x2_10 * x0_01;  dx2_10 += dxi * x1_01 * x0_01;
        dxi = W[38] * grad;  dx0_10 += dxi * x2_10 * x1_01;  dx1_01 += dxi * x2_10 * x0_10;  dx2_10 += dxi * x1_01 * x0_10;
        dxi = W[39] * grad;  dx0_11 += dxi * x2_10 * x1_01;  dx1_01 += dxi * x2_10 * x0_11;  dx2_10 += dxi * x1_01 * x0_11;
        dxi = W[40] * grad;  dx0_00 += dxi * x2_10 * x1_10;  dx1_10 += dxi * x2_10 * x0_00;  dx2_10 += dxi * x1_10 * x0_00;
        dxi = W[41] * grad;  dx0_01 += dxi * x2_10 * x1_10;  dx1_10 += dxi * x2_10 * x0_01;  dx2_10 += dxi * x1_10 * x0_01;
        dxi = W[42] * grad;  dx0_10 += dxi * x2_10 * x1_10;  dx1_10 += dxi * x2_10 * x0_10;  dx2_10 += dxi * x1_10 * x0_10;
        dxi = W[43] * grad;  dx0_11 += dxi * x2_10 * x1_10;  dx1_10 += dxi * x2_10 * x0_11;  dx2_10 += dxi * x1_10 * x0_11;
        dxi = W[44] * grad;  dx0_00 += dxi * x2_10 * x1_11;  dx1_11 += dxi * x2_10 * x0_00;  dx2_10 += dxi * x1_11 * x0_00;
        dxi = W[45] * grad;  dx0_01 += dxi * x2_10 * x1_11;  dx1_11 += dxi * x2_10 * x0_01;  dx2_10 += dxi * x1_11 * x0_01;
        dxi = W[46] * grad;  dx0_10 += dxi * x2_10 * x1_11;  dx1_11 += dxi * x2_10 * x0_10;  dx2_10 += dxi * x1_11 * x0_10;
        dxi = W[47] * grad;  dx0_11 += dxi * x2_10 * x1_11;  dx1_11 += dxi * x2_10 * x0_11;  dx2_10 += dxi * x1_11 * x0_11;
        dxi = W[48] * grad;  dx0_00 += dxi * x2_11 * x1_00;  dx1_00 += dxi * x2_11 * x0_00;  dx2_11 += dxi * x1_00 * x0_00;
        dxi = W[49] * grad;  dx0_01 += dxi * x2_11 * x1_00;  dx1_00 += dxi * x2_11 * x0_01;  dx2_11 += dxi * x1_00 * x0_01;
        dxi = W[50] * grad;  dx0_10 += dxi * x2_11 * x1_00;  dx1_00 += dxi * x2_11 * x0_10;  dx2_11 += dxi * x1_00 * x0_10;
        dxi = W[51] * grad;  dx0_11 += dxi * x2_11 * x1_00;  dx1_00 += dxi * x2_11 * x0_11;  dx2_11 += dxi * x1_00 * x0_11;
        dxi = W[52] * grad;  dx0_00 += dxi * x2_11 * x1_01;  dx1_01 += dxi * x2_11 * x0_00;  dx2_11 += dxi * x1_01 * x0_00;
        dxi = W[53] * grad;  dx0_01 += dxi * x2_11 * x1_01;  dx1_01 += dxi * x2_11 * x0_01;  dx2_11 += dxi * x1_01 * x0_01;
        dxi = W[54] * grad;  dx0_10 += dxi * x2_11 * x1_01;  dx1_01 += dxi * x2_11 * x0_10;  dx2_11 += dxi * x1_01 * x0_10;
        dxi = W[55] * grad;  dx0_11 += dxi * x2_11 * x1_01;  dx1_01 += dxi * x2_11 * x0_11;  dx2_11 += dxi * x1_01 * x0_11;
        dxi = W[56] * grad;  dx0_00 += dxi * x2_11 * x1_10;  dx1_10 += dxi * x2_11 * x0_00;  dx2_11 += dxi * x1_10 * x0_00;
        dxi = W[57] * grad;  dx0_01 += dxi * x2_11 * x1_10;  dx1_10 += dxi * x2_11 * x0_01;  dx2_11 += dxi * x1_10 * x0_01;
        dxi = W[58] * grad;  dx0_10 += dxi * x2_11 * x1_10;  dx1_10 += dxi * x2_11 * x0_10;  dx2_11 += dxi * x1_10 * x0_10;
        dxi = W[59] * grad;  dx0_11 += dxi * x2_11 * x1_10;  dx1_10 += dxi * x2_11 * x0_11;  dx2_11 += dxi * x1_10 * x0_11;
        dxi = W[60] * grad;  dx0_00 += dxi * x2_11 * x1_11;  dx1_11 += dxi * x2_11 * x0_00;  dx2_11 += dxi * x1_11 * x0_00;
        dxi = W[61] * grad;  dx0_01 += dxi * x2_11 * x1_11;  dx1_11 += dxi * x2_11 * x0_01;  dx2_11 += dxi * x1_11 * x0_01;
        dxi = W[62] * grad;  dx0_10 += dxi * x2_11 * x1_11;  dx1_11 += dxi * x2_11 * x0_10;  dx2_11 += dxi * x1_11 * x0_10;
        dxi = W[63] * grad;  dx0_11 += dxi * x2_11 * x1_11;  dx1_11 += dxi * x2_11 * x0_11;  dx2_11 += dxi * x1_11 * x0_11;
        
        float *dx_ptr = &dx_buf[(node*6)*frame_stride + frame];
        float dxn;
        float dxp;
        float dx;
        dxn  = dx0_00 * xn[1];    dxn += dx0_10 * xp[1];
        dxp  = dx0_01 * xn[1];    dxp += dx0_11 * xp[1];
        dx = dxp - dxn;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[0 * frame_stride] = dx;

        dxn  = dx0_00 * xn[0];
        dxn += dx0_01 * xp[0];
        dxp  = dx0_10 * xn[0];
        dxp += dx0_11 * xp[0];
        dx = dxp - dxn;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[1 * frame_stride] = dx;

        dxn  = dx1_00 * xn[3];     
        dxp  = dx1_01 * xn[3];     
        dxn += dx1_10 * xp[3];     
        dxp += dx1_11 * xp[3];     
        dx = dxp - dxn;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[2 * frame_stride] = dx;

        dxn  = dx1_00 * xn[2];
        dxn += dx1_01 * xp[2];
        dxp  = dx1_10 * xn[2];
        dxp += dx1_11 * xp[2];
        dx = dxp - dxn;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[3 * frame_stride] = dx;

        dxn  = dx2_00 * xn[5];     
        dxp  = dx2_01 * xn[5];     
        dxn += dx2_10 * xp[5];     
        dxp += dx2_11 * xp[5];     
        dx = dxp - dxn;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[4 * frame_stride] = dx;

        dxn  = dx2_00 * xn[4];
        dxn += dx2_01 * xp[4];
        dxp  = dx2_10 * xn[4];
        dxp += dx2_11 * xp[4];
        dx = dxp - dxn;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[5 * frame_stride] = dx;
    }

    for ( int i = 0; i < 64; ++i) {
        dW[i] = device_fp32_LocalSum(dW[i], buf);
    }
    if ( id == 0 ) {
        for ( int i = 0; i < 64; ++i) {
            dW_buf[node*64 + i] = dW[i] + dW_prev[i];
        }
    }
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
    int frame = blockDim.x * blockIdx.x + threadIdx.x;

    for ( int node = 0; node < node_size; ++node ) {
        if ( frame < frame_size ) {
            for ( int n = 0; n < 6; ++n ) {
                int in_idx = input_index[node*6 + n];
                float*       dst_buf_ptr = &dst_buf[frame_stride * in_idx];
                float        prev_data = dst_buf_ptr[frame];
                const float* src_buf_ptr = &src_buf[(6 * node + n) * frame_stride];
                
                dst_buf_ptr[frame] = prev_data + src_buf_ptr[frame];
            }
        }
        __syncthreads();
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
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             binary_mode,
            cudaStream_t    streamId
    )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    {
        int const thread_size = 256;
        dim3    block(thread_size);
        dim3    grid(output_node_size);
        while ( frame_size < (int)block.x / 2 ) {
            block.x /= 2;
        }

        kernal_fp32_StochasticLut6_Backward<thread_size><<<grid, block, 0, streamId>>>(
                dev_x_buf,
                dev_dy_buf,
                dev_dx_tmp,
                dev_input_index,
                dev_W,
                dev_dW,
                frame_size,
                frame_stride,
                binary_mode
            );
        BB_CUDA_CHECK_LAST_ERROR();
    }
    

    {
        BB_CUDA_SAFE_CALL(cudaMemset(dev_dx_buf, 0, input_node_size * frame_stride * sizeof(float)));

        int block_x = frame_size;
        while ( block_x > 1024 ) { block_x /= 2; }

        dim3    grid((frame_size + block_x - 1) /block_x, 1);
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





// end of file
