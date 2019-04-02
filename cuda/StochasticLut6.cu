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
    
#if 1
    __shared__ float    W[64];
    for ( int i = frame_base; i < 64; i += frame_step ) {
        W[i] = W_buf[node * 64 + i];
        if ( binary_mode ) {
            W[i] = W[i] > 0.5 ? 1.0 : 0.0;
        }
    }
    __syncthreads();

#else
    float        W[64];
    for ( int i = 0; i < 64; ++i) {
        W[i] = W_buf[node * 64 + i];
        if ( binary_mode ) {
            W[i] = W[i] > 0.5 ? 1.0 : 0.0;
        }
    }
#endif
    
    if (node >= node_size) {
        return;
    }
    
    float const  *x_ptr[6];
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

        /*
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
        */

        float sig = 0;
        sig += W[0 ] * x2_00 * x1_00 * x0_00;
        sig += W[1 ] * x2_00 * x1_00 * x0_01;
        sig += W[2 ] * x2_00 * x1_00 * x0_10;
        sig += W[3 ] * x2_00 * x1_00 * x0_11;
        sig += W[4 ] * x2_00 * x1_01 * x0_00;
        sig += W[5 ] * x2_00 * x1_01 * x0_01;
        sig += W[6 ] * x2_00 * x1_01 * x0_10;
        sig += W[7 ] * x2_00 * x1_01 * x0_11;
        sig += W[8 ] * x2_00 * x1_10 * x0_00;
        sig += W[9 ] * x2_00 * x1_10 * x0_01;
        sig += W[10] * x2_00 * x1_10 * x0_10;
        sig += W[11] * x2_00 * x1_10 * x0_11;
        sig += W[12] * x2_00 * x1_11 * x0_00;
        sig += W[13] * x2_00 * x1_11 * x0_01;
        sig += W[14] * x2_00 * x1_11 * x0_10;
        sig += W[15] * x2_00 * x1_11 * x0_11;
        sig += W[16] * x2_01 * x1_00 * x0_00;
        sig += W[17] * x2_01 * x1_00 * x0_01;
        sig += W[18] * x2_01 * x1_00 * x0_10;
        sig += W[19] * x2_01 * x1_00 * x0_11;
        sig += W[20] * x2_01 * x1_01 * x0_00;
        sig += W[21] * x2_01 * x1_01 * x0_01;
        sig += W[22] * x2_01 * x1_01 * x0_10;
        sig += W[23] * x2_01 * x1_01 * x0_11;
        sig += W[24] * x2_01 * x1_10 * x0_00;
        sig += W[25] * x2_01 * x1_10 * x0_01;
        sig += W[26] * x2_01 * x1_10 * x0_10;
        sig += W[27] * x2_01 * x1_10 * x0_11;
        sig += W[28] * x2_01 * x1_11 * x0_00;
        sig += W[29] * x2_01 * x1_11 * x0_01;
        sig += W[30] * x2_01 * x1_11 * x0_10;
        sig += W[31] * x2_01 * x1_11 * x0_11;
        sig += W[32] * x2_10 * x1_00 * x0_00;
        sig += W[33] * x2_10 * x1_00 * x0_01;
        sig += W[34] * x2_10 * x1_00 * x0_10;
        sig += W[35] * x2_10 * x1_00 * x0_11;
        sig += W[36] * x2_10 * x1_01 * x0_00;
        sig += W[37] * x2_10 * x1_01 * x0_01;
        sig += W[38] * x2_10 * x1_01 * x0_10;
        sig += W[39] * x2_10 * x1_01 * x0_11;
        sig += W[40] * x2_10 * x1_10 * x0_00;
        sig += W[41] * x2_10 * x1_10 * x0_01;
        sig += W[42] * x2_10 * x1_10 * x0_10;
        sig += W[43] * x2_10 * x1_10 * x0_11;
        sig += W[44] * x2_10 * x1_11 * x0_00;
        sig += W[45] * x2_10 * x1_11 * x0_01;
        sig += W[46] * x2_10 * x1_11 * x0_10;
        sig += W[47] * x2_10 * x1_11 * x0_11;
        sig += W[48] * x2_11 * x1_00 * x0_00;
        sig += W[49] * x2_11 * x1_00 * x0_01;
        sig += W[50] * x2_11 * x1_00 * x0_10;
        sig += W[51] * x2_11 * x1_00 * x0_11;
        sig += W[52] * x2_11 * x1_01 * x0_00;
        sig += W[53] * x2_11 * x1_01 * x0_01;
        sig += W[54] * x2_11 * x1_01 * x0_10;
        sig += W[55] * x2_11 * x1_01 * x0_11;
        sig += W[56] * x2_11 * x1_10 * x0_00;
        sig += W[57] * x2_11 * x1_10 * x0_01;
        sig += W[58] * x2_11 * x1_10 * x0_10;
        sig += W[59] * x2_11 * x1_10 * x0_11;
        sig += W[60] * x2_11 * x1_11 * x0_00;
        sig += W[61] * x2_11 * x1_11 * x0_01;
        sig += W[62] * x2_11 * x1_11 * x0_10;
        sig += W[63] * x2_11 * x1_11 * x0_11;
        
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

    dim3    block(512, 1);
    while ( frame_size < (int)block.x / 2 ) {
        block.x /= 2;
        block.y *= 2;
    }
    block.y = std::min((int)block.y, node_size);

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
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             binary_mode
        )
{
    __shared__ float buf[THREAD_SIZE];

    int node       = blockIdx.y * blockDim.y + threadIdx.y;
    int frame_step = blockDim.x;
    int frame_base = threadIdx.x;
   

    float W[64];
    float dW[64];
    for ( int i = 0; i < 64; ++i) {
        W[i] = W_buf[node * 64 + i];
        if ( binary_mode ) {
            W[i] = W[i] > 0.5 ? 1.0 : 0.0;
        }
        dW[i] = 0;
    }
    
    float const *x_ptr[6];
    for ( int i = 0; i < 6; ++i) {
        int input_node = input_index[6*node + i];
        x_ptr[i]  = &x_buf[frame_stride * input_node];
    }

    float const *dy_ptr = &dy_buf[node*frame_stride];

    if ( node < node_size) {
        for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
            float xp[6], xn[6];
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

            float grad = dy_ptr[frame];

            float dxi[64];
		    for ( int i = 0; i < 64; ++i) {
			    dW[i]  += xi[i] * grad;
			    dxi[i]  = W[i]  * grad;
		    }

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
            dx0_00 += dxi[0]  * x2_00 * x1_00;  dx1_00 += dxi[0]  * x2_00 * x0_00;  dx2_00 += dxi[0]  * x1_00 * x0_00;
            dx0_01 += dxi[1]  * x2_00 * x1_00;  dx1_00 += dxi[1]  * x2_00 * x0_01;  dx2_00 += dxi[1]  * x1_00 * x0_01;
            dx0_10 += dxi[2]  * x2_00 * x1_00;  dx1_00 += dxi[2]  * x2_00 * x0_10;  dx2_00 += dxi[2]  * x1_00 * x0_10;
            dx0_11 += dxi[3]  * x2_00 * x1_00;  dx1_00 += dxi[3]  * x2_00 * x0_11;  dx2_00 += dxi[3]  * x1_00 * x0_11;
            dx0_00 += dxi[4]  * x2_00 * x1_01;  dx1_01 += dxi[4]  * x2_00 * x0_00;  dx2_00 += dxi[4]  * x1_01 * x0_00;
            dx0_01 += dxi[5]  * x2_00 * x1_01;  dx1_01 += dxi[5]  * x2_00 * x0_01;  dx2_00 += dxi[5]  * x1_01 * x0_01;
            dx0_10 += dxi[6]  * x2_00 * x1_01;  dx1_01 += dxi[6]  * x2_00 * x0_10;  dx2_00 += dxi[6]  * x1_01 * x0_10;
            dx0_11 += dxi[7]  * x2_00 * x1_01;  dx1_01 += dxi[7]  * x2_00 * x0_11;  dx2_00 += dxi[7]  * x1_01 * x0_11;
            dx0_00 += dxi[8]  * x2_00 * x1_10;  dx1_10 += dxi[8]  * x2_00 * x0_00;  dx2_00 += dxi[8]  * x1_10 * x0_00;
            dx0_01 += dxi[9]  * x2_00 * x1_10;  dx1_10 += dxi[9]  * x2_00 * x0_01;  dx2_00 += dxi[9]  * x1_10 * x0_01;
            dx0_10 += dxi[10] * x2_00 * x1_10;  dx1_10 += dxi[10] * x2_00 * x0_10;  dx2_00 += dxi[10] * x1_10 * x0_10;
            dx0_11 += dxi[11] * x2_00 * x1_10;  dx1_10 += dxi[11] * x2_00 * x0_11;  dx2_00 += dxi[11] * x1_10 * x0_11;
            dx0_00 += dxi[12] * x2_00 * x1_11;  dx1_11 += dxi[12] * x2_00 * x0_00;  dx2_00 += dxi[12] * x1_11 * x0_00;
            dx0_01 += dxi[13] * x2_00 * x1_11;  dx1_11 += dxi[13] * x2_00 * x0_01;  dx2_00 += dxi[13] * x1_11 * x0_01;
            dx0_10 += dxi[14] * x2_00 * x1_11;  dx1_11 += dxi[14] * x2_00 * x0_10;  dx2_00 += dxi[14] * x1_11 * x0_10;
            dx0_11 += dxi[15] * x2_00 * x1_11;  dx1_11 += dxi[15] * x2_00 * x0_11;  dx2_00 += dxi[15] * x1_11 * x0_11;
            dx0_00 += dxi[16] * x2_01 * x1_00;  dx1_00 += dxi[16] * x2_01 * x0_00;  dx2_01 += dxi[16] * x1_00 * x0_00;
            dx0_01 += dxi[17] * x2_01 * x1_00;  dx1_00 += dxi[17] * x2_01 * x0_01;  dx2_01 += dxi[17] * x1_00 * x0_01;
            dx0_10 += dxi[18] * x2_01 * x1_00;  dx1_00 += dxi[18] * x2_01 * x0_10;  dx2_01 += dxi[18] * x1_00 * x0_10;
            dx0_11 += dxi[19] * x2_01 * x1_00;  dx1_00 += dxi[19] * x2_01 * x0_11;  dx2_01 += dxi[19] * x1_00 * x0_11;
            dx0_00 += dxi[20] * x2_01 * x1_01;  dx1_01 += dxi[20] * x2_01 * x0_00;  dx2_01 += dxi[20] * x1_01 * x0_00;
            dx0_01 += dxi[21] * x2_01 * x1_01;  dx1_01 += dxi[21] * x2_01 * x0_01;  dx2_01 += dxi[21] * x1_01 * x0_01;
            dx0_10 += dxi[22] * x2_01 * x1_01;  dx1_01 += dxi[22] * x2_01 * x0_10;  dx2_01 += dxi[22] * x1_01 * x0_10;
            dx0_11 += dxi[23] * x2_01 * x1_01;  dx1_01 += dxi[23] * x2_01 * x0_11;  dx2_01 += dxi[23] * x1_01 * x0_11;
            dx0_00 += dxi[24] * x2_01 * x1_10;  dx1_10 += dxi[24] * x2_01 * x0_00;  dx2_01 += dxi[24] * x1_10 * x0_00;
            dx0_01 += dxi[25] * x2_01 * x1_10;  dx1_10 += dxi[25] * x2_01 * x0_01;  dx2_01 += dxi[25] * x1_10 * x0_01;
            dx0_10 += dxi[26] * x2_01 * x1_10;  dx1_10 += dxi[26] * x2_01 * x0_10;  dx2_01 += dxi[26] * x1_10 * x0_10;
            dx0_11 += dxi[27] * x2_01 * x1_10;  dx1_10 += dxi[27] * x2_01 * x0_11;  dx2_01 += dxi[27] * x1_10 * x0_11;
            dx0_00 += dxi[28] * x2_01 * x1_11;  dx1_11 += dxi[28] * x2_01 * x0_00;  dx2_01 += dxi[28] * x1_11 * x0_00;
            dx0_01 += dxi[29] * x2_01 * x1_11;  dx1_11 += dxi[29] * x2_01 * x0_01;  dx2_01 += dxi[29] * x1_11 * x0_01;
            dx0_10 += dxi[30] * x2_01 * x1_11;  dx1_11 += dxi[30] * x2_01 * x0_10;  dx2_01 += dxi[30] * x1_11 * x0_10;
            dx0_11 += dxi[31] * x2_01 * x1_11;  dx1_11 += dxi[31] * x2_01 * x0_11;  dx2_01 += dxi[31] * x1_11 * x0_11;
            dx0_00 += dxi[32] * x2_10 * x1_00;  dx1_00 += dxi[32] * x2_10 * x0_00;  dx2_10 += dxi[32] * x1_00 * x0_00;
            dx0_01 += dxi[33] * x2_10 * x1_00;  dx1_00 += dxi[33] * x2_10 * x0_01;  dx2_10 += dxi[33] * x1_00 * x0_01;
            dx0_10 += dxi[34] * x2_10 * x1_00;  dx1_00 += dxi[34] * x2_10 * x0_10;  dx2_10 += dxi[34] * x1_00 * x0_10;
            dx0_11 += dxi[35] * x2_10 * x1_00;  dx1_00 += dxi[35] * x2_10 * x0_11;  dx2_10 += dxi[35] * x1_00 * x0_11;
            dx0_00 += dxi[36] * x2_10 * x1_01;  dx1_01 += dxi[36] * x2_10 * x0_00;  dx2_10 += dxi[36] * x1_01 * x0_00;
            dx0_01 += dxi[37] * x2_10 * x1_01;  dx1_01 += dxi[37] * x2_10 * x0_01;  dx2_10 += dxi[37] * x1_01 * x0_01;
            dx0_10 += dxi[38] * x2_10 * x1_01;  dx1_01 += dxi[38] * x2_10 * x0_10;  dx2_10 += dxi[38] * x1_01 * x0_10;
            dx0_11 += dxi[39] * x2_10 * x1_01;  dx1_01 += dxi[39] * x2_10 * x0_11;  dx2_10 += dxi[39] * x1_01 * x0_11;
            dx0_00 += dxi[40] * x2_10 * x1_10;  dx1_10 += dxi[40] * x2_10 * x0_00;  dx2_10 += dxi[40] * x1_10 * x0_00;
            dx0_01 += dxi[41] * x2_10 * x1_10;  dx1_10 += dxi[41] * x2_10 * x0_01;  dx2_10 += dxi[41] * x1_10 * x0_01;
            dx0_10 += dxi[42] * x2_10 * x1_10;  dx1_10 += dxi[42] * x2_10 * x0_10;  dx2_10 += dxi[42] * x1_10 * x0_10;
            dx0_11 += dxi[43] * x2_10 * x1_10;  dx1_10 += dxi[43] * x2_10 * x0_11;  dx2_10 += dxi[43] * x1_10 * x0_11;
            dx0_00 += dxi[44] * x2_10 * x1_11;  dx1_11 += dxi[44] * x2_10 * x0_00;  dx2_10 += dxi[44] * x1_11 * x0_00;
            dx0_01 += dxi[45] * x2_10 * x1_11;  dx1_11 += dxi[45] * x2_10 * x0_01;  dx2_10 += dxi[45] * x1_11 * x0_01;
            dx0_10 += dxi[46] * x2_10 * x1_11;  dx1_11 += dxi[46] * x2_10 * x0_10;  dx2_10 += dxi[46] * x1_11 * x0_10;
            dx0_11 += dxi[47] * x2_10 * x1_11;  dx1_11 += dxi[47] * x2_10 * x0_11;  dx2_10 += dxi[47] * x1_11 * x0_11;
            dx0_00 += dxi[48] * x2_11 * x1_00;  dx1_00 += dxi[48] * x2_11 * x0_00;  dx2_11 += dxi[48] * x1_00 * x0_00;
            dx0_01 += dxi[49] * x2_11 * x1_00;  dx1_00 += dxi[49] * x2_11 * x0_01;  dx2_11 += dxi[49] * x1_00 * x0_01;
            dx0_10 += dxi[50] * x2_11 * x1_00;  dx1_00 += dxi[50] * x2_11 * x0_10;  dx2_11 += dxi[50] * x1_00 * x0_10;
            dx0_11 += dxi[51] * x2_11 * x1_00;  dx1_00 += dxi[51] * x2_11 * x0_11;  dx2_11 += dxi[51] * x1_00 * x0_11;
            dx0_00 += dxi[52] * x2_11 * x1_01;  dx1_01 += dxi[52] * x2_11 * x0_00;  dx2_11 += dxi[52] * x1_01 * x0_00;
            dx0_01 += dxi[53] * x2_11 * x1_01;  dx1_01 += dxi[53] * x2_11 * x0_01;  dx2_11 += dxi[53] * x1_01 * x0_01;
            dx0_10 += dxi[54] * x2_11 * x1_01;  dx1_01 += dxi[54] * x2_11 * x0_10;  dx2_11 += dxi[54] * x1_01 * x0_10;
            dx0_11 += dxi[55] * x2_11 * x1_01;  dx1_01 += dxi[55] * x2_11 * x0_11;  dx2_11 += dxi[55] * x1_01 * x0_11;
            dx0_00 += dxi[56] * x2_11 * x1_10;  dx1_10 += dxi[56] * x2_11 * x0_00;  dx2_11 += dxi[56] * x1_10 * x0_00;
            dx0_01 += dxi[57] * x2_11 * x1_10;  dx1_10 += dxi[57] * x2_11 * x0_01;  dx2_11 += dxi[57] * x1_10 * x0_01;
            dx0_10 += dxi[58] * x2_11 * x1_10;  dx1_10 += dxi[58] * x2_11 * x0_10;  dx2_11 += dxi[58] * x1_10 * x0_10;
            dx0_11 += dxi[59] * x2_11 * x1_10;  dx1_10 += dxi[59] * x2_11 * x0_11;  dx2_11 += dxi[59] * x1_10 * x0_11;
            dx0_00 += dxi[60] * x2_11 * x1_11;  dx1_11 += dxi[60] * x2_11 * x0_00;  dx2_11 += dxi[60] * x1_11 * x0_00;
            dx0_01 += dxi[61] * x2_11 * x1_11;  dx1_11 += dxi[61] * x2_11 * x0_01;  dx2_11 += dxi[61] * x1_11 * x0_01;
            dx0_10 += dxi[62] * x2_11 * x1_11;  dx1_11 += dxi[62] * x2_11 * x0_10;  dx2_11 += dxi[62] * x1_11 * x0_10;
            dx0_11 += dxi[63] * x2_11 * x1_11;  dx1_11 += dxi[63] * x2_11 * x0_11;  dx2_11 += dxi[63] * x1_11 * x0_11;


            float dxn[6] = {0};
            float dxp[6] = {0};
            dxn[0] += dx0_00 * xn[1];     dxn[1] += dx0_00 * xn[0];
            dxp[0] += dx0_01 * xn[1];     dxn[1] += dx0_01 * xp[0];
            dxn[0] += dx0_10 * xp[1];     dxp[1] += dx0_10 * xn[0];
            dxp[0] += dx0_11 * xp[1];     dxp[1] += dx0_11 * xp[0];
            dxn[2] += dx1_00 * xn[3];     dxn[3] += dx1_00 * xn[2];
            dxp[2] += dx1_01 * xn[3];     dxn[3] += dx1_01 * xp[2];
            dxn[2] += dx1_10 * xp[3];     dxp[3] += dx1_10 * xn[2];
            dxp[2] += dx1_11 * xp[3];     dxp[3] += dx1_11 * xp[2];
            dxn[4] += dx2_00 * xn[5];     dxn[5] += dx2_00 * xn[4];
            dxp[4] += dx2_01 * xn[5];     dxn[5] += dx2_01 * xp[4];
            dxn[4] += dx2_10 * xp[5];     dxp[5] += dx2_10 * xn[4];
            dxp[4] += dx2_11 * xp[5];     dxp[5] += dx2_11 * xp[4];

    	    for ( int i = 0; i < 6; ++i) {
                dx_buf[(node*6 + i)*frame_stride + frame] = (dxp[i] - dxn[i]);
            }
	    }
    }

    for ( int i = 0; i < 64; ++i) {
        dW[i] = device_fp32_LocalSum(dW[i], buf);
        if ( threadIdx.x == 0 ) {
            dW_buf[node*64 + i] = dW[i];
        }
    }
}


__global__ void kernal_fp32_StochasticLut6_BackwardMarge(
			const float*	src_buf,
			float*			dst_buf,
			const int*		input_index,
			int				node_size,
			int				frame_size,
			int				frame_stride
		)
{
//	int n          = blockDim.y * blockIdx.y + threadIdx.y;
	int frame      = blockDim.x * blockIdx.x + threadIdx.x;
	

	for ( int node = 0; node < node_size; ++node ) {
        if ( frame < frame_size ) {
    	    for ( int n = 0; n < 6; ++n ) {
		        int in_idx = input_index[node*6 + n];
		        float*		 dst_buf_ptr = &dst_buf[frame_stride * in_idx];
		        const float* src_buf_ptr = &src_buf[(6 * node + n) * frame_stride];
		
		        dst_buf_ptr[frame] += src_buf_ptr[frame];
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
//        dim3    block;
//        block.x = std::min(192, frame_size);
//        block.y = std::min(192, output_node_size);
//        while (block.x * block.y > 192) {
//            block.y = (block.y + 1) / block.y;
//        }
//        dim3    grid;
//        grid.x = 1;
//        grid.y = (output_node_size + (block.y - 1)) / block.y;
        
        int const thread_size = 256;
        dim3    block(thread_size, 1);
        while ( frame_size < (int)block.x / 2 ) {
            block.x /= 2;
            block.y *= 2;
        }
        block.y = std::min((int)block.y, output_node_size);

        dim3    grid;
        grid.x = 1;
        grid.y = (output_node_size + (block.y - 1)) / block.y;

        kernal_fp32_StochasticLut6_Backward<thread_size><<<grid, block, 0, streamId>>>(
                dev_x_buf,
                dev_dy_buf,
                dev_dx_tmp,
                dev_input_index,
                dev_W,
                dev_dW,
                output_node_size,
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
