#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


#define THREAD_X_UNIT   512

// kernel
__global__ void kernel_fp32_MatrixColwiseSum(
            const float*    x_mat,
            float*          y_vec,
            int             frame_size,
            int             frame_stride)
{
    __shared__   float   buf[THREAD_X_UNIT];

    // 初期化
    int node       = blockIdx.x;
    int frame_base = threadIdx.x;
    int frame_step = blockDim.x;

    // 読み込み
    float acc = 0;
    const float* x_ptr = &x_mat[frame_stride * node];
    for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
        acc += x_ptr[frame];
    }
    buf[threadIdx.x] = acc;

    __syncthreads();

    int x = threadIdx.x;
    int comb = 1;
    while ( comb < blockDim.x ) {
        int next = comb * 2;
        int mask = next - 1;
        if ( (x & mask) == 0 ) {
            buf[x] += buf[x + comb];
        }
        comb = next;
        __syncthreads();
    }
    
    if ( threadIdx.x == 0 ) {
        y_vec[node] += buf[0];
    }
}


int bbcu_fp32_MatrixColwiseSum
        (
            const float*    dev_x_mat,
            float*          dev_y_vec,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    grid(node_size);
    dim3    block(THREAD_X_UNIT);    
    kernel_fp32_MatrixColwiseSum<<<grid, block, 0, streamId>>>(
            dev_x_mat,
            dev_y_vec,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}


