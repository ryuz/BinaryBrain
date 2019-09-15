#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


// kernel
template<int BUF_SIZE>
__global__ void kernel_fp32_MatrixColwiseMeanVar(
    const float*    src,
    float*          mean,
    float*          variance,
    int             frame_size,
    int             frame_stride)
{
    __shared__   float   buf[BUF_SIZE];
    

    // 初期化
    int node  = blockIdx.x;
    int frame = threadIdx.x;
    int frame_step = blockDim.x;
    
    
    // カハンの加算アルゴリズム(Kahan summation algorithm)
    float s1 = 0, c1 = 0, y1, t1;
    float s2 = 0, c2 = 0, y2, t2;
    const float* src_ptr = &src[frame_stride * node];
    while ( frame < frame_size ) {
        float x = src_ptr[frame];
        
        y1 = x - c1;
        t1 = s1 + y1;
        c1 += (t1 - s1) - y1;
        s1 = t1;
        
        y2 = (x * x) - c2;
        t2 = s2 + y2;
        c2 += (t2 - s2) - y2;
        s2 = t2;

        frame += frame_step;
    }
    
    float* buf1 = &buf[0];
    float* buf2 = &buf[blockDim.x];
    
    buf1[threadIdx.x] = s1;
    buf2[threadIdx.x] = s2;
    
    __syncthreads();

    // スレッド間集計
    int comb = 1;
    while (comb < frame_step) {
        int next = comb * 2;
        int mask = next - 1;
        if ((threadIdx.x & mask) == 0) {
            buf1[threadIdx.x] += buf1[threadIdx.x + comb];
            buf2[threadIdx.x] += buf2[threadIdx.x + comb];
        }
        comb = next;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float m = buf1[0] / frame_size;
        float v = max(0.0f, (buf2[0] / frame_size) - (m * m));

        mean[node] = m;
        variance[node] = v;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_MatrixColwiseMeanVar
(
    const float*    dev_src,
    float*          dev_mean,
    float*          dev_variance,
    int             node_size,
    int             frame_size,
    int             frame_stride,
    cudaStream_t    streamId
)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());
    
    int const   unit_x = 512;
    
    dim3    grid(node_size);
    dim3    block(unit_x);
    
    kernel_fp32_MatrixColwiseMeanVar<2 * unit_x><< <grid, block, 0, streamId >> > (
        dev_src,
        dev_mean,
        dev_variance,
        frame_size,
        frame_stride);
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


