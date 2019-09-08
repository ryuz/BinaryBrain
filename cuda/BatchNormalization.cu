#include <iostream>
#include <chrono>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#include "Common.cuh"



#define BBCU_BATCHNORM_FW_BLOCK_SIZE   128
#define BBCU_BATCHNORM_BW_BLOCK_SIZE   128


#if 0
//////////////////////////////
// common
//////////////////////////////

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
#endif



//////////////////////////////
// forward training
//////////////////////////////

__global__ void kernal_fp32_BatchNormalization_ForwardTraining(
            const float     *x_buf,
            float           *y_buf,
            float const     *gamma_buf,
            float const     *beta_buf,
            float           *mean_buf,
            float           *rstd_buf,
            float           *running_mean_buf,
            float           *running_var_buf,
            float           momentum,
            float           reciprocal_frame_size,
            int             frame_size,
            int             frame_stride
        )
{
    __shared__   float  buf[BBCU_BATCHNORM_FW_BLOCK_SIZE];

    // 初期化
    int const node    = blockIdx.x;
    int const id      = threadIdx.x;
    int const id_step = blockDim.x;
    
    // カハンの加算アルゴリズム(Kahan summation algorithm)
    float s1 = 0, c1 = 0, y1, t1;
    float s2 = 0, c2 = 0, y2, t2;
    const float* x_ptr = &x_buf[frame_stride * node];
    for ( int frame = id; frame < frame_size; frame += id_step) {
        float x = x_ptr[frame];

        y1 = x - c1;
        t1 = s1 + y1;
        c1 = (t1 - s1) - y1;
        s1 = t1;

        y2 = (x * x) - c2;
        t2 = s2 + y2;
        c2 = (t2 - s2) - y2;
        s2 = t2;
    }

    // 集計
    s1 = device_fp32_LocalSum(s1, buf);
    s2 = device_fp32_LocalSum(s2, buf);
    float mean = s1 * reciprocal_frame_size;
    float var = max(1.0e-7f, (s2 * reciprocal_frame_size) - (mean * mean));
    float rstd = rsqrt(var);

    if (id == 0) {
        running_mean_buf[node] = (running_mean_buf[node] * momentum) + (mean * (1.0 - momentum));
        running_var_buf[node]  = (running_var_buf[node]  * momentum) + (var  * (1.0 - momentum));
        mean_buf[node] = mean;
        rstd_buf[node] = rstd;
    }

    // 正規化
    float gamma = gamma_buf[node];
    float beta  = beta_buf[node];
    float* y_ptr = &y_buf[frame_stride * node];
    for ( int frame = id; frame < frame_size; frame += id_step) {
        float x = x_ptr[frame];
        x = (x - mean) * rstd;
        x = x * gamma + beta;
        y_ptr[frame] = x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_BatchNormalization_ForwardTraining
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float const     *dev_gamma_buf,
            float const     *dev_beta_buf,
            float           *dev_mean_buf,
            float           *dev_rstd_buf,
            float           *dev_running_mean_buf,
            float           *dev_running_var_buf,
            float           momentum,
            int             node_size,  
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    grid(node_size);
    dim3    block(BBCU_BATCHNORM_FW_BLOCK_SIZE);

    kernal_fp32_BatchNormalization_ForwardTraining<<<grid, block, 0, streamId>>> (
            dev_x_buf,
            dev_y_buf,
            dev_gamma_buf,
            dev_beta_buf,
            dev_mean_buf,
            dev_rstd_buf,
            dev_running_mean_buf,
            dev_running_var_buf,
            momentum,
            1.0f/ frame_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



//////////////////////////////
// ReForward
//////////////////////////////

__global__ void kernal_fp32_BatchNormalization_ReForward(
            const float     *x_buf,
            float           *y_buf,
            float const     *gamma_buf,
            float const     *beta_buf,
            float const     *mean_buf,
            float const     *rstd_buf,
            int             frame_size,
            int             frame_stride
        )
{
    // 初期化
    int const node    = blockIdx.x;
    int const id      = threadIdx.x;
    int const id_step = blockDim.x;

    float mean  = mean_buf[node];
    float rstd  = rstd_buf[node];
    float gamma = gamma_buf[node];
    float beta  = beta_buf[node];

    float const *x_ptr = &x_buf[frame_stride * node];
    float       *y_ptr = &y_buf[frame_stride * node];

    for ( int frame = id; frame < frame_size; frame += id_step) {
        float x = x_ptr[frame];
        x = (x - mean) * rstd;
        x = x * gamma + beta;
        y_ptr[frame] = x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_BatchNormalization_ReForward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float const     *dev_gamma_buf,
            float const     *dev_beta_buf,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            int             node_size,  
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    grid(node_size);
    dim3    block(BBCU_BATCHNORM_FW_BLOCK_SIZE);

    kernal_fp32_BatchNormalization_ReForward<<<grid, block, 0, streamId>>>
        (
            dev_x_buf,
            dev_y_buf,
            dev_gamma_buf,
            dev_beta_buf,
            dev_mean_buf,
            dev_rstd_buf,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



//////////////////////////////
// Forward Inference
//////////////////////////////

__global__ void kernal_fp32_BatchNormalization_ForwardInference(
            const float     *x_buf,
            float           *y_buf,
            float const     *gamma_buf,
            float const     *beta_buf,
            float const     *running_mean_buf,
            float const     *running_var_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int node    = blockDim.y * blockIdx.y + threadIdx.y;
    int id      = threadIdx.x;
    int id_step = blockDim.x;

    if ( node >= node_size) {
        return;
    }

    float gamma = gamma_buf[node];
    float beta  = beta_buf[node];
    float mean  = running_mean_buf[node];
    float var   = running_var_buf[node];

    float rstd = 1.0 / (sqrt(var) + 1.0e-7);

    float const *x_ptr = &x_buf[frame_stride * node];
    float       *y_ptr = &y_buf[frame_stride * node];
    for ( int frame = id; frame < frame_size; frame += id_step )  {
        float x = x_ptr[frame];
        y_ptr[frame] = ((x - mean) * rstd) * gamma + beta;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_BatchNormalization_ForwardInference
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float const     *dev_gamma_buf,
            float const     *dev_beta_buf,
            float const     *dev_running_mean_buf,
            float const     *dev_running_var_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 1024;
    unsigned int const MAX_FRAME_UNIT = 1024;
    unsigned int const MAX_NODE_UNIT  = 1024;

#if 1
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
    
    kernal_fp32_BatchNormalization_ForwardInference<<<grid, block, 0, streamId>>>
        (
            dev_x_buf,
            dev_y_buf,
            dev_gamma_buf,
            dev_beta_buf,
            dev_running_mean_buf,
            dev_running_var_buf,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



//////////////////////////////
// Backward
//////////////////////////////


__global__ void kernal_fp32_BatchNormalization_Backward
        (
            float const *x_buf,
            float const *dy_buf,
            float       *dx_buf,
            float const *gamma_buf,
            float       *dgamma_buf,
            float       *dbeta_buf,
            float const *mean_buf,
            float const *rstd_buf,
            float       reciprocal_frame_size,
            int         frame_size,
            int         frame_stride
        )
{
    __shared__   float  buf[BBCU_BATCHNORM_BW_BLOCK_SIZE];

    // 初期化
    int const node    = blockIdx.x;
    int const id      = threadIdx.x;
    int const id_step = blockDim.x;

    float mean = mean_buf[node];
    float rstd = rstd_buf[node];
    float gamma = gamma_buf[node];
    float dgamma = 0;
    float dbeta = 0;
    float dmeanx = 0;
    float dstd = 0;

    float dgamma_prev;
    float dbeta_prev;
    if (id == 0) {
        dgamma_prev = dgamma_buf[node];
        dbeta_prev  = dbeta_buf[node];
    }
    
    float rstd2 = rstd * rstd;

    float const * const x_ptr  = &x_buf[node * frame_stride];
    float const * const dy_ptr = &dy_buf[node * frame_stride];

    for ( int frame = id; frame < frame_size; frame += id_step) {
        float x = x_ptr[frame];
        float dy = dy_ptr[frame];
        float xc = x - mean;
        float xn = xc * rstd;
        dbeta += dy;
        dgamma += xn * dy;

        float dxn = gamma * dy;
        dstd += -(dxn * xc * rstd2);
        dmeanx += -(dxn * rstd);
    }

    dbeta = device_fp32_LocalSum(dbeta, buf);
    dgamma = device_fp32_LocalSum(dgamma, buf);
    if (id == 0) {
        dgamma_buf[node] = dgamma + dgamma_prev;
        dbeta_buf[node]  = dbeta  + dbeta_prev;
    }
    dstd   = device_fp32_LocalSum(dstd, buf);
    dmeanx = device_fp32_LocalSum(dmeanx, buf);

    float * const dx_ptr = &dx_buf[node * frame_stride];

    float dvar  = dstd * rstd;
    float dmean = (dmeanx - (mean * dvar)) * reciprocal_frame_size;

    for ( int frame = id; frame < frame_size; frame += id_step) {
        float dy = dy_ptr[frame];
        float x  = x_ptr[frame];
        float dxn = dy * gamma;
        float dxc = dxn * rstd;
        float dx  = dxc + dmean + (x * dvar * reciprocal_frame_size);
        dx_ptr[frame] = dx;
    }
}



BBCU_DLL_EXPORT int bbcu_fp32_BatchNormalization_Backward
        (
            const float     *dev_x_buf,
            const float     *dev_dy_buf,
            float           *dev_dx_buf,
            float const     *dev_gamma_buf,
            float           *dev_dgamma_buf,
            float           *dev_dbeta_buf,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            float           reciprocal_frame_size,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());


    dim3    grid(node_size);
    dim3    block(BBCU_BATCHNORM_BW_BLOCK_SIZE);

    kernal_fp32_BatchNormalization_Backward << <grid, block, 0, streamId >> > (
            dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
            dev_gamma_buf,
            dev_dgamma_buf,
            dev_dbeta_buf,
            dev_mean_buf,
            dev_rstd_buf,
            reciprocal_frame_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

