#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



__global__ void kernal_fp32_Adam(
            int     const   *size_table,
            float * const   *params_buf_table,
            float * const   *grads_buf_table,
            float * const   *m_buf_table,
            float * const   *v_buf_table,
            float           lr_t,
            float           neg_beta1,
            float           neg_beta2
        )
{
    int id      = threadIdx.x;
    int id_step = blockDim.x;
    int index   = blockDim.y * blockIdx.y + threadIdx.y;
    
    int   size       = size_table[index];

    float *params_buf = params_buf_table[index];
    float *grads_buf  = grads_buf_table[index];
    float *m_buf      = m_buf_table[index];
    float *v_buf      = v_buf_table[index];
    
    for ( int n = id; n < size; n += id_step ) {
        float param = params_buf[n];
        float grad  = grads_buf[n];
        float m     = m_buf[n];
        float v     = v_buf[n];

        m     += neg_beta1 * (grad - m);
        v     += neg_beta2 * (grad * grad - v);
        param -= lr_t * m / (sqrt(v) + 1e-7);

        m_buf[n]      = m;
        v_buf[n]      = v;
        params_buf[n] = param;
        grads_buf[n]  = 0;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Adam
        (
            int             size,
            int     const   *dev_size_table,
            float * const   *dev_params_buf_table,
            float * const   *dev_grads_buf_table,
            float * const   *dev_m_buf_table,
            float * const   *dev_v_buf_table,
            float           lr_t,
            float           beta1,
            float           beta2,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    grid(1, size);
    dim3    block(192, 1);
    
    kernal_fp32_Adam<<<grid, block, 0, streamId>>>(
            dev_size_table,
            dev_params_buf_table,
            dev_grads_buf_table,
            dev_m_buf_table,
            dev_v_buf_table,
            lr_t,
            (1.0f - beta1),
            (1.0f - beta2)
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}

// end of file
