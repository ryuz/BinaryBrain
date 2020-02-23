#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



template<typename T=float>
__global__ void kernal_ConvBitToReal(
            int   const     *x_buf,
            T               *y_buf,
            T               value0,
            T               value1,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             y_frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;
    int node  = blockDim.y * blockIdx.y + threadIdx.y;

    int bit      = (threadIdx.x & 0x1f);
    int bit_mask = (1 << bit);
    int unit     = (frame >> 5);

    if ( frame < frame_size && node < node_size ) {
        int x = x_buf[node * x_frame_stride + unit];
        T   y = (x & bit_mask) ? value1 : value0;
        y_buf[node * y_frame_stride + frame] = y;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_ConvBitToReal
        (
            int   const     *dev_x_buf,
            T               *dev_y_buf,
            T               value0,
            T               value1,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(32, 32);
    dim3    grid((frame_size + 31) / 32, (node_size + 31) / 32);
    
    kernal_ConvBitToReal<T><<<grid, block, 0, streamId>>>
        (
            dev_x_buf,
            dev_y_buf,
            value0,
            value1,
            node_size,
            frame_size,
            x_frame_stride,
            y_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


template BBCU_DLL_EXPORT int bbcu_ConvBitToReal<float>
        (
            int   const     *dev_x_buf,
            float           *dev_y_buf,
            float           value0,
            float           value1,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            cudaStream_t    streamId
        );


// end of file
