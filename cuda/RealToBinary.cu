#include <iostream>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_RealToBinary_Forward(
            const float*    x_buf,
            float*          y_buf,
            float           th_offset,
            float           th_step,
            int             modulation_size,
            int             node_size,
            int             x_frame_size,
            int             x_frame_stride,
            int             y_frame_stride
        )
{
    int x_frame = blockDim.x * blockIdx.x + threadIdx.x;
    int node    = blockDim.y * blockIdx.y + threadIdx.y;
    
    float const *x_ptr = &x_buf[node * x_frame_stride];
    float       *y_ptr = &y_buf[node * y_frame_stride];
    
    if ( x_frame < x_frame_size && node < node_size) {
        float x       = x_ptr[x_frame];
        int   y_frame = x_frame * modulation_size;
        float th      = th_offset;
        for ( int i = 0; i < modulation_size; ++i ) {
            y_ptr[y_frame + i] = (x > th) ? 1.0 : 0.0;
            th += th_step;
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_RealToBinary_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float           th_offset,
            float           th_step,
            int             modulation_size,
            int             node_size,
            int             x_frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 1024;
    unsigned int const MAX_FRAME_UNIT = 1024;
    unsigned int const MAX_NODE_UNIT  = 1024;

#if 1
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= x_frame_size )     { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size    ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= x_frame_size ) { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= node_size    ) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid((x_frame_size + (block.x - 1)) / block.x, (node_size + (block.y - 1)) / block.y);
    
    kernal_fp32_RealToBinary_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            th_offset,
            th_step,
            modulation_size,
            node_size,
            x_frame_size,
            x_frame_stride,
            y_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



// end of file
