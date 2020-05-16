#include <iostream>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"

#include "Common.cuh"



//////////////////////////////
// forward
//////////////////////////////


template<typename T>
__global__ void kernal_bit_BitEncode(
            T const         *x_buf,
            int             *y_buf,
            unsigned int    bit_size,
            T               clip_min,
            T               clip_max,
            T               scale,
            T               offset,
            unsigned int    node_size,
            unsigned int    frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride
        )
{
    unsigned int    frame_unit = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int    node       = blockDim.y * blockIdx.y + threadIdx.y;

    T const *x_ptr = &x_buf[(x_frame_stride * node)];
    int     *y_ptr = &y_buf[(y_frame_stride * node)];

    unsigned int y[32];
    for ( int bit = 0; bit < bit_size; ++bit ) {
        y[bit] = 0;
    }

    unsigned int y_mask = 1;
    for ( int i = 0; i < 32; ++i ) {
        unsigned int frame = frame_unit * 32 + i;
        if ( frame < frame_size ) {
            int x = (int)(min(clip_max, max(clip_min, x_ptr[frame])) * scale + offset);
            int x_mask = 1;
            for ( int bit = 0; bit < bit_size; ++bit ) {
                if ( (x & x_mask) != 0 ) {
                    y[bit] |= y_mask;
                }
                x_mask <<= 1;
            }
        }
        y_mask <<= 1;
    }

    for ( int bit = 0; bit < bit_size; ++bit ) {
        y_ptr[y_frame_stride * node_size * bit + frame_unit] = y[bit];
    }
}



template<typename T>
BBCU_DLL_EXPORT int bbcu_bit_BitEncode
        (
            T const         *dev_x_buf,
            int             *dev_y_buf,
            unsigned int    bit_size,
            T               clip_min,
            T               clip_max,
            T               scale,
            T               offset,
            unsigned int    node_size,
            unsigned int    frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const MIN_FRAME_UNIT = 1;

    unsigned int    frame_unit_size = (frame_size + 31) / 32;

    dim3    block(1024, 1);
    while ( block.x / 2 >= frame_unit_size && block.x > MIN_FRAME_UNIT ){ block.x /= 2; block.y *= 2; }

    block.x = std::min(block.x, frame_unit_size);
    block.y = std::min(block.y, node_size);
    dim3    grid;
    grid.x = (frame_unit_size + (block.x - 1)) / block.x;
    grid.y = (node_size       + (block.y - 1)) / block.y;
    
    kernal_bit_BitEncode<T><<<grid, block, 0, streamId>>>
                (
                    dev_x_buf,
                    dev_y_buf,
                    bit_size,
                    clip_min,
                    clip_max,
                    scale,
                    offset,
                    node_size,
                    frame_size,
                    x_frame_stride,
                    y_frame_stride
                );
    
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



template BBCU_DLL_EXPORT int bbcu_bit_BitEncode<float>(
            float const     *dev_x_buf,
            int             *dev_y_buf,
            unsigned int    bit_size,
            float           clip_min,
            float           clip_max,
            float           scale,
            float           offset,
            unsigned int    node_size,
            unsigned int    frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            cudaStream_t    streamId
        );


// end of file
