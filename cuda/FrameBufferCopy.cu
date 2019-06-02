#include <iostream>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



template <typename T>
__global__ void kernal_FrameBufferCopy
        (
            T           *dst_buf,
            T   const   *src_buf,
            int         node_size,
            int         dst_node_offset,
            int         src_node_offset,
            int         frame_size,
            int         dst_frame_offset,
            int         src_frame_offset,
            int         dst_frame_stride,
            int         src_frame_stride
        )
{
    int const node    = blockIdx.y * blockDim.y + threadIdx.y;
    int const id      = threadIdx.x;
    int const id_step = blockDim.x;
    
    T       *dst_ptr = &dst_buf[(node + dst_node_offset) * dst_frame_stride + dst_frame_offset];
    T const *src_ptr = &src_buf[(node + src_node_offset)   * src_frame_stride + src_frame_offset];

    if ( node < node_size ) {
        for ( int frame = id; frame < frame_size; frame += id_step ) {
            dst_ptr[frame] = src_ptr[frame];
        }
    }
}


BBCU_DLL_EXPORT int bbcu_int32_FrameBufferCopy
        (
            int             *dev_dst_buf,
            int const       *dev_src_buf,
            int             node_size,
            int             dst_node_offset,
            int             src_node_offset,
            int             frame_size,
            int             dst_frame_offset,
            int             src_frame_offset,
            int             dst_frame_stride,
            int             src_frame_stride,
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
    while ( (int)block.y / 2 >= node_size )  { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size)  { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (node_size + (block.y - 1)) / block.y);

    kernal_FrameBufferCopy<int><<<grid, block, 0, streamId>>>(
            dev_dst_buf,
            dev_src_buf,
            node_size,
            dst_node_offset,
            src_node_offset,
            frame_size,
            dst_frame_offset,
            src_frame_offset,
            dst_frame_stride,
            src_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// end of file
