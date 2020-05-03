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


template<typename T=int>
__global__ void kernal_Shuffle_Copy(
            T const         *x_buf,
            T               *y_buf,
            unsigned int    y_unit_size,
            unsigned int    y_group_size,
            unsigned int    node_size,
//          unsigned int    frame_size,
            unsigned int    frame_stride
        )
{
    unsigned int    frame  = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int    y_node = blockDim.y * blockIdx.y + threadIdx.y;

    if ( frame < frame_stride && y_node < node_size ) {
        unsigned int    unit  = y_node % y_unit_size;
        unsigned int    group = y_node / y_unit_size; 
        unsigned int    x_node = y_group_size * unit + group;

        T const *x_ptr = &x_buf[x_node * frame_stride + frame];
        T       *y_ptr = &y_buf[y_node * frame_stride + frame];

        *y_ptr = *x_ptr;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_Shuffle_Forward
        (
            T const         *dev_x_buf,
            T               *dev_y_buf,
            unsigned int    y_unit_size,
            unsigned int    node_size,
            unsigned int    frame_size,
            unsigned int    frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const MIN_FRAME_UNIT = 1;

    dim3    block(1024, 1);
    while ( block.x / 2 >= frame_size && block.x > MIN_FRAME_UNIT ){ block.x /= 2; block.y *= 2; }
    block.x = std::min(block.x, frame_size);
    block.y = std::min(block.y, node_size);
    dim3    grid;
    grid.x = (frame_size + (block.x - 1)) / block.x;
    grid.y = (node_size  + (block.y - 1)) / block.y;
    
    kernal_Shuffle_Copy<T><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            y_unit_size,
            node_size / y_unit_size,
            node_size,
//          frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_Shuffle_Backward
        (
            T const         *dev_dy_buf,
            T               *dev_dx_buf,
            unsigned int    y_unit_size,
            unsigned int    node_size,
            unsigned int    frame_size,
            unsigned int    frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const MIN_FRAME_UNIT = 1;

    dim3    block(1024, 1);
    while ( block.x / 2 >= frame_size && block.x > MIN_FRAME_UNIT ){ block.x /= 2; block.y *= 2; }
    block.x = std::min(block.x, frame_size);
    block.y = std::min(block.y, node_size);
    dim3    grid;
    grid.x = (frame_size + (block.x - 1)) / block.x;
    grid.y = (node_size  + (block.y - 1)) / block.y;
    
    kernal_Shuffle_Copy<T><<<grid, block, 0, streamId>>>(
            dev_dy_buf,
            dev_dx_buf,
            node_size / y_unit_size,
            y_unit_size,
            node_size,
//          frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



template BBCU_DLL_EXPORT int bbcu_Shuffle_Forward<float>(float const *, float *, unsigned int, unsigned int, unsigned int, unsigned int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_Shuffle_Forward<int>  (int   const *, int   *, unsigned int, unsigned int, unsigned int, unsigned int, cudaStream_t);

template BBCU_DLL_EXPORT int bbcu_Shuffle_Backward<float>(float const *, float *, unsigned int, unsigned int, unsigned int, unsigned int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_Shuffle_Backward<int>  (int   const *, int   *, unsigned int, unsigned int, unsigned int, unsigned int, cudaStream_t);


// end of file
