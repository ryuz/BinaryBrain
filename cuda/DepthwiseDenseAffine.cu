#include <iostream>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"

#include "Common.cuh"


#if 0

//////////////////////////////
// forward
//////////////////////////////

template<typename T=float, int MAX_W_SIZE=1024, int MAX_X_SIZE=1024>
__global__ void kernal_DepthwiseDenseAffine_Forward
        (
            T const         *x_buf,
            T               *y_buf,
            T const         *W_buf,
            T const         *b_buf,
            unsigned int    x_point_size,
            unsigned int    y_point_size,
            unsigned int    frame_loop,
            unsigned int    frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride
        )
{
    unsigned int const  frame_id   = threadIdx.x;
    unsigned int const  frame_step = blockDim.x;
    unsigned int const  node_id    = threadIdx.y;
    unsigned int const  node_step  = blockDim.y;
    unsigned int const  depth      = blockIdx.x;
    unsigned int const  y_point    = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int const  id         = blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int const  id_step    = blockDim.x * blockDim.y;
    
    __shared__  T   W[MAX_W_SIZE];   // [y_point_size][x_point_size]
    __shared__  T   x[MAX_X_SIZE];   // [x_point_size][frame_step]
    
    // read W
    T const *W_ptr = &W_buf[depth * y_point_size * x_point_size];
    for ( unsigned int i = id; i < x_point_size*y_point_size; i += id_step ) {
        W[i] = W_ptr[i];
    }
    __syncthreads();

    // read b
    T  b = b_buf[depth * y_point_size + y_point];

    T const *x_ptr = &x_buf[(depth * x_point_size + 0      ) * x_frame_stride];
    T       *y_ptr = &y_buf[(depth * y_point_size + y_point) * y_frame_stride];
    for ( unsigned int f = 0; f < frame_loop; ++f ) {
        unsigned int frame = f * frame_step + frame_id;

        // read x
        for ( unsigned int x_point = node_id; x_point < x_point_size; x_point += node_step ) {
            x[x_point * frame_step + frame_id] = x_ptr[x_point * x_frame_stride + frame];
        }
        __syncthreads();

        // calc
        if ( frame < frame_size && y_point < y_point_size ) {
            T y = b;
            for ( unsigned int x_point = 0; x_point < x_point_size; ++x_point ) {
                y += W[y_point * x_point_size + x_point] * x[x_point * frame_step + frame_id];
            }
            y_ptr[frame] = y;
        }
        __syncthreads();
    }
}



template<typename T>
BBCU_DLL_EXPORT int bbcu_DepthwiseDenseAffine_Forward
        (
            T const         *dev_x_buf,
            T               *dev_y_buf,
            T const         *dev_W_buf,
            T const         *dev_b_buf,
            unsigned int    depth_size,
            unsigned int    x_point_size,
            unsigned int    y_point_size,
            unsigned int    frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const MIN_FRAME_UNIT = 1;

    dim3    block(1024, 1);
    while ( block.x / 2 >= frame_size && block.x > MIN_FRAME_UNIT ){ block.x /= 2; block.y *= 2; }
    block.x = std::min(block.x, frame_size);
    block.y = std::min(block.y, y_point_size);
    dim3    grid;
    grid.x = (frame_size   + (block.x - 1)) / block.x;
    grid.y = (y_point_size + (block.y - 1)) / block.y;
    grid.z = depth_size;
    
    unsigned int    frame_loop = (frame_size + (block.x - 1)) / block.x;

    kernal_DepthwiseDenseAffine_Forward<T><<<grid, block, 0, streamId>>>
        (
              dev_x_buf,
              dev_y_buf,
              dev_W_buf,
              dev_b_buf,
              x_point_size,
              y_point_size,
              frame_loop,
              frame_size,
              x_frame_stride,
              y_frame_stride
        );
    
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



template BBCU_DLL_EXPORT int bbcu_DepthwiseDenseAffine_Forward<float>(float const *, float *, float const *, float const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, cudaStream_t);


#endif


// end of file
