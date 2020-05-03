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


#if 0

template<typename T=float>
__global__ void kernal_RealToBinary_Forward(
            T const         *x_buf,
            T               *y_buf,
            unsigned int    depth_modulation_size,
            T               depth_modulation_step,
            unsigned int    frame_modulation_size,
            T               frame_modulation_step,
            T               x_offset,
            T               x_scale,
            unsigned int    point_size,
            unsigned int    x_depth_size,
            unsigned int    x_frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            bool            binarize
        )
{
    unsigned int    y_frame = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int    x_depth = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int    point   = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int    y_depth = x_depth * depth_modulation_size;
    unsigned int    x_frame = y_frame / frame_modulation_size;
    unsigned int    frame   = y_frame % frame_modulation_size;

    T const *x_ptr = &x_buf[(x_depth * point_size + point) * x_frame_stride + x_frame];
    T       *y_ptr = &y_buf[(y_depth * point_size + point) * y_frame_stride + y_frame];

    if ( x_frame < x_frame_size && point < point_size && x_depth < x_depth_size ) {
        T   x = (*x_ptr - x_offset) * x_scale;
        T   depth_step_recip = (T)depth_modulation_size;  // reciprocal of depth_modulation_step
        T   frame_step_recip = (T)frame_modulation_size;  // reciprocal of frame_modulation_step

        for ( int depth = 0; depth < depth_modulation_size; ++depth ) {
            T   y = x;

            // modulation for depth
            y = (y - (T)(depth * depth_modulation_step)) * depth_step_recip;
            
            // modulation for frame
            y = (y - (T)(frame * frame_modulation_step)) * frame_step_recip;

            // clamp
            y = max((T)0.0, min((T)1.0, y));

            if ( binarize ) {
                y = (y > (T)0.5) ? (T)1.0 : (T)0.0;
            }
            y_ptr[depth * point_size * y_frame_stride] = y;
        }
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_RealToBinary_Forward
        (
            T   const       *dev_x_buf,
            T               *dev_y_buf,
            unsigned int    depth_modulation_size,
            unsigned int    frame_modulation_size,
            T               input_range_lo,
            T               input_range_hi,
            unsigned int    point_size,
            unsigned int    x_depth_size,
            unsigned int    x_frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            bool            binarize,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const MIN_FRAME_UNIT = 1;
    unsigned int const MIN_DEPTH_UNIT = 1;

    unsigned int    y_frame_size = x_frame_size * frame_modulation_size;
    T               depth_modulation_step = (T)1.0 / (T)depth_modulation_size;
    T               frame_modulation_step = (T)1.0 / (T)frame_modulation_size;

    dim3    block(32, 32, 1);
    while ( block.x / 2 >= y_frame_size && block.x > MIN_FRAME_UNIT ){ block.x /= 2; block.y *= 2; }
    while ( block.y / 2 >= x_depth_size && block.y > MIN_DEPTH_UNIT ){ block.y /= 2; block.z *= 2; }
    block.x = std::min(block.x, y_frame_size);
    block.y = std::min(block.y, x_depth_size);
    block.z = std::min(block.z, point_size);
    dim3    grid;
    grid.x = (y_frame_size + (block.x - 1)) / block.x;
    grid.y = (x_depth_size + (block.y - 1)) / block.y;
    grid.z = (point_size   + (block.z - 1)) / block.z;
    
    kernal_RealToBinary_Forward<T><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            depth_modulation_size,
            depth_modulation_step,
            frame_modulation_size,
            frame_modulation_step,
            input_range_lo,
            (T)1.0 / (input_range_hi - input_range_lo),
            point_size,
            x_depth_size,
            x_frame_size,
            x_frame_stride,
            y_frame_stride,
            binarize
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


#else


template<typename T=float>
__global__ void kernal_RealToBinary_Forward(
            T const         *x_buf,
            T               *y_buf,
            unsigned int    depth_modulation_size,
            T               depth_modulation_step,
            unsigned int    frame_modulation_size,
            T               frame_modulation_step,
            T               x_offset,
            T               x_scale,
            int             point_size,
            int             x_depth_size,
            int             x_frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            bool            binarize
        )
{
    unsigned int    y_frame = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int    y_depth = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int    point   = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int    x_depth = y_depth / depth_modulation_size;
    unsigned int    depth   = y_depth % depth_modulation_size;
    unsigned int    x_frame = y_frame / frame_modulation_size;
    unsigned int    frame   = y_frame % frame_modulation_size;

    T const *x_ptr = &x_buf[(x_depth * point_size + point) * x_frame_stride + x_frame];
    T       *y_ptr = &y_buf[(y_depth * point_size + point) * y_frame_stride + y_frame];

    if ( x_frame < x_frame_size && point < point_size && x_depth < x_depth_size ) {
        T   x = (*x_ptr - x_offset) * x_scale;

        T   depth_step_recip = (T)depth_modulation_size;  // reciprocal of depth_modulation_step
        T   frame_step_recip = (T)frame_modulation_size;  // reciprocal of frame_modulation_step
        T   y = x;

        // modulation for depth
        y = (y - (T)(depth * depth_modulation_step)) * depth_step_recip;

        // modulation for frame
        y = (y - (T)(frame * frame_modulation_step)) * frame_step_recip;

        // clamp
        y = max((T)0.0, min((T)1.0, y));
        
        // modulation for frame
        if ( binarize ) {
            y = (y > (T)0.5) ? (T)1.0 : (T)0.0;
        }

        *y_ptr = y;
    }
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_RealToBinary_Forward
        (
            T   const       *dev_x_buf,
            T               *dev_y_buf,
            unsigned int    depth_modulation_size,
            unsigned int    frame_modulation_size,
            T               input_range_lo,
            T               input_range_hi,
            unsigned int    point_size,
            unsigned int    x_depth_size,
            unsigned int    x_frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            bool            binarize,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const MIN_FRAME_UNIT = 1;
    unsigned int const MIN_DEPTH_UNIT = 1;

    unsigned int    y_frame_size = x_frame_size * frame_modulation_size;
    unsigned int    y_depth_size = x_depth_size * depth_modulation_size;
    T               depth_modulation_step = (T)1.0 / (T)depth_modulation_size;
    T               frame_modulation_step = (T)1.0 / (T)frame_modulation_size;

    dim3    block(32, 32, 1);
    while ( block.x / 2 >= y_frame_size && block.x > MIN_FRAME_UNIT ){ block.x /= 2; block.y *= 2; }
    while ( block.y / 2 >= y_depth_size && block.y > MIN_DEPTH_UNIT ){ block.y /= 2; block.z *= 2; }
    block.x = std::min(block.x, y_frame_size);
    block.y = std::min(block.y, y_depth_size);
    block.z = std::min(block.z, point_size);
    dim3    grid;
    grid.x = (y_frame_size + (block.x - 1)) / block.x;
    grid.y = (y_depth_size + (block.y - 1)) / block.y;
    grid.z = (point_size   + (block.z - 1)) / block.z;
    
    kernal_RealToBinary_Forward<T><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            depth_modulation_size,
            depth_modulation_step,
            frame_modulation_size,
            frame_modulation_step,
            input_range_lo,
            (T)1.0 / (input_range_hi - input_range_lo),
            point_size,
            x_depth_size,
            x_frame_size,
            x_frame_stride,
            y_frame_stride,
            binarize
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}
#endif


template BBCU_DLL_EXPORT int bbcu_RealToBinary_Forward<float>(float const *, float *, unsigned int, unsigned int, float, float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, cudaStream_t);




template<typename T=float>
__global__ void kernal_bit_RealToBinary_Forward(
            T const         *x_buf,
            int             *y_buf,
            unsigned int    depth_modulation_size,
            T               depth_modulation_step,
            unsigned int    frame_modulation_size,
            T               frame_modulation_step,
            T               x_offset,
            T               x_scale,
            int             point_size,
            int             x_depth_size,
            int             x_frame_size,
            int             x_frame_stride,
            int             y_frame_stride
        )
{
    unsigned int    y_unit  = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int    y_depth = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int    point   = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int    x_depth = y_depth / depth_modulation_size;
    unsigned int    depth   = y_depth % depth_modulation_size;

    T const *x_ptr = &x_buf[(x_depth * point_size + point) * x_frame_stride];

    int y_bit    = 0;
    int bit_mask = 1;
    for ( int bit = 0; bit < 32; ++bit ) {
        unsigned int    y_frame  = y_unit * 32 + bit;
        unsigned int    x_frame  = y_frame / frame_modulation_size;
        unsigned int    frame    = y_frame % frame_modulation_size;

        if ( x_frame < x_frame_size && point < point_size && x_depth < x_depth_size ) {
            T   x = (x_ptr[x_frame] - x_offset) * x_scale;

            T   depth_step_recip = (T)depth_modulation_size;  // reciprocal of depth_modulation_step
            T   frame_step_recip = (T)frame_modulation_size;  // reciprocal of frame_modulation_step
            T   y = x;

            // modulation for depth
            y = (y - (T)(depth * depth_modulation_step)) * depth_step_recip;

            // modulation for frame
            y = (y - (T)(frame * frame_modulation_step)) * frame_step_recip;

            // clamp
            y = max((T)0.0, min((T)1.0, y));
        
            // modulation for frame
            if (y > (T)0.5) {
                y_bit |= bit_mask;
            }

            bit_mask <<= 1;
        }
    }

    int   *y_ptr = &y_buf[(y_depth * point_size + point) * y_frame_stride];
    y_ptr[y_unit] = y_bit;
}


template<typename T>
BBCU_DLL_EXPORT int bbcu_bit_RealToBinary_Forward
        (
            T   const       *dev_x_buf,
            int             *dev_y_buf,
            unsigned int    depth_modulation_size,
            unsigned int    frame_modulation_size,
            T               input_range_lo,
            T               input_range_hi,
            unsigned int    point_size,
            unsigned int    x_depth_size,
            unsigned int    x_frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const MIN_FRAME_UNIT = 1;
    unsigned int const MIN_DEPTH_UNIT = 1;

    unsigned int    y_frame_size = x_frame_size * frame_modulation_size;
    unsigned int    y_depth_size = x_depth_size * depth_modulation_size;
    T               depth_modulation_step = (T)1.0 / (T)depth_modulation_size;
    T               frame_modulation_step = (T)1.0 / (T)frame_modulation_size;
    unsigned int    y_unit_size = (y_frame_size + 31) / 32;
    
//  dim3    block(32, 32, 1);
    dim3    block(32, 16, 1);  // 1024スレッド作るにはレジスタが足りない模様
    while ( block.x / 2 >= y_unit_size  && block.x > MIN_FRAME_UNIT ){ block.x /= 2; block.y *= 2; }
    while ( block.y / 2 >= y_depth_size && block.y > MIN_DEPTH_UNIT ){ block.y /= 2; block.z *= 2; }
    block.x = std::min(block.x, y_unit_size);
    block.y = std::min(block.y, y_depth_size);
    block.z = std::min(block.z, point_size);
    block.x = std::min(block.x, 1024U);
    block.y = std::min(block.y, 1024U);
    block.z = std::min(block.z, 64U);

    dim3    grid;
    grid.x = (y_unit_size  + (block.x - 1)) / block.x;
    grid.y = (y_depth_size + (block.y - 1)) / block.y;
    grid.z = (point_size   + (block.z - 1)) / block.z;
    
    kernal_bit_RealToBinary_Forward<T><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            depth_modulation_size,
            depth_modulation_step,
            frame_modulation_size,
            frame_modulation_step,
            input_range_lo,
            (T)1.0 / (input_range_hi - input_range_lo),
            point_size,
            x_depth_size,
            x_frame_size,
            x_frame_stride,
            y_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


template BBCU_DLL_EXPORT int bbcu_bit_RealToBinary_Forward<float>(float const *, int *, unsigned int, unsigned int, float, float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, cudaStream_t);


#if 0




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



//////////////////


template <int MAX_NODE_UNIT>
__global__ void kernal_fp32_bit_no_modulation_RealToBinary_Forward
        (
            float const     *x_buf,
            int             *y_buf,
            float           th,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             y_frame_stride
        )
{
    int frame   = blockDim.x * blockIdx.x + threadIdx.x;
    int node    = blockDim.y * blockIdx.y + threadIdx.y;
    int unit_id = ((threadIdx.y * blockDim.x + threadIdx.x) >> 5);
    
    __shared__ int     sbuf[MAX_NODE_UNIT][32];

    float const *x_ptr = &x_buf[node * x_frame_stride];
    int         *y_ptr = &y_buf[node * y_frame_stride];
    
    int bit  = (frame & 0x1f);
    int unit = (frame >> 5);

    int y = 0;
    if ( frame < frame_size && node < node_size) {
        float x = x_ptr[frame];
        y = (x > th) ? (1 << bit) : 0;
    }

    y = device_int_LocalOr(y, bit, sbuf[unit_id]);

    if ( frame < frame_size && node < node_size && bit == 0 ) {
        y_ptr[unit] = y;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_bit_no_modulation_RealToBinary_Forward
        (
            float const     *dev_x_buf,
            int             *dev_y_buf,
            float           th,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 1024;
    unsigned int const MAX_FRAME_UNIT = 1024;
    unsigned int const MIN_FRAME_UNIT = 32;
    unsigned int const MAX_NODE_UNIT  = 32;

#if 1
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size && block.x > MIN_FRAME_UNIT ){ block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size                              ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= node_size                              ) { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size && block.x > MIN_FRAME_UNIT ) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid((frame_size + (block.x - 1)) / block.x, (node_size + (block.y - 1)) / block.y);
    
    kernal_fp32_bit_no_modulation_RealToBinary_Forward<MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            th,
            node_size,
            frame_size,
            x_frame_stride,
            y_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


#endif


// end of file
