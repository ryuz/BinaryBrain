#include <iostream>
#include <chrono>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



// -------------------------------------------------
//  Forward
// -------------------------------------------------


template <typename T>
__global__ void kernal_AverageLut_Forward(
            T   const   *x_buf,
            T           *y_buf,
            int const   *input_index,
            int         n,
            T           reciprocal_n,
            int         node_size,
            int         frame_size,
            int         frame_stride,
            bool        binarize_input,
            bool        binarize_output
        )
{
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    int node  = blockIdx.y * blockDim.y + threadIdx.y;

    if ( node < node_size && frame < frame_size ) {
        T   sum = 0;
        for ( int i = 0; i < n; ++i ) {
            int input_node = input_index[node*n + i];
            T   x = x_buf[input_node*frame_stride + frame];
            if (binarize_input) {
                x = (x > 0) ? (T)BB_BINARY_HI : (T)BB_BINARY_LO;
            }
            sum += x;
        }

        T   average;
        if (binarize_output) {
            average = (sum > 0) ? (T)BB_BINARY_HI : (T)BB_BINARY_LO;
        }
        else {
            average = sum * reciprocal_n;
        }

        y_buf[node*frame_stride + frame] = average;
    }
}


template <typename T>
BBCU_DLL_EXPORT int bbcu_AverageLut_Forward
        (
            T   const       *dev_x_buf,
            T               *dev_y_buf,
            int const       *dev_input_index,
            int             n,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            bool            binarize_input,
            bool            binarize_output,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(1024, 1);
    while (block.x / 2 >= (unsigned int)frame_size) {
        block.x /= 2;
        block.y *= 2;
    }
    block.x = std::min(block.x, (unsigned int)frame_size);
    block.y = std::min(block.y, (unsigned int)node_size);

    dim3    grid;
    grid.x = (frame_size + block.x - 1) / block.x;
    grid.y = (node_size  + block.y - 1) / block.y;
    
    kernal_AverageLut_Forward<T><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            n,
            (T)1.0 / (T)n,
            node_size,
            frame_size,
            frame_stride,
            binarize_input,
            binarize_output
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}

template BBCU_DLL_EXPORT int bbcu_AverageLut_Forward<float >(float  const *, float  *, int const *, int, int, int, int, bool, bool, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_AverageLut_Forward<double>(double const *, double *, int const *, int, int, int, int, bool, bool, cudaStream_t);



__global__ void kernal_bit_AverageLut_Forward(
            int const   *x_buf,
            int         *y_buf,
            int const   *input_index,
            int         n,
            int         node_size,
            int         frame_size,
            int         frame_stride
        )
{
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    int node  = blockIdx.y * blockDim.y + threadIdx.y;

    if ( node < node_size && frame < frame_size ) {
        int bitmask = 1;
        int y = 0;
        for ( int bitpos = 0; bitpos < 32; ++bitpos ) {
            int count = 0;
            for ( int i = 0; i < n; ++i ) {
                int input_node = input_index[node*n + i];
                int x = x_buf[input_node*frame_stride + frame];
                count += (x & bitmask) ? +1 : -1;
            }
            y |= (count > 0) ? bitmask : 0;
            bitmask <<= 1;
        }
        y_buf[node*frame_stride + frame] = y;
    }
}

BBCU_DLL_EXPORT int bbcu_bit_AverageLut_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int const       *dev_input_index,
            int             n,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    frame_size = (frame_size + 31) / 32;

    dim3    block(1024, 1);
    while (block.x / 2 >= (unsigned int)frame_size) {
        block.x /= 2;
        block.y *= 2;
    }
    block.x = std::min(block.x, (unsigned int)frame_size);
    block.y = std::min(block.y, (unsigned int)node_size);

    dim3    grid;
    grid.x = (frame_size + block.x - 1) / block.x;
    grid.y = (node_size  + block.y - 1) / block.y;
    
    kernal_bit_AverageLut_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            n,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}



// -------------------------------------------------
//  Backward
// -------------------------------------------------

template <typename T>
__global__ void kernal_AverageLut_Backward(
            T   const   *dy_buf,
            T           *dx_buf,
            int const   *input_index,
            int         n,
            T           reciprocal_n,
            int         node_size,
            int         frame_size,
            int         frame_stride
        )
{
    int frame = blockIdx.x * blockDim.x + threadIdx.x;

    if ( frame < frame_size ) {
        for ( int node = 0; node < node_size; ++node ) {
            T dy = dy_buf[node*frame_stride + frame] * reciprocal_n;
            for ( int i = 0; i < n; ++i ) {
                int input_node = input_index[node*n + i];
                dx_buf[input_node*frame_stride + frame] += dy;
            }
        }
    }
}


template <typename T>
BBCU_DLL_EXPORT int bbcu_AverageLut_Backward
        (
            T   const       *dev_dy_buf,
            T               *dev_dx_buf,
            int const       *dev_input_index,
            int             n,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    cudaMemset(dev_dx_buf, 0, sizeof(T)*frame_stride*input_node_size);

    dim3    block(std::min(frame_size, 1024));
    dim3    grid((frame_size + 1023) / 1024);
    
    kernal_AverageLut_Backward<T><<<grid, block, 0, streamId>>>(
            dev_dy_buf,
            dev_dx_buf,
            dev_input_index,
            n,
            (T)1/ (T)n,
            output_node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}

template BBCU_DLL_EXPORT int bbcu_AverageLut_Backward<float >(float  const *, float  *, int const *, int, int, int, int, int, cudaStream_t);
template BBCU_DLL_EXPORT int bbcu_AverageLut_Backward<double>(double const *, double *, int const *, int, int, int, int, int, cudaStream_t);



// end of file
