#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



__device__ __forceinline__ 
int gen_rand(int seed, int node, int frame, int node_size, int frame_size) {
    seed += frame_size * node + frame;
    return ((1103515245 * seed + 12345) & 0xffff);
}



//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_BitError_Forward
        (
            float*              x_buf,
            int                 seed,
            unsigned int        error_rate,
            int                 node_size,
            int                 frame_size,
            int                 frame_stride
        )
{
    int frame = threadIdx.x + blockIdx.x * blockDim.x;
    int node  = threadIdx.y + blockIdx.y * blockDim.y;

    if (node < node_size && frame < frame_size) {
        float v = x_buf[frame_stride * node + frame];

        int rnd = gen_rand(seed, node, frame, node_size, frame_size);

        if (rnd < error_rate) {
            v = 1.0f - v;
        }

        x_buf[frame_stride * node + frame] = v;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_BitError_Forward
        (
            float*              dev_x_buf,
            int                 seed,
            double              error_rate,
            int                 node_size,
            int                 frame_size,
            int                 frame_stride,
            cudaStream_t        streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    grid;
    dim3    block;
    block.x = std::min(frame_size, 1024);
    block.y = std::min(node_size, (int)(1024 / block.x));
    grid.x = (frame_size + block.x - 1) / block.x;
    grid.y = (node_size  + block.y - 1) / block.y;

    kernal_fp32_BitError_Forward << <grid, block, 0, streamId >> > (
        dev_x_buf,
        seed,
        (int)(error_rate * 0x10000),
        node_size,
        frame_size,
        frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


__global__ void kernal_bit_BitError_Forward
(
    int*                x_buf,
    int                 seed,
    unsigned int        error_rate,
    int                 node_size,
    int                 frame_size,
    int                 frame_stride
)
{
    int frame = threadIdx.x + blockIdx.x * blockDim.x;
    int node  = threadIdx.y + blockIdx.y * blockDim.y;
    int index = frame_stride * node + frame;
    frame *= 32;

    if (node < node_size && frame < frame_size) {
        int v = x_buf[index];
        for (int i = 0; i < 32; ++i) {
            int rnd = gen_rand(seed, node, frame + i, node_size, frame_size);
            if (rnd < error_rate) {
                v ^= (1 << i);
            }
        }
        x_buf[index] = v;
    }
}


BBCU_DLL_EXPORT int bbcu_bit_BitError_Forward
(
    int*                dev_x_buf,
    int                 seed,
    double              error_rate,
    int                 node_size,
    int                 frame_size,
    int                 frame_stride,
    cudaStream_t        streamId
)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    grid;
    dim3    block;
    block.x = std::min((frame_size + 31) / 32, 1024);
    block.y = std::min(node_size, (int)(1024 / block.x));
    grid.x = (frame_size + block.x - 1) / block.x;
    grid.y = (node_size  + block.y - 1) / block.y;

    kernal_bit_BitError_Forward << <grid, block, 0, streamId >> > (
        dev_x_buf,
        seed,
        (int)(error_rate * 0x10000),
        node_size,
        frame_size,
        frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}




//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_BitError_Backward
        (
            float*  dy_buf,
            int     seed,
            double  error_rate,
            float   weight0,
            float   weight1,
            int     node_size,
            int     frame_size,
            int     frame_stride
        )
{
    int frame = threadIdx.x + blockIdx.x * blockDim.x;
    int node  = threadIdx.y + blockIdx.y * blockDim.y;

    if (node < node_size && frame < frame_size) {
        float v = dy_buf[frame_stride * node + frame];

        int rnd = gen_rand(seed, node, frame, node_size, frame_size);
        v *= (rnd < error_rate) ? weight1 : weight0;

        dy_buf[frame_stride * node + frame] = v;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_BitError_Backward
        (
            float*              dev_dy_buf,
            int                 seed,
            double              error_rate,
            float               weight0,
            float               weight1,
            int                 node_size,
            int                 frame_size,
            int                 frame_stride,
            cudaStream_t        streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    grid;
    dim3    block;
    block.x = std::min(frame_size, 1024);
    block.y = std::min(node_size, (int)(1024 / block.x));
    grid.x = (frame_size + block.x - 1) / block.x;
    grid.y = (node_size  + block.y - 1) / block.y;

    kernal_fp32_BitError_Backward << <grid, block, 0, streamId >> > (
        dev_dy_buf,
        seed,
        (int)(error_rate * 0x10000),
        weight0,
        weight1,
        node_size,
        frame_size,
        frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// end of file
