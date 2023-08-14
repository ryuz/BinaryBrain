#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"





//////////////////////////////
// update
//////////////////////////////

__global__ void kernal_BitError_RandUpdsate
(
    unsigned int*   rand_seed
)
{
    int id = blockDim.x * threadIdx.y + threadIdx.x;

    rand_seed[id] = (48271 * rand_seed[id]) % 0x7fffffff;

}

BBCU_DLL_EXPORT int bbcu_BitError_RandUpdsate
(
    unsigned int* dev_rand_seed,
    cudaStream_t    streamId
)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    grid;
    dim3    block;
    block.x = 1024;
    kernal_BitError_RandUpdsate << <grid, block, 0, streamId >> > (
        dev_rand_seed
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_BitError_Forward
        (
            float*              x_buf,
            const unsigned int* rand_seed,
            unsigned int        error_rate,
            int                 node_size,
            int                 frame_size,
            int                 frame_stride
        )
{
    int id = blockDim.x * threadIdx.y + threadIdx.x;

    unsigned int r = rand_seed[id];
    for (int node = threadIdx.y; node < node_size; node += blockDim.y) {
        for (int frame = threadIdx.x; frame < frame_size; frame += blockDim.x) {
            for (int i = 0; i < 32; ++i) {
                auto f = frame * 32 + i;
                if (f < frame_size) {
                    r = 1103515245 * r + 12345;
                    if (r < error_rate) {
                        x_buf[frame_stride * node + f] = 1.0f - x_buf[frame_stride * node + f];
                    }
                }
            }
        }
    }
}

BBCU_DLL_EXPORT int bbcu_fp32_BitError_Forward
        (
            float*              dev_x_buf,
            const unsigned int* dev_rand_seed,
            double              error_rate,
            int                 node_size,
            int                 frame_size,
            int                 frame_stride,
            cudaStream_t        streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    int f = (frame_size + 31) / 32;

    dim3    grid;
    dim3    block;
    block.x = std::min(f, 1024);
    block.y = std::min(node_size, 1024);
    while (block.y > 1 && block.x * block.y > 1024) {
        block.y = (block.y + 1) / 2;
    }
    while (block.x > 1 && block.x * block.y > 1024) {
        block.x = (block.x + 1) / 2;
    }

    kernal_fp32_BitError_Forward << <grid, block, 0, streamId >> > (
        dev_x_buf,
        dev_rand_seed,
        (unsigned int)(error_rate * 0xffffffff),
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
    const unsigned int* rand_seed,
    unsigned int        error_rate,
    int                 node_size,
    int                 frame_size,
    int                 frame_stride
)
{
    int id = blockDim.x * threadIdx.y + threadIdx.x;

    unsigned int r = rand_seed[id];
    for (int node = threadIdx.y; node < node_size; node += blockDim.y) {
        for (int frame = threadIdx.x; frame < frame_size; frame += blockDim.x) {
            int v = x_buf[frame_stride * node + frame];
            int bit = 1;
            for (int i = 0; i < 32; ++i) {
                auto f = frame * 32 + i;
                if (f < frame_size) {
                    r = 1103515245 * r + 12345;
                    if (r < error_rate) {
                        v ^= bit;
                    }
                    bit <<= 1;
                }
            }
            x_buf[frame_stride * node + frame] = v;
        }
    }
}

BBCU_DLL_EXPORT int bbcu_bit_BitError_Forward
(
    int*            dev_x_buf,
    unsigned int*   dev_rand_seed,
    double          error_rate,
    int             node_size,
    int             frame_size,
    int             frame_stride,
    cudaStream_t    streamId
)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    int f = (frame_size + 31) / 32;

    dim3    grid;
    dim3    block;
    block.x = std::min(f, 1024);
    block.y = std::min(node_size, 1024);
    while (block.y > 1 && block.x * block.y > 1024) {
        block.y = (block.y + 1) / 2;
    }
    while (block.x > 1 && block.x * block.y > 1024) {
        block.x = (block.x + 1) / 2;
    }

    kernal_bit_BitError_Forward << <grid, block, 0, streamId >> > (
        dev_x_buf,
        dev_rand_seed,
        (unsigned int)(error_rate * 0xffffffff),
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
            float*                  dy_buf,
            const unsigned int*     rand_seed,
            double                  error_rate,
            float                   weight0,
            float                   weight1,
            int                     node_size,
            int                     frame_size,
            int                     frame_stride
        )
{
    int id = blockDim.x * threadIdx.y + threadIdx.x;

    unsigned int r = rand_seed[id];
    for (int node = threadIdx.y; node < node_size; node += blockDim.y) {
        for (int frame = threadIdx.x; frame < frame_size; frame += blockDim.x) {
            for (int i = 0; i < 32; ++i) {
                auto f = frame * 32 + i;
                if (f < frame_size) {
                    r = 1103515245 * r + 12345;
                    dy_buf[frame_stride * node + frame] *= (r < error_rate) ? weight1 : weight0;
                }
            }
        }
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_BitError_Backward
        (
            float*              dev_dy_buf,
            const unsigned int* dev_rand_seed,
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

    int f = (frame_size + 31) / 32;

    dim3    grid;
    dim3    block;
    block.x = std::min(f, 1024);
    block.y = std::min(node_size, 1024);
    while (block.y > 1 && block.x * block.y > 1024) {
        block.y = (block.y + 1) / 2;
    }
    while (block.x > 1 && block.x * block.y > 1024) {
        block.x = (block.x + 1) / 2;
    }

    kernal_fp32_BitError_Backward << <grid, block, 0, streamId >> > (
        dev_dy_buf,
        dev_rand_seed,
        (unsigned int)(error_rate * 0xffffffff),
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
