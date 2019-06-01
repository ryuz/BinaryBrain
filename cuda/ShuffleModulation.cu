#include <iostream>
#include <algorithm>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_bit_ShuffleModulation_Forward(
            int const       *x_buf,
            int             *y_buf,
            int const       *table,
            int             shuffle_size,
            int             lowering_size,
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int id_step = blockDim.x;
    int id      = threadIdx.x;
    int node    = blockDim.y * blockIdx.y + threadIdx.y;

    int const *x_ptr     = &x_buf[frame_stride * node];
    int       *y_ptr     = &y_buf[frame_stride * node];
    int const *table_ptr = &table[shuffle_size * node];

    int f_size = frame_size / 32;

    for ( int f = id; f < f_size; f += id_step ) {
        int y = 0;
        for ( int bit = 0; bit < 32; ++bit ) {
            int frame = f*32 + bit;
            if ( frame < frame_size ) {
                int i = frame / (lowering_size * shuffle_size);
                int j = frame / lowering_size % shuffle_size;
                int k = frame % lowering_size;

                int input_frame = i * (lowering_size * shuffle_size) + table_ptr[j] * lowering_size + k;
                int x = ((x_ptr[input_frame / 32] >> (input_frame % 32)) & 1);
                y |= (x << bit);
            }
        }
        y_ptr[f] = y;
    }
}


BBCU_DLL_EXPORT int bbcu_bit_ShuffleModulation_Forward
        (
            int const     *dev_x_buf,
            int           *dev_y_buf,
            int const     *dev_table,
            int           shuffle_size,
            int           lowering_size,
            int           node_size,
            int           frame_size,
            int           frame_stride,
            cudaStream_t  streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    int f_size = frame_size / 32;

    dim3    block(1024, 1);
    while ( (int)block.x / 2 >= f_size    ) { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= node_size ) { block.y /= 2; }

    dim3    grid;
    grid.x = (f_size    + (block.x - 1)) / block.x;
    grid.y = (node_size + (block.y - 1)) / block.y;

    kernal_bit_ShuffleModulation_Forward<<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_table,
            shuffle_size,
            lowering_size,
            node_size,
            frame_size,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// end of file
