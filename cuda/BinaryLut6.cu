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

template<int LUT, int VAL>
__forceinline__ __device__ int device_lut_mask_unit(int &val, int &lut)
{
    if ((LUT & (1 << VAL)) == 0) {
        return ((~val) & lut);
    }
    else {
        return (val & lut);
    }
}

template<int LUT>
__forceinline__ __device__  void device_lut6_mask(int &msk, int lut, int val[6])
{
    lut = device_lut_mask_unit<LUT, 0>(val[0], lut);
    lut = device_lut_mask_unit<LUT, 1>(val[1], lut);
    lut = device_lut_mask_unit<LUT, 2>(val[2], lut);
    lut = device_lut_mask_unit<LUT, 3>(val[3], lut);
    lut = device_lut_mask_unit<LUT, 4>(val[4], lut);
    lut = device_lut_mask_unit<LUT, 5>(val[5], lut);
    msk = (msk | lut);
}

template <int BUF_SIZE>
__global__ void kernal_bit_BinaryLut6_Forward(
            int const   *x_buf,
            int         *y_buf,
            int const   *input_index,
            int const   *lut_table,
            int         node_size,
            int         frame_size,
            int         frame_stride
        )
{
    int id       = threadIdx.x;
    int id_step  = blockDim.x;

    int node_idx  = threadIdx.y;
    int node_step = blockDim.y;
    int node      = blockIdx.y * blockDim.y + threadIdx.y;
    
     __shared__ int buf[BUF_SIZE];
    int *table  = &buf[0];
    int *in_idx = &table[64 * node_step];

    if ( node < node_size ) {
        for ( int i = id; i < 6; i += id_step ) {
            in_idx[node_idx*6 + i] = input_index[node*6 + i];
        }

        int t0 = lut_table[node*2 + 0];
        for ( int i = id; i < 32; i += id_step) {
            table[node_step*(i+ 0) + node_idx] = (t0 & (1 << i)) ? 0xffffffff : 0x00000000;
        }
        int t1 = lut_table[node*2 + 1];
        for ( int i = id; i < 32; i += id_step) {
            table[node_step*(i+32) + node_idx] = (t1 & (1 << i)) ? 0xffffffff : 0x00000000;
        }
    }

    __syncthreads();

    if ( node < node_size ) {
        int *y_ptr = &y_buf[node * frame_stride];

        for ( int frame = id; frame < frame_size; frame += id_step ) {
            // input
            int x[6];
            x[0] = x_buf[in_idx[node_idx*6 + 0]*frame_stride + frame];
            x[1] = x_buf[in_idx[node_idx*6 + 1]*frame_stride + frame];
            x[2] = x_buf[in_idx[node_idx*6 + 2]*frame_stride + frame];
            x[3] = x_buf[in_idx[node_idx*6 + 3]*frame_stride + frame];
            x[4] = x_buf[in_idx[node_idx*6 + 4]*frame_stride + frame];
            x[5] = x_buf[in_idx[node_idx*6 + 5]*frame_stride + frame];

            // LUT
            int y = 0;
            device_lut6_mask< 0>(y, (int)table[(node_step* 0) + node_idx], x);
            device_lut6_mask< 1>(y, (int)table[(node_step* 1) + node_idx], x);
            device_lut6_mask< 2>(y, (int)table[(node_step* 2) + node_idx], x);
            device_lut6_mask< 3>(y, (int)table[(node_step* 3) + node_idx], x);
            device_lut6_mask< 4>(y, (int)table[(node_step* 4) + node_idx], x);
            device_lut6_mask< 5>(y, (int)table[(node_step* 5) + node_idx], x);
            device_lut6_mask< 6>(y, (int)table[(node_step* 6) + node_idx], x);
            device_lut6_mask< 7>(y, (int)table[(node_step* 7) + node_idx], x);
            device_lut6_mask< 8>(y, (int)table[(node_step* 8) + node_idx], x);
            device_lut6_mask< 9>(y, (int)table[(node_step* 9) + node_idx], x);
            device_lut6_mask<10>(y, (int)table[(node_step*10) + node_idx], x);
            device_lut6_mask<11>(y, (int)table[(node_step*11) + node_idx], x);
            device_lut6_mask<12>(y, (int)table[(node_step*12) + node_idx], x);
            device_lut6_mask<13>(y, (int)table[(node_step*13) + node_idx], x);
            device_lut6_mask<14>(y, (int)table[(node_step*14) + node_idx], x);
            device_lut6_mask<15>(y, (int)table[(node_step*15) + node_idx], x);
            device_lut6_mask<16>(y, (int)table[(node_step*16) + node_idx], x);
            device_lut6_mask<17>(y, (int)table[(node_step*17) + node_idx], x);
            device_lut6_mask<18>(y, (int)table[(node_step*18) + node_idx], x);
            device_lut6_mask<19>(y, (int)table[(node_step*19) + node_idx], x);
            device_lut6_mask<20>(y, (int)table[(node_step*20) + node_idx], x);
            device_lut6_mask<21>(y, (int)table[(node_step*21) + node_idx], x);
            device_lut6_mask<22>(y, (int)table[(node_step*22) + node_idx], x);
            device_lut6_mask<23>(y, (int)table[(node_step*23) + node_idx], x);
            device_lut6_mask<24>(y, (int)table[(node_step*24) + node_idx], x);
            device_lut6_mask<25>(y, (int)table[(node_step*25) + node_idx], x);
            device_lut6_mask<26>(y, (int)table[(node_step*26) + node_idx], x);
            device_lut6_mask<27>(y, (int)table[(node_step*27) + node_idx], x);
            device_lut6_mask<28>(y, (int)table[(node_step*28) + node_idx], x);
            device_lut6_mask<29>(y, (int)table[(node_step*29) + node_idx], x);
            device_lut6_mask<30>(y, (int)table[(node_step*30) + node_idx], x);
            device_lut6_mask<31>(y, (int)table[(node_step*31) + node_idx], x);
            device_lut6_mask<32>(y, (int)table[(node_step*32) + node_idx], x);
            device_lut6_mask<33>(y, (int)table[(node_step*33) + node_idx], x);
            device_lut6_mask<34>(y, (int)table[(node_step*34) + node_idx], x);
            device_lut6_mask<35>(y, (int)table[(node_step*35) + node_idx], x);
            device_lut6_mask<36>(y, (int)table[(node_step*36) + node_idx], x);
            device_lut6_mask<37>(y, (int)table[(node_step*37) + node_idx], x);
            device_lut6_mask<38>(y, (int)table[(node_step*38) + node_idx], x);
            device_lut6_mask<39>(y, (int)table[(node_step*39) + node_idx], x);
            device_lut6_mask<40>(y, (int)table[(node_step*40) + node_idx], x);
            device_lut6_mask<41>(y, (int)table[(node_step*41) + node_idx], x);
            device_lut6_mask<42>(y, (int)table[(node_step*42) + node_idx], x);
            device_lut6_mask<43>(y, (int)table[(node_step*43) + node_idx], x);
            device_lut6_mask<44>(y, (int)table[(node_step*44) + node_idx], x);
            device_lut6_mask<45>(y, (int)table[(node_step*45) + node_idx], x);
            device_lut6_mask<46>(y, (int)table[(node_step*46) + node_idx], x);
            device_lut6_mask<47>(y, (int)table[(node_step*47) + node_idx], x);
            device_lut6_mask<48>(y, (int)table[(node_step*48) + node_idx], x);
            device_lut6_mask<49>(y, (int)table[(node_step*49) + node_idx], x);
            device_lut6_mask<50>(y, (int)table[(node_step*50) + node_idx], x);
            device_lut6_mask<51>(y, (int)table[(node_step*51) + node_idx], x);
            device_lut6_mask<52>(y, (int)table[(node_step*52) + node_idx], x);
            device_lut6_mask<53>(y, (int)table[(node_step*53) + node_idx], x);
            device_lut6_mask<54>(y, (int)table[(node_step*54) + node_idx], x);
            device_lut6_mask<55>(y, (int)table[(node_step*55) + node_idx], x);
            device_lut6_mask<56>(y, (int)table[(node_step*56) + node_idx], x);
            device_lut6_mask<57>(y, (int)table[(node_step*57) + node_idx], x);
            device_lut6_mask<58>(y, (int)table[(node_step*58) + node_idx], x);
            device_lut6_mask<59>(y, (int)table[(node_step*59) + node_idx], x);
            device_lut6_mask<60>(y, (int)table[(node_step*60) + node_idx], x);
            device_lut6_mask<61>(y, (int)table[(node_step*61) + node_idx], x);
            device_lut6_mask<62>(y, (int)table[(node_step*62) + node_idx], x);
            device_lut6_mask<63>(y, (int)table[(node_step*63) + node_idx], x);

            y_ptr[frame] = y;
        }
    }
}


int bbcu_bit_BinatyLut6_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int const       *dev_input_index,
            int const       *dev_table,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    dim3    block(256, 1);
    dim3    grid(1, node_size);
    
//    size_t shared_mem_size = block.y * ((sizeof(int) * 64) + (sizeof(int) * 6));

    kernal_bit_BinaryLut6_Forward<64 + 6><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_table,
            node_size,
            (frame_size + 31) / 32,
            frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}



// end of file
