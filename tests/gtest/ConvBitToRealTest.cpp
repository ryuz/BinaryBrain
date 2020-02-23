#include <stdio.h>
#include <iostream>
#include <random>
#include "gtest/gtest.h"

#include "bbcu/bbcu.h"
#include "bb/FrameBuffer.h"



#ifdef BB_WITH_CUDA

TEST(ConvBitToRealTest, testConvBitToRealTest)
{
    int frame_size = 1234;
    int node_size  = 3456;

    bb::FrameBuffer buf_bit (frame_size, {node_size}, BB_TYPE_BIT);
    bb::FrameBuffer buf_fp32(frame_size, {node_size}, BB_TYPE_FP32);
    
    std::mt19937_64 mt(1);
    std::uniform_int_distribution<int> dist(0, 1);
    for (int frame = 0; frame < frame_size; ++frame ) {
        for (int node = 0; node < node_size; ++node ) {
            buf_bit.SetBit(frame, node, dist(mt) == 1);
        }
    }

    {
        auto x_ptr = buf_bit.LockDeviceMemoryConst();
        auto y_ptr = buf_fp32.LockDeviceMemory(true);
    
        bbcu_ConvBitToReal<float>(
                (int const *)x_ptr.GetAddr(),
                (float     *)y_ptr.GetAddr(),
                0.0f,
                1.0f,
                (int)node_size,
                (int)frame_size,
                (int)(buf_bit.GetFrameStride() / sizeof(int)),
                (int)(buf_fp32.GetFrameStride() / sizeof(float))
            );
    }

    for (int frame = 0; frame < frame_size; ++frame ) {
        for (int node = 0; node < node_size; ++node ) {
            bool  x = buf_bit.GetBit(frame, node);
            float y = buf_fp32.GetFP32(frame, node);
            EXPECT_EQ(x ? 1.0f : 0.0f, y);
        }
    }
}


#endif


// end of file

