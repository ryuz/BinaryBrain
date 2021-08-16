#include <stdio.h>
#include <iostream>
#include <random>
#include "gtest/gtest.h"

#include "bbcu/bbcu.h"
#include "bb/Concatenate.h"



#ifdef BB_WITH_CUDA


template <typename T>
void PrintFrameBuf(bb::FrameBuffer buf)
{
    for (bb::index_t frame = 0; frame < buf.GetFrameSize(); ++frame) {
        for (bb::index_t node = 0; node < buf.GetNodeSize(); ++node) {
            std::cout << "[" << frame << "]" << "[" << node << "] : "
                    << buf.Get<T, T>(frame, node) << std::endl;
        }
    }
}


TEST(ConcatenateTest, testConcatenateTest)
{
    int frame_size = 4;
    int node0_size = 2;
    int node1_size = 3;

    bb::FrameBuffer x0_buf(frame_size, {node0_size}, BB_TYPE_FP32);
    bb::FrameBuffer x1_buf(frame_size, {node1_size}, BB_TYPE_FP32);
    
    std::mt19937_64 mt(1);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int frame = 0; frame < frame_size; ++frame ) {
        for (int node = 0; node < node0_size; ++node ) {
            x0_buf.SetFP32(frame, node, dist(mt));
        }
        for (int node = 0; node < node1_size; ++node ) {
            x1_buf.SetFP32(frame, node, dist(mt));
        }
    }

    auto concat = bb::Concatenate::Create();

    auto y_bufs = concat->ForwardMulti({x0_buf, x1_buf});
    EXPECT_EQ(y_bufs.size(), 1);
    EXPECT_EQ(y_bufs[0].GetFrameSize(), frame_size);
    EXPECT_EQ(y_bufs[0].GetNodeSize(), node0_size+node1_size);

    auto dx_bufs = concat->BackwardMulti(y_bufs);


    for (int frame = 0; frame < frame_size; ++frame ) {
        for (int node = 0; node < node0_size; ++node ) {
            EXPECT_EQ(x0_buf.GetFP32(frame, node), dx_bufs[0].GetFP32(frame, node));
        }
        for (int node = 0; node < node1_size; ++node ) {
            EXPECT_EQ(x1_buf.GetFP32(frame, node), dx_bufs[1].GetFP32(frame, node));
        }
    }

#if 0
    std::cout << "x0_buf" << std::endl;
    PrintFrameBuf<float>(x0_buf);
    std::cout << "x1_buf" << std::endl;
    PrintFrameBuf<float>(x1_buf);
    std::cout << "y_bufs" << std::endl;
    PrintFrameBuf<float>(y_bufs[0]);
#endif
}


#endif


// end of file

