#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/RealToBinary.h"
#include "bb/UniformDistributionGenerator.h"


#define USE_BACKWARD    0

TEST(RealToBinaryTest, testRealToBinary)
{
    const int node_size = 3;
    const int mux_size = 2;
    const int frame_size = 1;

    auto real2bin = bb::RealToBinary<bb::Bit>::Create(mux_size);

    // forward
    bb::FrameBuffer x_buf(frame_size, {node_size}, BB_TYPE_FP32);
    real2bin->SetInputShape(x_buf.GetShape());

    x_buf.SetFP32(0, 0, 0.0f);
    x_buf.SetFP32(0, 1, 1.0f);
    x_buf.SetFP32(0, 2, 0.5f);
    
    auto y_buf = real2bin->Forward(x_buf);

    EXPECT_EQ(frame_size,          x_buf.GetFrameSize());
    EXPECT_EQ(frame_size*mux_size, y_buf.GetFrameSize());

    EXPECT_EQ(false, y_buf.GetBit(0, 0));
    EXPECT_EQ(false, y_buf.GetBit(1, 0));
    EXPECT_EQ(true,  y_buf.GetBit(0, 1));
    EXPECT_EQ(true,  y_buf.GetBit(1, 1));

#if USE_BACKWARD 
    // backward
    bb::FrameBuffer dy_buf(BB_TYPE_FP32, y_buf.GetFrameSize(), y_buf.GetShape());

    dy_buf.SetFP32(0, 0, 0);
    dy_buf.SetFP32(1, 0, 0);
    dy_buf.SetFP32(0, 1, 1);
    dy_buf.SetFP32(1, 1, 1);
    dy_buf.SetFP32(0, 2, 2);
    dy_buf.SetFP32(1, 2, 2);
    
    auto dx_buf = real2bin->Backward(dy_buf);
    
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, 0));
    EXPECT_EQ(2.0f, dx_buf.GetFP32(0, 1));
    EXPECT_EQ(4.0f, dx_buf.GetFP32(0, 2));
#endif
}


TEST(RealToBinaryTest, testRealToBinaryBatch)
{
    const int node_size = 3;
    const int mux_size  = 2;
    const int frame_size = 2;

    auto real2bin = bb::RealToBinary<bb::Bit>::Create(mux_size);

    // forward
    bb::FrameBuffer x_buf(frame_size, {node_size}, BB_TYPE_FP32);
    real2bin->SetInputShape(x_buf.GetShape());

    x_buf.SetFP32(0, 0, 0.0f);
    x_buf.SetFP32(0, 1, 1.0f);
    x_buf.SetFP32(0, 2, 0.5f);
    x_buf.SetFP32(1, 0, 1.0f);
    x_buf.SetFP32(1, 1, 0.5f);
    x_buf.SetFP32(1, 2, 0.0f);

    auto y_buf = real2bin->Forward(x_buf);

    EXPECT_EQ(frame_size,          x_buf.GetFrameSize());
    EXPECT_EQ(frame_size*mux_size, y_buf.GetFrameSize());
    
    EXPECT_EQ(true,  y_buf.GetBit(2, 0));
    EXPECT_EQ(true,  y_buf.GetBit(3, 0));
    EXPECT_EQ(false, y_buf.GetBit(2, 2));
    EXPECT_EQ(false, y_buf.GetBit(3, 2));

    // backward
#if USE_BACKWARD 
    bb::FrameBuffer dy_buf(BB_TYPE_FP32, frame_size*mux_size, node_size);

    dy_buf.SetFP32(0, 0, 0);
    dy_buf.SetFP32(1, 0, 0);
    dy_buf.SetFP32(0, 1, 1);
    dy_buf.SetFP32(1, 1, 2);
    dy_buf.SetFP32(0, 2, 3);
    dy_buf.SetFP32(1, 2, 4);
    dy_buf.SetFP32(2, 0, 5);
    dy_buf.SetFP32(3, 0, 6);
    dy_buf.SetFP32(2, 1, 7);
    dy_buf.SetFP32(3, 1, 8);
    dy_buf.SetFP32(2, 2, 9);
    dy_buf.SetFP32(3, 2, 10);
    
    auto dx_buf = real2bin->Backward(dy_buf);
    
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, 0));
    EXPECT_EQ(3.0f, dx_buf.GetFP32(0, 1));
    EXPECT_EQ(7.0f, dx_buf.GetFP32(0, 2));

    EXPECT_EQ(11.0f, dx_buf.GetFP32(1, 0));
    EXPECT_EQ(15.0f, dx_buf.GetFP32(1, 1));
    EXPECT_EQ(19.0f, dx_buf.GetFP32(1, 2));
#endif
}


void RealToBinaryTest_cmp_bit(int node_size, int frame_size, int loop_num = 1)
{
    auto real2bin0 = bb::RealToBinary<float>::Create();
    auto real2bin1 = bb::RealToBinary<bb::Bit>::Create();

    // forward
    bb::FrameBuffer x_buf0(frame_size, {node_size}, BB_TYPE_FP32);
    bb::FrameBuffer x_buf1(frame_size, {node_size}, BB_TYPE_FP32);
    real2bin0->SetInputShape(x_buf0.GetShape());
    real2bin1->SetInputShape(x_buf1.GetShape());

    auto valgen = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1);

    for ( int loop = 0; loop < loop_num; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                float val = valgen->GetValue();
                x_buf0.SetFP32(frame, node, val);
                x_buf1.SetFP32(frame, node, val);
            }
        }

        auto y_buf0 = real2bin0->Forward(x_buf0);
        auto y_buf1 = real2bin1->Forward(x_buf1);

        EXPECT_EQ(BB_TYPE_FP32, y_buf0.GetType());
        EXPECT_EQ(BB_TYPE_BIT,  y_buf1.GetType());

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                float val0 = x_buf0.GetFP32(frame, node);
                float val1 = x_buf1.GetFP32(frame, node);
                EXPECT_EQ(val0, val1);
            }
        }

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                auto val0 = (float)y_buf0.GetFP32(frame, node);
                auto val1 = (float)y_buf1.GetBit(frame, node);
//                std::cout << "fp32 x : " << x_buf0.GetFP32(frame, node) << "  y : " << y_buf0.GetFP32(frame, node) << std::endl;
//                std::cout << "bit  x : " << x_buf1.GetFP32(frame, node) << "  y : " << (float)y_buf1.GetBit(frame, node) << std::endl;
                EXPECT_EQ(val0, val1);
            }
        }
    }
}


TEST(RealToBinaryTest, testRealToBinary_cmp_bit)
{
    RealToBinaryTest_cmp_bit(1, 8);
    RealToBinaryTest_cmp_bit(1, 2048);
    RealToBinaryTest_cmp_bit(2048, 1);
    RealToBinaryTest_cmp_bit(32, 32);
    RealToBinaryTest_cmp_bit(1024, 1024);
}

