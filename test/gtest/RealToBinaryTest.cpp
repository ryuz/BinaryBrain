#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/RealToBinary.h"


#define USE_BACKWARD    0

TEST(RealToBinaryTest, testRealToBinary)
{
    const int node_size = 3;
    const int mux_size = 2;
    const int frame_size = 1;

    auto real2bin = bb::RealToBinary<float, bb::Bit>::Create(mux_size);

    // forward
    bb::FrameBuffer x_buf(BB_TYPE_FP32, frame_size, node_size);
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


TEST(RealToBinaryTest, testRealToBinaryyBatch)
{
    const int node_size = 3;
    const int mux_size  = 2;
    const int frame_size = 2;

    auto real2bin = bb::RealToBinary<float, bb::Bit>::Create(mux_size);

    // forward
    bb::FrameBuffer x_buf(BB_TYPE_FP32, frame_size, node_size);
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

