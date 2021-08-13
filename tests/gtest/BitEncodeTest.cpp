
#include <stdio.h>
#include <iostream>

#include "gtest/gtest.h"
#include "bb/BitEncode.h"



TEST(BitEncodeTest, testBitEncode_test0)
{
    auto bitenc = bb::BitEncode<bb::Bit>::Create(4);
    
    bb::FrameBuffer x(2, {3}, BB_TYPE_FP32);
    bitenc->SetInputShape(x.GetShape());

    x.SetFP32(0, 0, 0x55 / 255.0f);
    x.SetFP32(0, 1, 0x00 / 255.0f);
    x.SetFP32(0, 2, 0xff / 255.0f);
    x.SetFP32(1, 0, 0x11 / 255.0f);
    x.SetFP32(1, 1, 0x22 / 255.0f);
    x.SetFP32(1, 2, 0xaa / 255.0f);

    auto y = bitenc->Forward(x);
    EXPECT_EQ(bb::DataType<bb::Bit>::type, y.GetType());
    EXPECT_EQ(12, y.GetNodeSize());

    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(0, 3*0+0));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(0, 3*1+0));
    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(0, 3*2+0));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(0, 3*3+0));

    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(0, 3*0+1));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(0, 3*1+1));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(0, 3*2+1));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(0, 3*3+1));
    
    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(0, 3*3+2));
    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(0, 3*3+2));
    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(0, 3*3+2));
    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(0, 3*3+2));
    

    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(1, 3*0+0));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(1, 3*1+0));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(1, 3*2+0));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(1, 3*3+0));
    
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(1, 3*0+1));
    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(1, 3*1+1));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(1, 3*2+1));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(1, 3*3+1));

    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(1, 3*0+2));
    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(1, 3*1+2));
    EXPECT_EQ(BB_BINARY_LO, y.GetFP32(1, 3*2+2));
    EXPECT_EQ(BB_BINARY_HI, y.GetFP32(1, 3*3+2));
}


