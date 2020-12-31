
#include <stdio.h>
#include <iostream>

#include "gtest/gtest.h"
#include "bb/BitEncode.h"



TEST(BitEncodeTest, testBitEncode_test0)
{
    auto bitenc = bb::BitEncode<bb::Bit>::Create(4);
    
    bb::FrameBuffer x(2, {3}, BB_TYPE_FP32);
    bitenc->SetInputShape(x.GetShape());

    x.SetFP32(0, 0, 0x5a / 255.0f);
    x.SetFP32(0, 1, 0x0b / 255.0f);
    x.SetFP32(0, 2, 0xfc / 255.0f);
    x.SetFP32(1, 0, 0x1d / 255.0f);
    x.SetFP32(1, 1, 0x2e / 255.0f);
    x.SetFP32(1, 2, 0xaf / 255.0f);

    auto y = bitenc->Forward(x);
    EXPECT_EQ(bb::DataType<bb::Bit>::type, y.GetType());
    EXPECT_EQ(12, y.GetNodeSize());

    EXPECT_EQ(1, y.GetFP32(0, 3*0+0));
    EXPECT_EQ(0, y.GetFP32(0, 3*1+0));
    EXPECT_EQ(1, y.GetFP32(0, 3*2+0));
    EXPECT_EQ(0, y.GetFP32(0, 3*3+0));

    EXPECT_EQ(0, y.GetFP32(0, 3*0+1));
    EXPECT_EQ(0, y.GetFP32(0, 3*1+1));
    EXPECT_EQ(0, y.GetFP32(0, 3*2+1));
    EXPECT_EQ(0, y.GetFP32(0, 3*3+1));
    
    EXPECT_EQ(1, y.GetFP32(0, 3*3+2));
    EXPECT_EQ(1, y.GetFP32(0, 3*3+2));
    EXPECT_EQ(1, y.GetFP32(0, 3*3+2));
    EXPECT_EQ(1, y.GetFP32(0, 3*3+2));
    

    EXPECT_EQ(1, y.GetFP32(1, 3*0+0));
    EXPECT_EQ(0, y.GetFP32(1, 3*1+0));
    EXPECT_EQ(0, y.GetFP32(1, 3*2+0));
    EXPECT_EQ(0, y.GetFP32(1, 3*3+0));
    
    EXPECT_EQ(0, y.GetFP32(1, 3*0+1));
    EXPECT_EQ(1, y.GetFP32(1, 3*1+1));
    EXPECT_EQ(0, y.GetFP32(1, 3*2+1));
    EXPECT_EQ(0, y.GetFP32(1, 3*3+1));

    EXPECT_EQ(0, y.GetFP32(1, 3*0+2));
    EXPECT_EQ(1, y.GetFP32(1, 3*1+2));
    EXPECT_EQ(0, y.GetFP32(1, 3*2+2));
    EXPECT_EQ(1, y.GetFP32(1, 3*3+2));
}


