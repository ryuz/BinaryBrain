#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/BitEncode.h"



TEST(BitEncodeTest, testBitEncode_test0)
{
    auto bitenc = bb::BitEncode<float>::Create(4);
    
    bb::FrameBuffer x(2, {3}, BB_TYPE_FP32);
    bitenc->SetInputShape(x.GetShape());

    x.SetFP32(0, 0, 0x55 / 255.0f);
    x.SetFP32(0, 1, 0x00 / 255.0f);
    x.SetFP32(0, 2, 0xff / 255.0f);
    x.SetFP32(1, 0, 0x11 / 255.0f);
    x.SetFP32(1, 1, 0x22 / 255.0f);
    x.SetFP32(1, 2, 0xaa / 255.0f);

    auto y = bitenc->Forward(x);
    
    EXPECT_EQ(1, (int)y.GetBit(0, 0));
    EXPECT_EQ(0, (int)y.GetBit(0, 1));
    EXPECT_EQ(1, (int)y.GetBit(0, 2));
    EXPECT_EQ(0, (int)y.GetBit(0, 3));

    EXPECT_EQ(0, (int)y.GetBit(0, 4+0));
    EXPECT_EQ(0, (int)y.GetBit(0, 4+1));
    EXPECT_EQ(0, (int)y.GetBit(0, 4+2));
    EXPECT_EQ(0, (int)y.GetBit(0, 4+3));

    EXPECT_EQ(1, (int)y.GetBit(0, 8+0));
    EXPECT_EQ(1, (int)y.GetBit(0, 8+1));
    EXPECT_EQ(1, (int)y.GetBit(0, 8+2));
    EXPECT_EQ(1, (int)y.GetBit(0, 8+3));


    EXPECT_EQ(1, (int)y.GetBit(1, 0));
    EXPECT_EQ(0, (int)y.GetBit(1, 1));
    EXPECT_EQ(0, (int)y.GetBit(1, 2));
    EXPECT_EQ(0, (int)y.GetBit(1, 3));

    EXPECT_EQ(0, (int)y.GetBit(1, 4+0));
    EXPECT_EQ(1, (int)y.GetBit(1, 4+1));
    EXPECT_EQ(0, (int)y.GetBit(1, 4+2));
    EXPECT_EQ(0, (int)y.GetBit(1, 4+3));

    EXPECT_EQ(0, (int)y.GetBit(1, 8+0));
    EXPECT_EQ(1, (int)y.GetBit(1, 8+1));
    EXPECT_EQ(0, (int)y.GetBit(1, 8+2));
    EXPECT_EQ(1, (int)y.GetBit(1, 8+3));
}


