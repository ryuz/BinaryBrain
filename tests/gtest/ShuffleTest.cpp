#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Shuffle.h"



TEST(SuffleTest, testSuffle_test0)
{
    auto shuffle = bb::Shuffle::Create(3);
    
    bb::FrameBuffer x(2, {6}, BB_TYPE_FP32);
    shuffle->SetInputShape(x.GetShape());

    x.SetFP32(0, 0, 11);
    x.SetFP32(0, 1, 12);
    x.SetFP32(0, 2, 13);
    x.SetFP32(0, 3, 14);
    x.SetFP32(0, 4, 15);
    x.SetFP32(0, 5, 16);
    x.SetFP32(1, 0, 21);
    x.SetFP32(1, 1, 22);
    x.SetFP32(1, 2, 23);
    x.SetFP32(1, 3, 24);
    x.SetFP32(1, 4, 25);
    x.SetFP32(1, 5, 26);

    auto y = shuffle->Forward(x);
    
    EXPECT_EQ(11, y.GetFP32(0, 0));
    EXPECT_EQ(13, y.GetFP32(0, 1));
    EXPECT_EQ(15, y.GetFP32(0, 2));
    EXPECT_EQ(12, y.GetFP32(0, 3));
    EXPECT_EQ(14, y.GetFP32(0, 4));
    EXPECT_EQ(16, y.GetFP32(0, 5));
    EXPECT_EQ(21, y.GetFP32(1, 0));
    EXPECT_EQ(23, y.GetFP32(1, 1));
    EXPECT_EQ(25, y.GetFP32(1, 2));
    EXPECT_EQ(22, y.GetFP32(1, 3));
    EXPECT_EQ(24, y.GetFP32(1, 4));
    EXPECT_EQ(26, y.GetFP32(1, 5));

    // backward
    auto dx = shuffle->Backward(y);

    EXPECT_EQ(11, dx.GetFP32(0, 0));
    EXPECT_EQ(12, dx.GetFP32(0, 1));
    EXPECT_EQ(13, dx.GetFP32(0, 2));
    EXPECT_EQ(14, dx.GetFP32(0, 3));
    EXPECT_EQ(15, dx.GetFP32(0, 4));
    EXPECT_EQ(16, dx.GetFP32(0, 5));
    EXPECT_EQ(21, dx.GetFP32(1, 0));
    EXPECT_EQ(22, dx.GetFP32(1, 1));
    EXPECT_EQ(23, dx.GetFP32(1, 2));
    EXPECT_EQ(24, dx.GetFP32(1, 3));
    EXPECT_EQ(25, dx.GetFP32(1, 4));
    EXPECT_EQ(26, dx.GetFP32(1, 5));
}


