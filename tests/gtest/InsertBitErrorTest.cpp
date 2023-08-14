#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/InsertBitError.h"


TEST(InsertBitErrorTest, InsertBitErrorTest_test0)
{
    auto biterr = bb::InsertBitError<>::Create(0.5);
    

    for (int i = 0; i < 3; ++i) {
        bb::FrameBuffer x(2, { 3 }, BB_TYPE_FP32);

        x.SetFP32(0, 0, 0);
        x.SetFP32(0, 1, 0);
        x.SetFP32(0, 2, 0);
        x.SetFP32(1, 0, 0);
        x.SetFP32(1, 1, 0);
        x.SetFP32(1, 2, 0);
        auto y = biterr->Forward(x);

        bool err[6];
        err[0] = y.GetFP32(0, 0);
        err[1] = y.GetFP32(0, 1);
        err[2] = y.GetFP32(0, 2);
        err[3] = y.GetFP32(1, 0);
        err[4] = y.GetFP32(1, 1);
        err[5] = y.GetFP32(1, 2);


        bb::FrameBuffer dy(2, { 3 }, BB_TYPE_FP32);
        dy.SetFP32(0, 0, +1);
        dy.SetFP32(0, 1, +1);
        dy.SetFP32(0, 2, +1);
        dy.SetFP32(1, 0, +1);
        dy.SetFP32(1, 1, +1);
        dy.SetFP32(1, 2, +1);

        auto dx = biterr->Backward(dy);
        EXPECT_EQ(dx.GetFP32(0, 0), err[0] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(0, 1), err[1] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(0, 2), err[2] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(1, 0), err[3] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(1, 1), err[4] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(1, 2), err[5] ? -1 : +1);
    }
}



TEST(InsertBitErrorTest, InsertBitErrorTest_test1)
{
    auto biterr = bb::InsertBitError<bb::Bit>::Create(0.5);
    
    for (int i = 0; i < 3; ++i) {
        bb::FrameBuffer x(2, { 3 }, BB_TYPE_BIT);

        x.SetBit(0, 0, 0);
        x.SetBit(0, 1, 0);
        x.SetBit(0, 2, 0);
        x.SetBit(1, 0, 0);
        x.SetBit(1, 1, 0);
        x.SetBit(1, 2, 0);
        auto y = biterr->Forward(x);

        bool err[6];
        err[0] = y.GetBit(0, 0);
        err[1] = y.GetBit(0, 1);
        err[2] = y.GetBit(0, 2);
        err[3] = y.GetBit(1, 0);
        err[4] = y.GetBit(1, 1);
        err[5] = y.GetBit(1, 2);

        bb::FrameBuffer dy(2, { 3 }, BB_TYPE_FP32);
        dy.SetFP32(0, 0, +1);
        dy.SetFP32(0, 1, +1);
        dy.SetFP32(0, 2, +1);
        dy.SetFP32(1, 0, +1);
        dy.SetFP32(1, 1, +1);
        dy.SetFP32(1, 2, +1);

        auto dx = biterr->Backward(dy);
        EXPECT_EQ(dx.GetFP32(0, 0), err[0] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(0, 1), err[1] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(0, 2), err[2] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(1, 0), err[3] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(1, 1), err[4] ? -1 : +1);
        EXPECT_EQ(dx.GetFP32(1, 2), err[5] ? -1 : +1);
    }
}

