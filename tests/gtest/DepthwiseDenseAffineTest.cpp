
#include <stdio.h>
#include <iostream>

#include "gtest/gtest.h"
#include "bb/DepthwiseDenseAffine.h"


TEST(DepthwiseDenseAffineTest, testAffine)
{
    auto affine = bb::DepthwiseDenseAffine<>::Create({3});
    
    affine->SetInputShape({3, 2});

    // forward
    bb::FrameBuffer x_buf(1, {3, 2}, BB_TYPE_FP32);

    x_buf.SetFP32(0, 0, 1);
    x_buf.SetFP32(0, 1, 2);
    x_buf.SetFP32(0, 2, 3);
    x_buf.SetFP32(0, 3, 4);
    x_buf.SetFP32(0, 4, 5);
    x_buf.SetFP32(0, 5, 6);
    
    EXPECT_EQ(1, x_buf.GetFP32(0, 0));
    EXPECT_EQ(2, x_buf.GetFP32(0, 1));
    EXPECT_EQ(3, x_buf.GetFP32(0, 2));
    EXPECT_EQ(4, x_buf.GetFP32(0, 3));
    EXPECT_EQ(5, x_buf.GetFP32(0, 4));
    EXPECT_EQ(6, x_buf.GetFP32(0, 5));

    {
        auto W = affine->lock_W();
        auto b = affine->lock_b();
        W(0, 0, 0) = 1;
        W(0, 0, 1) = 2;
        W(1, 0, 0) = 10;
        W(1, 0, 1) = 20;
        W(2, 0, 0) = 100;
        W(2, 0, 1) = 200;
        b(0, 0) = 1000;
        b(1, 0) = 2000;
        b(2, 0) = 3000;
    }

    auto y_buf = affine->Forward(x_buf);

    EXPECT_EQ(1 *   1 + 2 *   2 + 1000, y_buf.GetFP32(0, 0));
    EXPECT_EQ(3 *  10 + 4 *  20 + 2000, y_buf.GetFP32(0, 1));
    EXPECT_EQ(5 * 100 + 6 * 200 + 3000, y_buf.GetFP32(0, 2));

    
    // backward
    bb::FrameBuffer dy_buf(1, {3}, BB_TYPE_FP32);

    dy_buf.SetFP32(0, 0, 123);
    dy_buf.SetFP32(0, 1, 456);
    dy_buf.SetFP32(0, 2, 789);

    auto dx_buf = affine->Backward(dy_buf);

    EXPECT_EQ(123 * 1, dx_buf.GetFP32(0, 0));
    EXPECT_EQ(123 * 2, dx_buf.GetFP32(0, 1));
    EXPECT_EQ(456 * 10, dx_buf.GetFP32(0, 2));
    EXPECT_EQ(456 * 20, dx_buf.GetFP32(0, 3));
    EXPECT_EQ(789 * 100, dx_buf.GetFP32(0, 4));
    EXPECT_EQ(789 * 200, dx_buf.GetFP32(0, 5));

    {
        auto db = affine->lock_db_const();

        EXPECT_EQ(123, db(0, 0));
        EXPECT_EQ(456, db(1, 0));
        EXPECT_EQ(789, db(2, 0));
    }

    {
        auto dW = affine->lock_dW_const();

        EXPECT_EQ(1 * 123, dW(0, 0, 0));
        EXPECT_EQ(2 * 123, dW(0, 0, 1));
        EXPECT_EQ(3 * 456, dW(1, 0, 0));
        EXPECT_EQ(4 * 456, dW(1, 0, 1));
        EXPECT_EQ(5 * 789, dW(2, 0, 0));
        EXPECT_EQ(6 * 789, dW(2, 0, 1));
    }
}

