
#include <stdio.h>
#include <iostream>

#include "gtest/gtest.h"
#include "bb/DenseAffineQuantize.h"




TEST(DenseAffineQuantizeTest, test1)
{
    auto affine = bb::DenseAffineQuantize<>::Create(3);
    
#if 0
    affine->SetInputShape({2});

    // forward
    bb::FrameBuffer x_buf(1, {2}, BB_TYPE_FP32);

    x_buf.SetFP32(0, 0, 1);
    x_buf.SetFP32(0, 1, 2);
    EXPECT_EQ(1, x_buf.GetFP32(0, 0));
    EXPECT_EQ(2, x_buf.GetFP32(0, 1));

    {
        auto W = affine->lock_W();
        auto b = affine->lock_b();
        W(0, 0) = 1;
        W(0, 1) = 2;
        W(1, 0) = 10;
        W(1, 1) = 20;
        W(2, 0) = 100;
        W(2, 1) = 200;
        b(0) = 1000;
        b(1) = 2000;
        b(2) = 3000;
    }

    auto y_buf = affine->Forward(x_buf);

    EXPECT_EQ(1 *   1 + 2 *   2 + 1000, y_buf.GetFP32(0, 0));
    EXPECT_EQ(1 *  10 + 2 *  20 + 2000, y_buf.GetFP32(0, 1));
    EXPECT_EQ(1 * 100 + 2 * 200 + 3000, y_buf.GetFP32(0, 2));

    
    // backward
    
    bb::FrameBuffer dy_buf(1, {3}, BB_TYPE_FP32);

    dy_buf.SetFP32(0, 0, 998);
    dy_buf.SetFP32(0, 1, 2042);
    dy_buf.SetFP32(0, 2, 3491);

    auto dx_buf = affine->Backward(dy_buf);

    EXPECT_EQ(370518, dx_buf.GetFP32(0, 0));
    EXPECT_EQ(741036, dx_buf.GetFP32(0, 1));

    {
        auto dW = affine->lock_dW_const();

        EXPECT_EQ(998,  dW(0, 0));
        EXPECT_EQ(2042, dW(1, 0));
        EXPECT_EQ(3491, dW(2, 0));
        EXPECT_EQ(1996, dW(0, 1));
        EXPECT_EQ(4084, dW(1, 1));
        EXPECT_EQ(6982, dW(2, 1));
    }
#endif
}

