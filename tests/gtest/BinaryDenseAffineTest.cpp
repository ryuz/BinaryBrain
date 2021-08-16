

#include <random>
#include <iostream>
#include <stdio.h>

#include "gtest/gtest.h"
#include "bb/BinaryDenseAffine.h"


TEST(BinaryDenseAffineTest, test0)
{
    const int n = 1330;
    const int c = 67;

    auto bda = bb::BinaryDenseAffine<>::Create({c});

    auto affine = bb::DenseAffine<>::Create({c});
    auto bn     = bb::BatchNormalization<>::Create();
    auto act    = bb::Binarize<>::Create();
    
    bda->SetInputShape({c});
    affine->SetInputShape({c});
    bn->SetInputShape({c});
    act->SetInputShape({c});

    std::mt19937_64                 mt;
    std::normal_distribution<float> dist(0, 1);

    // forward
    bb::FrameBuffer x_buf(n, {c}, BB_TYPE_FP32);
    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            x_buf.SetFP32(i, j, dist(mt));
        }
    }

    auto y_buf0 = bda->Forward(x_buf, true);

    auto x0_buf1 = affine->Forward(x_buf, true);
    auto x1_buf1 = bn->Forward(x0_buf1, true);
    auto y_buf1 = act->Forward(x1_buf1, true);

    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            EXPECT_EQ(y_buf0.GetFP32(i, j), y_buf1.GetFP32(i, j));
        }
    }

    // backward
    bb::FrameBuffer dy_buf(n, {c}, BB_TYPE_FP32);
    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            dy_buf.SetFP32(i, j, dist(mt));
        }
    }

    auto dx_buf0 = bda->Backward(dy_buf);

    auto dy0_buf1 = act->Backward(dy_buf);
    auto dy1_buf1 = bn->Backward(dy0_buf1);
    auto dx_buf1 = affine->Backward(dy1_buf1);

    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            EXPECT_EQ(dx_buf0.GetFP32(i, j), dx_buf1.GetFP32(i, j));
        }
    }
}



TEST(BinaryDenseAffineTest, test1)
{
    const int n = 1330;
    const int c = 67;

    auto bda = bb::BinaryDenseAffine<bb::Bit>::Create({c});

    auto affine = bb::DenseAffine<>::Create({c});
    auto bn     = bb::BatchNormalization<>::Create();
    auto act    = bb::Binarize<bb::Bit>::Create();
    
    bda->SetInputShape({c});
    affine->SetInputShape({c});
    bn->SetInputShape({c});
    act->SetInputShape({c});

    std::mt19937_64                 mt;
    std::normal_distribution<float> dist(0, 1);

    // forward
    bb::FrameBuffer x_buf(n, {c}, BB_TYPE_BIT);
    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            x_buf.SetBit(i, j, dist(mt) > 0);
        }
    }

    auto y_buf0 = bda->Forward(x_buf, true);

    auto x0_buf1 = affine->Forward(x_buf, true);
    auto x1_buf1 = bn->Forward(x0_buf1, true);
    auto y_buf1 = act->Forward(x1_buf1, true);

    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            EXPECT_EQ(y_buf0.GetBit(i, j), y_buf1.GetBit(i, j));
        }
    }

    // backward
    bb::FrameBuffer dy_buf(n, {c}, BB_TYPE_FP32);
    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            dy_buf.SetFP32(i, j, dist(mt));
        }
    }

    auto dx_buf0 = bda->Backward(dy_buf);

    auto dy0_buf1 = act->Backward(dy_buf);
    auto dy1_buf1 = bn->Backward(dy0_buf1);
    auto dx_buf1 = affine->Backward(dy1_buf1);

    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            EXPECT_EQ(dx_buf0.GetFP32(i, j), dx_buf1.GetFP32(i, j));
        }
    }
}





TEST(BinaryDenseAffineTest, test2)
{
    const int n = 1330;
    const int c = 67;

    auto bda = bb::BinaryDenseAffine<bb::Bit>::Create({c});
    bda->SendCommand("memory_saving false");

    auto affine = bb::DenseAffine<>::Create({c});
    auto bn     = bb::BatchNormalization<>::Create();
    auto act    = bb::Binarize<bb::Bit>::Create();
    
    bda->SetInputShape({c});
    affine->SetInputShape({c});
    bn->SetInputShape({c});
    act->SetInputShape({c});

    std::mt19937_64                 mt;
    std::normal_distribution<float> dist(0, 1);

    // forward
    bb::FrameBuffer x_buf(n, {c}, BB_TYPE_BIT);
    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            x_buf.SetBit(i, j, dist(mt) > 0);
        }
    }

    auto y_buf0 = bda->Forward(x_buf, true);

    auto x0_buf1 = affine->Forward(x_buf, true);
    auto x1_buf1 = bn->Forward(x0_buf1, true);
    auto y_buf1 = act->Forward(x1_buf1, true);

    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            EXPECT_EQ(y_buf0.GetBit(i, j), y_buf1.GetBit(i, j));
        }
    }

    // backward
    bb::FrameBuffer dy_buf(n, {c}, BB_TYPE_FP32);
    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            dy_buf.SetFP32(i, j, dist(mt));
        }
    }

    auto dx_buf0 = bda->Backward(dy_buf);

    auto dy0_buf1 = act->Backward(dy_buf);
    auto dy1_buf1 = bn->Backward(dy0_buf1);
    auto dx_buf1 = affine->Backward(dy1_buf1);

    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < c; ++j ) {
            EXPECT_EQ(dx_buf0.GetFP32(i, j), dx_buf1.GetFP32(i, j));
        }
    }
}



