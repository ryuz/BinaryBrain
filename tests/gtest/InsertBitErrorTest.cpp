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


void ErrTest_bit(double error_rate, bool rev=false)
{
    const int node_size = 171;
    const int frame_size = 123;
    auto biterr = bb::InsertBitError<bb::Bit>::Create(error_rate);
    bb::FrameBuffer x(frame_size, { node_size }, BB_TYPE_BIT);
    for (int f = 0; f < frame_size; ++f) {
        for (int n = 0; n < node_size; ++n) {
            x.SetBit(f, n, rev ? 1 : 0);
        }
    }

    auto y = biterr->Forward(x);

    bool err[frame_size][node_size];
    int  e = 0;
    for (int f = 0; f < frame_size; ++f) {
        for (int n = 0; n < node_size; ++n) {
            err[f][n] = rev ? !y.GetBit(f, n) : y.GetBit(f, n);
            if (err[f][n]) {
                e++;
            }
        }
    }
    std::cout << "error_rate = " << error_rate << "  " << e << "/" << frame_size * node_size << " = " << (double)e / (double)(frame_size * node_size) << std::endl;

    bb::FrameBuffer dy(frame_size, { node_size }, BB_TYPE_FP32);
    for (int f = 0; f < frame_size; ++f) {
        for (int n = 0; n < node_size; ++n) {
            dy.SetFP32(f, n, +1);
        }
    }
    auto dx = biterr->Backward(dy);
    for (int f = 0; f < frame_size; ++f) {
        for (int n = 0; n < node_size; ++n) {
            EXPECT_EQ(dx.GetFP32(f, n), err[f][n] ? -1 : +1);
        }
    }
}


void ErrTest_fp32(double error_rate, bool rev=false)
{
    const int node_size = 171;
    const int frame_size = 123;
    auto biterr = bb::InsertBitError<float>::Create(error_rate);
    bb::FrameBuffer x(frame_size, { node_size }, BB_TYPE_FP32);
    for (int f = 0; f < frame_size; ++f) {
        for (int n = 0; n < node_size; ++n) {
            x.SetFP32(f, n, rev ? 1.0f : 0.0f);
        }
    }

    auto y = biterr->Forward(x);

    bool err[frame_size][node_size];
    int  e = 0;
    for (int f = 0; f < frame_size; ++f) {
        for (int n = 0; n < node_size; ++n) {
            err[f][n] = rev ? !y.GetFP32(f, n) : y.GetFP32(f, n);
            if (err[f][n]) {
                e++;
            }
        }
    }
    std::cout << "error_rate = " << error_rate << "  " << e << "/" << frame_size * node_size << " = " << (double)e / (double)(frame_size * node_size) << std::endl;

    bb::FrameBuffer dy(frame_size, { node_size }, BB_TYPE_FP32);
    for (int f = 0; f < frame_size; ++f) {
        for (int n = 0; n < node_size; ++n) {
            dy.SetFP32(f, n, +1);
        }
    }
    auto dx = biterr->Backward(dy);
    for (int f = 0; f < frame_size; ++f) {
        for (int n = 0; n < node_size; ++n) {
            EXPECT_EQ(dx.GetFP32(f, n), err[f][n] ? -1 : +1);
        }
    }
}



TEST(InsertBitErrorTest, InsertBitErrorTest_test3)
{
    ErrTest_bit(0.1);
    ErrTest_bit(0.2);
    ErrTest_bit(0.5);
    ErrTest_bit(0.7);
    ErrTest_bit(0.9);

    ErrTest_fp32(0.1);
    ErrTest_fp32(0.2);
    ErrTest_fp32(0.5);
    ErrTest_fp32(0.7);
    ErrTest_fp32(0.9);

    ErrTest_bit(0.1, true);
    ErrTest_bit(0.2, true);
    ErrTest_bit(0.5, true);
    ErrTest_bit(0.7, true);
    ErrTest_bit(0.9, true);

    ErrTest_fp32(0.1, true);
    ErrTest_fp32(0.2, true);
    ErrTest_fp32(0.5, true);
    ErrTest_fp32(0.7, true);
    ErrTest_fp32(0.9, true);
}
