#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/UpSampling.h"
#include "bb/UniformDistributionGenerator.h"



TEST(UpSamplingTest, testUpSampling_call)
{
    auto upsmp = bb::UpSampling<>::Create(2, 3);

    bb::FrameBuffer x_buf(16, {3, 28, 28}, BB_TYPE_FP32);
    upsmp->SetInputShape(x_buf.GetShape());

    auto y_buf  = upsmp->Forward(x_buf);
    auto dx_buf = upsmp->Backward(y_buf);
}


TEST(UpSamplingTest, testUpSampling_test)
{
    auto upsmp = bb::UpSampling<>::Create(2, 3);
    
    bb::FrameBuffer x_buf(2, {4, 3, 2}, BB_TYPE_FP32);
    upsmp->SetInputShape(x_buf.GetShape());

    for ( bb::index_t f = 0; f < 2; ++f) {
        for (bb::index_t c = 0; c < 4; ++c) {
            for (bb::index_t y = 0; y < 3; ++y) {
                for (bb::index_t x = 0; x < 2; ++x) {
                    x_buf.SetFP32(f, { c, y, x }, (float)(1000 * f + 100 * c + 10 * y + x));
                }
            }
        }
    }

    auto y_buf = upsmp->Forward(x_buf);

    EXPECT_EQ(bb::indices_t({4, 3*2, 2*3}), y_buf.GetShape());
    EXPECT_EQ(2, y_buf.GetFrameSize());
    
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 0, 0, 1 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 0, 0, 2 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 0, 1, 1 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 0, 1, 2 }));

    EXPECT_EQ(1,     y_buf.GetFP32(0, { 0, 0, 3 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 0, 0, 4 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 0, 0, 5 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 0, 1, 3 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 0, 1, 4 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 0, 1, 5 }));

    EXPECT_EQ(11,    y_buf.GetFP32(0, { 0, 2, 3 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 0, 2, 4 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 0, 2, 5 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 0, 3, 3 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 0, 3, 4 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 0, 3, 5 }));

    EXPECT_EQ(111,   y_buf.GetFP32(0, { 1, 2, 3 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 1, 2, 4 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 1, 2, 5 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 1, 3, 3 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 1, 3, 4 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 1, 3, 5 }));

    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 3, 4, 3 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 3, 4, 4 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 3, 4, 5 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 3, 5, 3 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 3, 5, 4 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 3, 5, 5 }));

    // backward
    bb::FrameBuffer dy_buf(2, {4, 3*2, 2*3}, BB_TYPE_FP32);

    dy_buf.SetFP32(0, { 0, 0, 0 }, 11);
    dy_buf.SetFP32(0, { 0, 0, 1 }, 12);
    dy_buf.SetFP32(0, { 0, 0, 2 }, 13);
    dy_buf.SetFP32(0, { 0, 1, 0 }, 14);
    dy_buf.SetFP32(0, { 0, 1, 1 }, 15);
    dy_buf.SetFP32(0, { 0, 1, 2 }, 16);

    dy_buf.SetFP32(0, { 0, 0, 3 }, 21);
    dy_buf.SetFP32(0, { 0, 0, 4 }, 22);
    dy_buf.SetFP32(0, { 0, 0, 5 }, 23);
    dy_buf.SetFP32(0, { 0, 1, 3 }, 24);
    dy_buf.SetFP32(0, { 0, 1, 4 }, 25);
    dy_buf.SetFP32(0, { 0, 1, 5 }, 26);

    dy_buf.SetFP32(0, { 0, 2, 3 }, 31);
    dy_buf.SetFP32(0, { 0, 2, 4 }, 32);
    dy_buf.SetFP32(0, { 0, 2, 5 }, 33);
    dy_buf.SetFP32(0, { 0, 3, 3 }, 34);
    dy_buf.SetFP32(0, { 0, 3, 4 }, 35);
    dy_buf.SetFP32(0, { 0, 3, 5 }, 36);

    dy_buf.SetFP32(0, { 1, 2, 3 }, 41);
    dy_buf.SetFP32(0, { 1, 2, 4 }, 42);
    dy_buf.SetFP32(0, { 1, 2, 5 }, 43);
    dy_buf.SetFP32(0, { 1, 3, 3 }, 44);
    dy_buf.SetFP32(0, { 1, 3, 4 }, 45);
    dy_buf.SetFP32(0, { 1, 3, 5 }, 46);

    dy_buf.SetFP32(1, { 3, 4, 3 }, 51);
    dy_buf.SetFP32(1, { 3, 4, 4 }, 52);
    dy_buf.SetFP32(1, { 3, 4, 5 }, 53);
    dy_buf.SetFP32(1, { 3, 5, 3 }, 54);
    dy_buf.SetFP32(1, { 3, 5, 4 }, 55);
    dy_buf.SetFP32(1, { 3, 5, 5 }, 56);


    auto dx_buf = upsmp->Backward(dy_buf);
    EXPECT_EQ(x_buf.GetShape(),     dx_buf.GetShape());
    EXPECT_EQ(x_buf.GetFrameSize(), dx_buf.GetFrameSize());

    EXPECT_EQ(11+12+13+14+15+16, dx_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(21+22+23+24+25+26, dx_buf.GetFP32(0, { 0, 0, 1 }));                    
    EXPECT_EQ(31+32+33+34+35+36, dx_buf.GetFP32(0, { 0, 1, 1 }));
    EXPECT_EQ(41+42+43+44+45+46, dx_buf.GetFP32(0, { 1, 1, 1 }));
    EXPECT_EQ(51+52+53+54+55+56, dx_buf.GetFP32(1, { 3, 2, 1 }));
}



#ifdef BB_WITH_CUDA

template<typename FT = float, typename BT = float>
void UpSamplingTest_cmp
        (
            int     frame_size,
            int     input_w_size,
            int     input_h_size, 
            int     c_size,
            int     filter_w_size,
            int     filter_h_size,
            bool    fill,
            int     loop_num,
            bool    host_only = true
        )
{
    auto model0 = bb::UpSampling<FT, BT>::Create(filter_h_size, filter_w_size, fill);
    auto model1 = bb::UpSampling<FT, BT>::Create(filter_h_size, filter_w_size, fill);

    if ( host_only ) {
        model1->SendCommand("host_only true");
    }

    bb::FrameBuffer x_buf0(frame_size, {c_size, input_h_size, input_w_size}, bb::DataType<FT>::type, false);
    bb::FrameBuffer x_buf1(frame_size, {c_size, input_h_size, input_w_size}, bb::DataType<FT>::type, host_only);

    bb::indices_t output_shape({c_size, input_h_size*filter_h_size, input_w_size*filter_w_size});

    auto input_node_size  = x_buf0.GetNodeSize();
    auto output_node_size = input_node_size * filter_w_size * filter_h_size;

    model0->SetInputShape(x_buf0.GetShape());
    model1->SetInputShape(x_buf1.GetShape());
    
    auto valgen = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1);

    for ( int loop = 0; loop < loop_num; ++ loop ) 
    {
        {
            auto x_ptr0 = x_buf0.Lock<FT>();
            auto x_ptr1 = x_buf1.Lock<FT>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < input_node_size; ++node ) {
                    if ( bb::DataType<FT>::type == BB_TYPE_BIT ) {
                        bool val = (valgen->GetValue() > 0.5);
                        x_ptr0.Set(frame, node, val);
                        x_ptr1.Set(frame, node, val);
                    }
                    else {
                        FT val = (FT)valgen->GetValue();
                        x_ptr0.Set(frame, node, val);
                        x_ptr1.Set(frame, node, val);
                    }
                }
            }
        }

        auto y_buf0 = model0->Forward(x_buf0);
        auto y_buf1 = model1->Forward(x_buf1);

        EXPECT_EQ(output_node_size, y_buf0.GetNodeSize());
        EXPECT_EQ(output_node_size, y_buf1.GetNodeSize());
        EXPECT_EQ(frame_size, y_buf0.GetFrameSize());
        EXPECT_EQ(frame_size, y_buf1.GetFrameSize());

        {
            auto y_ptr0 = y_buf0.LockConst<FT>();
            auto y_ptr1 = y_buf1.LockConst<FT>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < output_node_size; ++node ) {
                    FT val0 = y_ptr0.Get(frame, node);
                    FT val1 = y_ptr1.Get(frame, node);
                    EXPECT_EQ(val0, val1);
                }
            }
        }

        // backward
        bb::FrameBuffer dy_buf0(frame_size, output_shape, BB_TYPE_FP32);
        bb::FrameBuffer dy_buf1(frame_size, output_shape, BB_TYPE_FP32);
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                float val = valgen->GetValue();
                dy_buf0.SetFP32(frame, node, val);
                dy_buf1.SetFP32(frame, node, val);
            }
        }

        auto dx_buf0 = model0->Backward(dy_buf0);
        auto dx_buf1 = model1->Backward(dy_buf1);

        EXPECT_EQ(input_node_size, dx_buf0.GetNodeSize());
        EXPECT_EQ(input_node_size, dx_buf1.GetNodeSize());
        EXPECT_EQ(frame_size, dx_buf0.GetFrameSize());
        EXPECT_EQ(frame_size, dx_buf1.GetFrameSize());

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                auto val0 = dx_buf0.GetFP32(frame, node);
                auto val1 = dx_buf1.GetFP32(frame, node);
                EXPECT_NEAR(val0, val1, 0.001f);
            }
        }
    }
}


TEST(UpSamplingTest, testUpSampling_cmp)
{
     UpSamplingTest_cmp<float>(32, 3, 4, 32, 2, 2, true, 2);
     UpSamplingTest_cmp<bb::Bit>(32, 3, 4, 32, 2, 2, true, 2);
}


TEST(UpSamplingTest, testUpSampling_stack)
{
    int frame_size = 17;

    bb::FrameBuffer x0(frame_size, {32, 32, 72}, BB_TYPE_BIT);
    bb::FrameBuffer x1(frame_size, {32, 16, 24}, BB_TYPE_BIT);
    bb::FrameBuffer x2(frame_size, {32,  8, 12}, BB_TYPE_BIT);
    bb::FrameBuffer x3(frame_size, {32, 33, 17}, BB_TYPE_BIT);

    bb::FrameBuffer dy0(frame_size, {32, 32*2, 72*2}, BB_TYPE_FP32);
    bb::FrameBuffer dy1(frame_size, {32, 16*2, 24*2}, BB_TYPE_FP32);
    bb::FrameBuffer dy2(frame_size, {32,  8*2, 12*2}, BB_TYPE_FP32);
    bb::FrameBuffer dy3(frame_size, {32, 33*2, 17*2}, BB_TYPE_FP32);

    auto up = bb::UpSampling<bb::Bit>::Create(2, 2);
    up->SetInputShape(x0.GetShape());
    auto y0 = up->Forward(x0);
    auto y1 = up->Forward(x1);
    auto y2 = up->Forward(x2);
    auto y3 = up->Forward(x3);

    auto dx3 = up->Backward(dy3);
    auto dx2 = up->Backward(dy2);
    auto dx1 = up->Backward(dy1);
    auto dx0 = up->Backward(dy0);

    EXPECT_EQ(y0.GetShape(), dy0.GetShape());
    EXPECT_EQ(y1.GetShape(), dy1.GetShape());
    EXPECT_EQ(y2.GetShape(), dy2.GetShape());
    EXPECT_EQ(y3.GetShape(), dy3.GetShape());
    EXPECT_EQ(dx0.GetShape(), x0.GetShape());
    EXPECT_EQ(dx1.GetShape(), x1.GetShape());
    EXPECT_EQ(dx2.GetShape(), x2.GetShape());
    EXPECT_EQ(dx3.GetShape(), x3.GetShape());
}



#endif



