#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/UpSampling.h"



TEST(UpSamplingTest, testUpSampling_call)
{
    auto upsmp = bb::UpSampling<>::Create(2, 3);

    bb::FrameBuffer x_buf(BB_TYPE_FP32, 16, {28, 28, 3});
    upsmp->SetInputShape(x_buf.GetShape());

    auto y_buf  = upsmp->Forward(x_buf);
    auto dx_buf = upsmp->Backward(y_buf);
}


TEST(UpSamplingTest, testUpSampling_test)
{
    auto upsmp = bb::UpSampling<>::Create(2, 3);
    
    bb::FrameBuffer x_buf(BB_TYPE_FP32, 2, {2, 3, 4});
    upsmp->SetInputShape(x_buf.GetShape());

    for ( bb::index_t f = 0; f < 2; ++f) {
        for (bb::index_t c = 0; c < 4; ++c) {
            for (bb::index_t y = 0; y < 3; ++y) {
                for (bb::index_t x = 0; x < 2; ++x) {
                    x_buf.SetFP32(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
                }
            }
        }
    }

    auto y_buf = upsmp->Forward(x_buf);

    EXPECT_EQ(bb::indices_t({2*3, 3*2, 4}), y_buf.GetShape());
    EXPECT_EQ(2, y_buf.GetFrameSize());
    
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 1, 0, 0 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 2, 0, 0 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 1, 1, 0 }));
    EXPECT_EQ(0,     y_buf.GetFP32(0, { 2, 1, 0 }));
                     
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 3, 0, 0 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 4, 0, 0 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 5, 0, 0 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 3, 1, 0 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 4, 1, 0 }));
    EXPECT_EQ(1,     y_buf.GetFP32(0, { 5, 1, 0 }));
                     
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 3, 2, 0 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 4, 2, 0 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 5, 2, 0 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 3, 3, 0 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 4, 3, 0 }));
    EXPECT_EQ(11,    y_buf.GetFP32(0, { 5, 3, 0 }));

    EXPECT_EQ(111,   y_buf.GetFP32(0, { 3, 2, 1 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 4, 2, 1 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 5, 2, 1 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 3, 3, 1 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 4, 3, 1 }));
    EXPECT_EQ(111,   y_buf.GetFP32(0, { 5, 3, 1 }));

    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 3, 4, 3 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 4, 4, 3 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 5, 4, 3 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 3, 5, 3 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 4, 5, 3 }));
    EXPECT_EQ(1321,  y_buf.GetFP32(1, { 5, 5, 3 }));

    // backward
    bb::FrameBuffer dy_buf(BB_TYPE_FP32, 2, {2*3, 3*2, 4});

    dy_buf.SetFP32(0, { 0, 0, 0 }, 11);
    dy_buf.SetFP32(0, { 1, 0, 0 }, 12);
    dy_buf.SetFP32(0, { 2, 0, 0 }, 13);
    dy_buf.SetFP32(0, { 0, 1, 0 }, 14);
    dy_buf.SetFP32(0, { 1, 1, 0 }, 15);
    dy_buf.SetFP32(0, { 2, 1, 0 }, 16);

    dy_buf.SetFP32(0, { 3, 0, 0 }, 21);
    dy_buf.SetFP32(0, { 4, 0, 0 }, 22);
    dy_buf.SetFP32(0, { 5, 0, 0 }, 23);
    dy_buf.SetFP32(0, { 3, 1, 0 }, 24);
    dy_buf.SetFP32(0, { 4, 1, 0 }, 25);
    dy_buf.SetFP32(0, { 5, 1, 0 }, 26);

    dy_buf.SetFP32(0, { 3, 2, 0 }, 31);
    dy_buf.SetFP32(0, { 4, 2, 0 }, 32);
    dy_buf.SetFP32(0, { 5, 2, 0 }, 33);
    dy_buf.SetFP32(0, { 3, 3, 0 }, 34);
    dy_buf.SetFP32(0, { 4, 3, 0 }, 35);
    dy_buf.SetFP32(0, { 5, 3, 0 }, 36);

    dy_buf.SetFP32(0, { 3, 2, 1 }, 41);
    dy_buf.SetFP32(0, { 4, 2, 1 }, 42);
    dy_buf.SetFP32(0, { 5, 2, 1 }, 43);
    dy_buf.SetFP32(0, { 3, 3, 1 }, 44);
    dy_buf.SetFP32(0, { 4, 3, 1 }, 45);
    dy_buf.SetFP32(0, { 5, 3, 1 }, 46);

    dy_buf.SetFP32(1, { 3, 4, 3 }, 51);
    dy_buf.SetFP32(1, { 4, 4, 3 }, 52);
    dy_buf.SetFP32(1, { 5, 4, 3 }, 53);
    dy_buf.SetFP32(1, { 3, 5, 3 }, 54);
    dy_buf.SetFP32(1, { 4, 5, 3 }, 55);
    dy_buf.SetFP32(1, { 5, 5, 3 }, 56);


    auto dx_buf = upsmp->Backward(dy_buf);
    EXPECT_EQ(x_buf.GetShape(),     dx_buf.GetShape());
    EXPECT_EQ(x_buf.GetFrameSize(), dx_buf.GetFrameSize());

    EXPECT_EQ(11+12+13+14+15+16, dx_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(21+22+23+24+25+26, dx_buf.GetFP32(0, { 1, 0, 0 }));                    
    EXPECT_EQ(31+32+33+34+35+36, dx_buf.GetFP32(0, { 1, 1, 0 }));
    EXPECT_EQ(41+42+43+44+45+46, dx_buf.GetFP32(0, { 1, 1, 1 }));
    EXPECT_EQ(51+52+53+54+55+56, dx_buf.GetFP32(1, { 1, 2, 3 }));
}

