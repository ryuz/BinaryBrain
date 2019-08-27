#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/ConvolutionIm2Col.h"
// #include "bb/ConvolutionCol2Im.h"


TEST(ConvolutionIm2ColTest, testConvolutionIm2Col_xy)
{
    auto cnvim2col = bb::ConvolutionIm2Col<>::Create(2, 3);

    bb::FrameBuffer buf_x(16, {28, 28, 3}, BB_TYPE_FP32);
//  cnvim2col->SetInputShape({28, 28, 3});
    bb::FrameBuffer buf_y = cnvim2col->Forward(buf_x);
}


TEST(ConvolutionIm2ColTest, testConvolutionIm2Col)
{
    auto cnvim2col = bb::ConvolutionIm2Col<>::Create(2, 3);
    
    bb::FrameBuffer buf_x(2, {4, 3, 2}, BB_TYPE_FP32);

    for ( bb::index_t f = 0; f < 2; ++f) {
        for (bb::index_t c = 0; c < 2; ++c) {
            for (bb::index_t y = 0; y < 3; ++y) {
                for (bb::index_t x = 0; x < 4; ++x) {
                    buf_x.SetFP32(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
                }
            }
        }
    }

    auto buf_y = cnvim2col->Forward(buf_x);

//  buf_y.Reshape({ 3, 2, 2 });
    EXPECT_EQ(0,   buf_y.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(1,   buf_y.GetFP32(0, { 1, 0, 0 }));
    EXPECT_EQ(2,   buf_y.GetFP32(0, { 2, 0, 0 }));
    EXPECT_EQ(10,  buf_y.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(11,  buf_y.GetFP32(0, { 1, 1, 0 }));
    EXPECT_EQ(12,  buf_y.GetFP32(0, { 2, 1, 0 }));
    EXPECT_EQ(100, buf_y.GetFP32(0, { 0, 0, 1 }));
    EXPECT_EQ(101, buf_y.GetFP32(0, { 1, 0, 1 }));
    EXPECT_EQ(102, buf_y.GetFP32(0, { 2, 0, 1 }));
    EXPECT_EQ(110, buf_y.GetFP32(0, { 0, 1, 1 }));
    EXPECT_EQ(111, buf_y.GetFP32(0, { 1, 1, 1 }));
    EXPECT_EQ(112, buf_y.GetFP32(0, { 2, 1, 1 }));

    EXPECT_EQ(1,   buf_y.GetFP32(1, { 0, 0, 0 }));
    EXPECT_EQ(2,   buf_y.GetFP32(1, { 1, 0, 0 }));
    EXPECT_EQ(3,   buf_y.GetFP32(1, { 2, 0, 0 }));
    EXPECT_EQ(11,  buf_y.GetFP32(1, { 0, 1, 0 }));
    EXPECT_EQ(12,  buf_y.GetFP32(1, { 1, 1, 0 }));
    EXPECT_EQ(13,  buf_y.GetFP32(1, { 2, 1, 0 }));
    EXPECT_EQ(101, buf_y.GetFP32(1, { 0, 0, 1 }));
    EXPECT_EQ(102, buf_y.GetFP32(1, { 1, 0, 1 }));
    EXPECT_EQ(103, buf_y.GetFP32(1, { 2, 0, 1 }));
    EXPECT_EQ(111, buf_y.GetFP32(1, { 0, 1, 1 }));
    EXPECT_EQ(112, buf_y.GetFP32(1, { 1, 1, 1 }));
    EXPECT_EQ(113, buf_y.GetFP32(1, { 2, 1, 1 }));

    EXPECT_EQ(10,  buf_y.GetFP32(2, { 0, 0, 0 }));
    EXPECT_EQ(11,  buf_y.GetFP32(2, { 1, 0, 0 }));
    EXPECT_EQ(12,  buf_y.GetFP32(2, { 2, 0, 0 }));
    EXPECT_EQ(20,  buf_y.GetFP32(2, { 0, 1, 0 }));
    EXPECT_EQ(21,  buf_y.GetFP32(2, { 1, 1, 0 }));
    EXPECT_EQ(22,  buf_y.GetFP32(2, { 2, 1, 0 }));
    EXPECT_EQ(110, buf_y.GetFP32(2, { 0, 0, 1 }));
    EXPECT_EQ(111, buf_y.GetFP32(2, { 1, 0, 1 }));
    EXPECT_EQ(112, buf_y.GetFP32(2, { 2, 0, 1 }));
    EXPECT_EQ(120, buf_y.GetFP32(2, { 0, 1, 1 }));
    EXPECT_EQ(121, buf_y.GetFP32(2, { 1, 1, 1 }));
    EXPECT_EQ(122, buf_y.GetFP32(2, { 2, 1, 1 }));

    EXPECT_EQ(11,   buf_y.GetFP32(3, { 0, 0, 0 }));
    EXPECT_EQ(12,   buf_y.GetFP32(3, { 1, 0, 0 }));
    EXPECT_EQ(13,   buf_y.GetFP32(3, { 2, 0, 0 }));
    EXPECT_EQ(21,   buf_y.GetFP32(3, { 0, 1, 0 }));
    EXPECT_EQ(22,   buf_y.GetFP32(3, { 1, 1, 0 }));
    EXPECT_EQ(23,   buf_y.GetFP32(3, { 2, 1, 0 }));
    EXPECT_EQ(111,  buf_y.GetFP32(3, { 0, 0, 1 }));
    EXPECT_EQ(112,  buf_y.GetFP32(3, { 1, 0, 1 }));
    EXPECT_EQ(113,  buf_y.GetFP32(3, { 2, 0, 1 }));
    EXPECT_EQ(121,  buf_y.GetFP32(3, { 0, 1, 1 }));
    EXPECT_EQ(122,  buf_y.GetFP32(3, { 1, 1, 1 }));
    EXPECT_EQ(123,  buf_y.GetFP32(3, { 2, 1, 1 }));

    EXPECT_EQ(1010, buf_y.GetFP32(6, { 0, 0, 0 }));
    EXPECT_EQ(1011, buf_y.GetFP32(6, { 1, 0, 0 }));
    EXPECT_EQ(1012, buf_y.GetFP32(6, { 2, 0, 0 }));
    EXPECT_EQ(1020, buf_y.GetFP32(6, { 0, 1, 0 }));
    EXPECT_EQ(1021, buf_y.GetFP32(6, { 1, 1, 0 }));
    EXPECT_EQ(1022, buf_y.GetFP32(6, { 2, 1, 0 }));
    EXPECT_EQ(1110, buf_y.GetFP32(6, { 0, 0, 1 }));
    EXPECT_EQ(1111, buf_y.GetFP32(6, { 1, 0, 1 }));
    EXPECT_EQ(1112, buf_y.GetFP32(6, { 2, 0, 1 }));
    EXPECT_EQ(1120, buf_y.GetFP32(6, { 0, 1, 1 }));
    EXPECT_EQ(1121, buf_y.GetFP32(6, { 1, 1, 1 }));
    EXPECT_EQ(1122, buf_y.GetFP32(6, { 2, 1, 1 }));

    EXPECT_EQ(1011, buf_y.GetFP32(7, { 0, 0, 0 }));
    EXPECT_EQ(1012, buf_y.GetFP32(7, { 1, 0, 0 }));
    EXPECT_EQ(1013, buf_y.GetFP32(7, { 2, 0, 0 }));
    EXPECT_EQ(1021, buf_y.GetFP32(7, { 0, 1, 0 }));
    EXPECT_EQ(1022, buf_y.GetFP32(7, { 1, 1, 0 }));
    EXPECT_EQ(1023, buf_y.GetFP32(7, { 2, 1, 0 }));
    EXPECT_EQ(1111, buf_y.GetFP32(7, { 0, 0, 1 }));
    EXPECT_EQ(1112, buf_y.GetFP32(7, { 1, 0, 1 }));
    EXPECT_EQ(1113, buf_y.GetFP32(7, { 2, 0, 1 }));
    EXPECT_EQ(1121, buf_y.GetFP32(7, { 0, 1, 1 }));
    EXPECT_EQ(1122, buf_y.GetFP32(7, { 1, 1, 1 }));
    EXPECT_EQ(1123, buf_y.GetFP32(7, { 2, 1, 1 }));

    // backward
    bb::FrameBuffer buf_dy(8, { 3, 2, 2 }, BB_TYPE_FP32);
    
    float dy_data[8][2][2][3];

    buf_dy = buf_y.Clone();
    for (bb::index_t f = 0; f < 8; ++f) {
        for (bb::index_t c = 0; c < 2; ++c) {
            for (bb::index_t y = 0; y < 2; ++y) {
                for (bb::index_t x = 0; x < 3; ++x) {
                    dy_data[f][c][y][x] = (float)(1000 * f + 100 * c + 10 * y + x);
                    buf_dy.SetFP32(f, { x, y, c }, dy_data[f][c][y][x]);
                }
            }
        }
    }
    
    auto buf_dx = cnvim2col->Backward(buf_dy);

    for ( bb::index_t f = 0; f < 2; ++f ) {
        for ( bb::index_t c = 0; c < 2; ++c ) {
            EXPECT_EQ(dy_data[(f*4)+0][c][0][0],
                    buf_dx.GetFP32(f, { 0, 0, c }));

            EXPECT_EQ(dy_data[(f*4)+0][c][0][1]
                    + dy_data[(f*4)+1][c][0][0],
                    buf_dx.GetFP32(f, { 1, 0, c }));

            EXPECT_EQ(dy_data[(f*4)+0][c][0][2]
                    + dy_data[(f*4)+1][c][0][1],
                    buf_dx.GetFP32(f, { 2, 0, c }));

            EXPECT_EQ(dy_data[(f*4)+1][c][0][2],
                    buf_dx.GetFP32(f, { 3, 0, c }));



            EXPECT_EQ(dy_data[(f*4)+0][c][1][0]
                    + dy_data[(f*4)+2][c][0][0],
                    buf_dx.GetFP32(f, { 0, 1, c }));

            EXPECT_EQ(dy_data[(f*4)+0][c][1][1]
                    + dy_data[(f*4)+1][c][1][0]
                    + dy_data[(f*4)+2][c][0][1]
                    + dy_data[(f*4)+3][c][0][0],
                    buf_dx.GetFP32(f, { 1, 1, c }));

            EXPECT_EQ(dy_data[(f*4)+0][c][1][2]
                    + dy_data[(f*4)+1][c][1][1]
                    + dy_data[(f*4)+2][c][0][2]
                    + dy_data[(f*4)+3][c][0][1],
                    buf_dx.GetFP32(f, { 2, 1, c }));
            
            EXPECT_EQ(dy_data[(f*4)+1][c][1][2]
                    + dy_data[(f*4)+3][c][0][2],
                    buf_dx.GetFP32(f, { 3, 1, c }));



            EXPECT_EQ(dy_data[(f*4)+2][c][1][0],
                    buf_dx.GetFP32(f, { 0, 2, c }));

            EXPECT_EQ(dy_data[(f*4)+2][c][1][1]
                    + dy_data[(f*4)+3][c][1][0],
                    buf_dx.GetFP32(f, { 1, 2, c }));

            EXPECT_EQ(dy_data[(f*4)+2][c][1][2]
                    + dy_data[(f*4)+3][c][1][1],
                    buf_dx.GetFP32(f, { 2, 2, c }));

            EXPECT_EQ(dy_data[(f*4)+3][c][1][2],
                    buf_dx.GetFP32(f, { 3, 2, c }));
        }
    }
}



TEST(ConvolutionIm2ColTest, testConvolutionIm2Col_float)
{
    auto cnvim2col = bb::ConvolutionIm2Col<float>::Create(2, 2);
    
    bb::FrameBuffer buf_x(1, {3, 3, 1}, BB_TYPE_FP32);

    // 0 1 1
    // 1 0 1
    // 0 1 0

    buf_x.SetFP32(0, {0, 0, 0}, 0);
    buf_x.SetFP32(0, {1, 0, 0}, 1);
    buf_x.SetFP32(0, {2, 0, 0}, 1);
    buf_x.SetFP32(0, {0, 1, 0}, 1);
    buf_x.SetFP32(0, {1, 1, 0}, 0);
    buf_x.SetFP32(0, {2, 1, 0}, 1);
    buf_x.SetFP32(0, {0, 2, 0}, 0);
    buf_x.SetFP32(0, {1, 2, 0}, 1);
    buf_x.SetFP32(0, {2, 2, 0}, 0);

    auto buf_y = cnvim2col->Forward(buf_x);

    EXPECT_EQ(0, buf_y.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(1, buf_y.GetFP32(0, { 1, 0, 0 }));
    EXPECT_EQ(1, buf_y.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(0, buf_y.GetFP32(0, { 1, 1, 0 }));
    EXPECT_EQ(1, buf_y.GetFP32(1, { 0, 0, 0 }));
    EXPECT_EQ(1, buf_y.GetFP32(1, { 1, 0, 0 }));
    EXPECT_EQ(0, buf_y.GetFP32(1, { 0, 1, 0 }));
    EXPECT_EQ(1, buf_y.GetFP32(1, { 1, 1, 0 }));
    EXPECT_EQ(1, buf_y.GetFP32(2, { 0, 0, 0 }));
    EXPECT_EQ(0, buf_y.GetFP32(2, { 1, 0, 0 }));
    EXPECT_EQ(0, buf_y.GetFP32(2, { 0, 1, 0 }));
    EXPECT_EQ(1, buf_y.GetFP32(2, { 1, 1, 0 }));
    EXPECT_EQ(0, buf_y.GetFP32(3, { 0, 0, 0 }));
    EXPECT_EQ(1, buf_y.GetFP32(3, { 1, 0, 0 }));
    EXPECT_EQ(1, buf_y.GetFP32(3, { 0, 1, 0 }));
    EXPECT_EQ(0, buf_y.GetFP32(3, { 1, 1, 0 }));
}


TEST(ConvolutionIm2ColTest, testConvolutionIm2Col_bit)
{
    auto cnvim2col = bb::ConvolutionIm2Col<bb::Bit>::Create(2, 2);
    
    bb::FrameBuffer buf_x(2, {3, 3, 2}, BB_TYPE_BIT);

    // 0 1 1
    // 1 0 1
    // 0 1 0

    buf_x.SetBit(0, {0, 0, 0}, 0);
    buf_x.SetBit(0, {1, 0, 0}, 1);
    buf_x.SetBit(0, {2, 0, 0}, 1);
    buf_x.SetBit(0, {0, 1, 0}, 1);
    buf_x.SetBit(0, {1, 1, 0}, 0);
    buf_x.SetBit(0, {2, 1, 0}, 1);
    buf_x.SetBit(0, {0, 2, 0}, 0);
    buf_x.SetBit(0, {1, 2, 0}, 1);
    buf_x.SetBit(0, {2, 2, 0}, 0);
    
    // 1 1 0
    // 1 0 0
    // 1 1 0
    buf_x.SetBit(0, {0, 0, 1}, 1);
    buf_x.SetBit(0, {1, 0, 1}, 1);
    buf_x.SetBit(0, {2, 0, 1}, 0);
    buf_x.SetBit(0, {0, 1, 1}, 1);
    buf_x.SetBit(0, {1, 1, 1}, 0);
    buf_x.SetBit(0, {2, 1, 1}, 0);
    buf_x.SetBit(0, {0, 2, 1}, 1);
    buf_x.SetBit(0, {1, 2, 1}, 1);
    buf_x.SetBit(0, {2, 2, 1}, 0);


    // 1 0 1
    // 1 1 0
    // 0 0 1

    buf_x.SetBit(1, {0, 0, 0}, 1);
    buf_x.SetBit(1, {1, 0, 0}, 0);
    buf_x.SetBit(1, {2, 0, 0}, 1);
    buf_x.SetBit(1, {0, 1, 0}, 1);
    buf_x.SetBit(1, {1, 1, 0}, 1);
    buf_x.SetBit(1, {2, 1, 0}, 0);
    buf_x.SetBit(1, {0, 2, 0}, 0);
    buf_x.SetBit(1, {1, 2, 0}, 0);
    buf_x.SetBit(1, {2, 2, 0}, 1);
    
    // 1 1 0
    // 0 1 1
    // 0 1 0
    buf_x.SetBit(1, {0, 0, 1}, 1);
    buf_x.SetBit(1, {1, 0, 1}, 1);
    buf_x.SetBit(1, {2, 0, 1}, 0);
    buf_x.SetBit(1, {0, 1, 1}, 0);
    buf_x.SetBit(1, {1, 1, 1}, 1);
    buf_x.SetBit(1, {2, 1, 1}, 1);
    buf_x.SetBit(1, {0, 2, 1}, 0);
    buf_x.SetBit(1, {1, 2, 1}, 1);
    buf_x.SetBit(1, {2, 2, 1}, 0);

    auto buf_y = cnvim2col->Forward(buf_x);

    // 0 1 1
    // 1 0 1
    // 0 1 0
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(0, { 0, 0, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(0, { 1, 0, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(0, { 0, 1, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(0, { 1, 1, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(1, { 0, 0, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(1, { 1, 0, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(1, { 0, 1, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(1, { 1, 1, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(2, { 0, 0, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(2, { 1, 0, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(2, { 0, 1, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(2, { 1, 1, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(3, { 0, 0, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(3, { 1, 0, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(3, { 0, 1, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(3, { 1, 1, 0 }));

    // 1 1 0
    // 1 0 0
    // 1 1 0
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(0, { 0, 0, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(0, { 1, 0, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(0, { 0, 1, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(0, { 1, 1, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(1, { 0, 0, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(1, { 1, 0, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(1, { 0, 1, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(1, { 1, 1, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(2, { 0, 0, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(2, { 1, 0, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(2, { 0, 1, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(2, { 1, 1, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(3, { 0, 0, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(3, { 1, 0, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(3, { 0, 1, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(3, { 1, 1, 1 }));

    // 1 0 1
    // 1 1 0
    // 0 0 1
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(4, { 0, 0, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(4, { 1, 0, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(4, { 0, 1, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(4, { 1, 1, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(5, { 0, 0, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(5, { 1, 0, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(5, { 0, 1, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(5, { 1, 1, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(6, { 0, 0, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(6, { 1, 0, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(6, { 0, 1, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(6, { 1, 1, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(7, { 0, 0, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(7, { 1, 0, 0 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(7, { 0, 1, 0 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(7, { 1, 1, 0 }));

    // 1 1 0
    // 0 1 1
    // 0 1 0
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(4, { 0, 0, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(4, { 1, 0, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(4, { 0, 1, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(4, { 1, 1, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(5, { 0, 0, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(5, { 1, 0, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(5, { 0, 1, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(5, { 1, 1, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(6, { 0, 0, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(6, { 1, 0, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(6, { 0, 1, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(6, { 1, 1, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(7, { 0, 0, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(7, { 1, 0, 1 }));
    EXPECT_EQ((bb::Bit)1, buf_y.GetBit(7, { 0, 1, 1 }));
    EXPECT_EQ((bb::Bit)0, buf_y.GetBit(7, { 1, 1, 1 }));
}





TEST(ConvolutionIm2ColTest, testConvolutionIm2Col_stride)
{
    bb::index_t const input_frame_size = 2;  
    bb::index_t const input_w_size = 7;
    bb::index_t const input_h_size = 5; 
    bb::index_t const input_c_size = 2;  

    auto cnvim2col = bb::ConvolutionIm2Col<>::Create(3, 3, 2, 2);
    
    bb::FrameBuffer x_buf(input_frame_size, {input_w_size, input_h_size, input_c_size}, BB_TYPE_FP32);
    cnvim2col->SetInputShape(x_buf.GetShape());

    for ( bb::index_t f = 0; f < input_frame_size; ++f) {
        for (bb::index_t c = 0; c < input_c_size; ++c) {
            for (bb::index_t y = 0; y < input_h_size; ++y) {
                for (bb::index_t x = 0; x < input_w_size; ++x) {
                    x_buf.SetFP32(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
                }
            }
        }
    }

    auto y_buf = cnvim2col->Forward(x_buf);
    EXPECT_EQ(2 * 3 * 2, y_buf.GetFrameSize());

    auto y_shape = y_buf.GetShape();
    EXPECT_EQ(3, y_shape.size());
    EXPECT_EQ(3, y_shape[0]);
    EXPECT_EQ(3, y_shape[1]);
    EXPECT_EQ(2, y_shape[2]);

    EXPECT_EQ( 0, y_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ( 1, y_buf.GetFP32(0, { 1, 0, 0 }));
    EXPECT_EQ( 2, y_buf.GetFP32(0, { 2, 0, 0 }));
    EXPECT_EQ(10, y_buf.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(11, y_buf.GetFP32(0, { 1, 1, 0 }));
    EXPECT_EQ(12, y_buf.GetFP32(0, { 2, 1, 0 }));
    EXPECT_EQ(20, y_buf.GetFP32(0, { 0, 2, 0 }));
    EXPECT_EQ(21, y_buf.GetFP32(0, { 1, 2, 0 }));
    EXPECT_EQ(22, y_buf.GetFP32(0, { 2, 2, 0 }));

    EXPECT_EQ( 2, y_buf.GetFP32(1, { 0, 0, 0 }));
    EXPECT_EQ( 3, y_buf.GetFP32(1, { 1, 0, 0 }));
    EXPECT_EQ( 4, y_buf.GetFP32(1, { 2, 0, 0 }));
    EXPECT_EQ(12, y_buf.GetFP32(1, { 0, 1, 0 }));
    EXPECT_EQ(13, y_buf.GetFP32(1, { 1, 1, 0 }));
    EXPECT_EQ(14, y_buf.GetFP32(1, { 2, 1, 0 }));
    EXPECT_EQ(22, y_buf.GetFP32(1, { 0, 2, 0 }));
    EXPECT_EQ(23, y_buf.GetFP32(1, { 1, 2, 0 }));
    EXPECT_EQ(24, y_buf.GetFP32(1, { 2, 2, 0 }));

    EXPECT_EQ( 4, y_buf.GetFP32(2, { 0, 0, 0 }));
    EXPECT_EQ( 5, y_buf.GetFP32(2, { 1, 0, 0 }));
    EXPECT_EQ( 6, y_buf.GetFP32(2, { 2, 0, 0 }));
    EXPECT_EQ(14, y_buf.GetFP32(2, { 0, 1, 0 }));
    EXPECT_EQ(15, y_buf.GetFP32(2, { 1, 1, 0 }));
    EXPECT_EQ(16, y_buf.GetFP32(2, { 2, 1, 0 }));
    EXPECT_EQ(24, y_buf.GetFP32(2, { 0, 2, 0 }));
    EXPECT_EQ(25, y_buf.GetFP32(2, { 1, 2, 0 }));
    EXPECT_EQ(26, y_buf.GetFP32(2, { 2, 2, 0 }));

    EXPECT_EQ(122, y_buf.GetFP32(4, { 0, 0, 1 }));
    EXPECT_EQ(123, y_buf.GetFP32(4, { 1, 0, 1 }));
    EXPECT_EQ(124, y_buf.GetFP32(4, { 2, 0, 1 }));
    EXPECT_EQ(132, y_buf.GetFP32(4, { 0, 1, 1 }));
    EXPECT_EQ(133, y_buf.GetFP32(4, { 1, 1, 1 }));
    EXPECT_EQ(134, y_buf.GetFP32(4, { 2, 1, 1 }));
    EXPECT_EQ(142, y_buf.GetFP32(4, { 0, 2, 1 }));
    EXPECT_EQ(143, y_buf.GetFP32(4, { 1, 2, 1 }));
    EXPECT_EQ(144, y_buf.GetFP32(4, { 2, 2, 1 }));

    EXPECT_EQ(24, y_buf.GetFP32(5, { 0, 0, 0 }));
    EXPECT_EQ(25, y_buf.GetFP32(5, { 1, 0, 0 }));
    EXPECT_EQ(26, y_buf.GetFP32(5, { 2, 0, 0 }));
    EXPECT_EQ(34, y_buf.GetFP32(5, { 0, 1, 0 }));
    EXPECT_EQ(35, y_buf.GetFP32(5, { 1, 1, 0 }));
    EXPECT_EQ(36, y_buf.GetFP32(5, { 2, 1, 0 }));
    EXPECT_EQ(44, y_buf.GetFP32(5, { 0, 2, 0 }));
    EXPECT_EQ(45, y_buf.GetFP32(5, { 1, 2, 0 }));
    EXPECT_EQ(46, y_buf.GetFP32(5, { 2, 2, 0 }));

    EXPECT_EQ(1024, y_buf.GetFP32(11, { 0, 0, 0 }));
    EXPECT_EQ(1025, y_buf.GetFP32(11, { 1, 0, 0 }));
    EXPECT_EQ(1026, y_buf.GetFP32(11, { 2, 0, 0 }));
    EXPECT_EQ(1034, y_buf.GetFP32(11, { 0, 1, 0 }));
    EXPECT_EQ(1035, y_buf.GetFP32(11, { 1, 1, 0 }));
    EXPECT_EQ(1036, y_buf.GetFP32(11, { 2, 1, 0 }));
    EXPECT_EQ(1044, y_buf.GetFP32(11, { 0, 2, 0 }));
    EXPECT_EQ(1045, y_buf.GetFP32(11, { 1, 2, 0 }));
    EXPECT_EQ(1046, y_buf.GetFP32(11, { 2, 2, 0 }));

    EXPECT_EQ(1124, y_buf.GetFP32(11, { 0, 0, 1 }));
    EXPECT_EQ(1125, y_buf.GetFP32(11, { 1, 0, 1 }));
    EXPECT_EQ(1126, y_buf.GetFP32(11, { 2, 0, 1 }));
    EXPECT_EQ(1134, y_buf.GetFP32(11, { 0, 1, 1 }));
    EXPECT_EQ(1135, y_buf.GetFP32(11, { 1, 1, 1 }));
    EXPECT_EQ(1136, y_buf.GetFP32(11, { 2, 1, 1 }));
    EXPECT_EQ(1144, y_buf.GetFP32(11, { 0, 2, 1 }));
    EXPECT_EQ(1145, y_buf.GetFP32(11, { 1, 2, 1 }));
    EXPECT_EQ(1146, y_buf.GetFP32(11, { 2, 2, 1 }));


    // backward
    bb::FrameBuffer dy_buf(12, { 3, 3, 2 }, BB_TYPE_FP32);
    
    float dy_data[12][2][3][3];

    dy_buf = dy_buf.Clone();
    for (bb::index_t f = 0; f < 12; ++f) {
        for (bb::index_t c = 0; c < 2; ++c) {
            for (bb::index_t y = 0; y < 3; ++y) {
                for (bb::index_t x = 0; x < 3; ++x) {
                    dy_data[f][c][y][x] = (float)(1000 * f + 100 * c + 10 * y + x);
                    dy_buf.SetFP32(f, { x, y, c }, dy_data[f][c][y][x]);
                }
            }
        }
    }
    
    auto dx_buf = cnvim2col->Backward(dy_buf);
    
    for ( bb::index_t f = 0; f < 2; ++f ) {
        for ( bb::index_t c = 0; c < 2; ++c ) {
            EXPECT_EQ(dy_data[(f*6)+0][c][0][0],
                    dx_buf.GetFP32(f, { 0, 0, c }));

            EXPECT_EQ(dy_data[(f*6)+0][c][0][1],
                    dx_buf.GetFP32(f, { 1, 0, c }));

            EXPECT_EQ(dy_data[(f*6)+0][c][0][2]
                    + dy_data[(f*6)+1][c][0][0],
                    dx_buf.GetFP32(f, { 2, 0, c }));

            EXPECT_EQ(dy_data[(f*6)+1][c][0][1],
                    dx_buf.GetFP32(f, { 3, 0, c }));

            EXPECT_EQ(dy_data[(f*6)+1][c][0][2]
                    + dy_data[(f*6)+2][c][0][0],
                    dx_buf.GetFP32(f, { 4, 0, c }));

            EXPECT_EQ(dy_data[(f*6)+2][c][0][1],
                    dx_buf.GetFP32(f, { 5, 0, c }));

            EXPECT_EQ(dy_data[(f*6)+2][c][0][2],
                    dx_buf.GetFP32(f, { 6, 0, c }));


            EXPECT_EQ(dy_data[(f*6)+0][c][1][0 ],
                    dx_buf.GetFP32(f, { 0, 1, c }));

            EXPECT_EQ(dy_data[(f*6)+0][c][1][1],
                    dx_buf.GetFP32(f, { 1, 1, c }));

            EXPECT_EQ(dy_data[(f*6)+0][c][1][2]
                    + dy_data[(f*6)+1][c][1][0],
                    dx_buf.GetFP32(f, { 2, 1, c }));

            EXPECT_EQ(dy_data[(f*6)+1][c][1][1],
                    dx_buf.GetFP32(f, { 3, 1, c }));

            EXPECT_EQ(dy_data[(f*6)+1][c][1][2]
                    + dy_data[(f*6)+2][c][1][0],
                    dx_buf.GetFP32(f, { 4, 1, c }));

            EXPECT_EQ(dy_data[(f*6)+2][c][1][1],
                    dx_buf.GetFP32(f, { 5, 1, c }));

            EXPECT_EQ(dy_data[(f*6)+2][c][1][2],
                    dx_buf.GetFP32(f, { 6, 1, c }));


            EXPECT_EQ(dy_data[(f*6)+0][c][2][0]
                    + dy_data[(f*6)+3][c][0][0],
                    dx_buf.GetFP32(f, { 0, 2, c }));

            EXPECT_EQ(dy_data[(f*6)+0][c][2][1]
                    + dy_data[(f*6)+3][c][0][1],
                    dx_buf.GetFP32(f, { 1, 2, c }));

            EXPECT_EQ(dy_data[(f*6)+0][c][2][2]
                    + dy_data[(f*6)+1][c][2][0]
                    + dy_data[(f*6)+3][c][0][2]
                    + dy_data[(f*6)+4][c][0][0],
                    dx_buf.GetFP32(f, { 2, 2, c }));

            EXPECT_EQ(dy_data[(f*6)+1][c][2][1]
                    + dy_data[(f*6)+4][c][0][1],
                    dx_buf.GetFP32(f, { 3, 2, c }));

            EXPECT_EQ(dy_data[(f*6)+1][c][2][2]
                    + dy_data[(f*6)+2][c][2][0]
                    + dy_data[(f*6)+4][c][0][2]
                    + dy_data[(f*6)+5][c][0][0],
                    dx_buf.GetFP32(f, { 4, 2, c }));

            EXPECT_EQ(dy_data[(f*6)+4][c][1][2]
                    + dy_data[(f*6)+5][c][1][0],
                    dx_buf.GetFP32(f, { 4, 3, c }));

            EXPECT_EQ(dy_data[(f*6)+5][c][1][1],
                    dx_buf.GetFP32(f, { 5, 3, c }));

            EXPECT_EQ(dy_data[(f*6)+5][c][2][2],
                    dx_buf.GetFP32(f, { 6, 4, c }));

        }
    }
}




TEST(ConvolutionIm2ColTest, testConvolutionIm2Col_same)
{
    bb::index_t const filter_h_size = 2;
    bb::index_t const filter_w_size = 3;

    bb::index_t const input_frame_size = 2;  
    bb::index_t const input_w_size     = 3;
    bb::index_t const input_h_size     = 2; 
    bb::index_t const input_c_size     = 2;

    bb::index_t const output_frame_size = input_frame_size * input_h_size * input_w_size;
    bb::index_t const output_h_size     = filter_h_size; 
    bb::index_t const output_w_size     = filter_w_size; 
    bb::index_t const output_c_size     = input_c_size;  

    bb::indices_t     input_shape({input_w_size, input_h_size, input_c_size});
    bb::indices_t     output_shape({output_w_size, output_h_size, output_c_size});

    auto cnvim2col = bb::ConvolutionIm2Col<>::Create(filter_h_size, filter_w_size, 1, 1, "same", bb::ConvolutionIm2Col<>::BORDER_CONSTANT);
    
    bb::FrameBuffer x_buf(input_frame_size, input_shape, BB_TYPE_FP32);
    cnvim2col->SetInputShape(x_buf.GetShape());

    for ( bb::index_t f = 0; f < input_frame_size; ++f) {
        for (bb::index_t c = 0; c < input_c_size; ++c) {
            for (bb::index_t y = 0; y < input_h_size; ++y) {
                for (bb::index_t x = 0; x < input_w_size; ++x) {
                    x_buf.SetFP32(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x + 10000));
                }
            }
        }
    }

    auto y_buf = cnvim2col->Forward(x_buf);

    EXPECT_EQ(output_frame_size, y_buf.GetFrameSize());
    EXPECT_EQ(output_shape,      y_buf.GetShape());

    EXPECT_EQ(    0, y_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(10000, y_buf.GetFP32(0, { 1, 0, 0 }));
    EXPECT_EQ(10001, y_buf.GetFP32(0, { 2, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(10010, y_buf.GetFP32(0, { 1, 1, 0 }));
    EXPECT_EQ(10011, y_buf.GetFP32(0, { 2, 1, 0 }));

    EXPECT_EQ(10000, y_buf.GetFP32(1, { 0, 0, 0 }));
    EXPECT_EQ(10001, y_buf.GetFP32(1, { 1, 0, 0 }));
    EXPECT_EQ(10002, y_buf.GetFP32(1, { 2, 0, 0 }));
    EXPECT_EQ(10010, y_buf.GetFP32(1, { 0, 1, 0 }));
    EXPECT_EQ(10011, y_buf.GetFP32(1, { 1, 1, 0 }));
    EXPECT_EQ(10012, y_buf.GetFP32(1, { 2, 1, 0 }));

    EXPECT_EQ(10001, y_buf.GetFP32(2, { 0, 0, 0 }));
    EXPECT_EQ(10002, y_buf.GetFP32(2, { 1, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(2, { 2, 0, 0 }));
    EXPECT_EQ(10011, y_buf.GetFP32(2, { 0, 1, 0 }));
    EXPECT_EQ(10012, y_buf.GetFP32(2, { 1, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(2, { 2, 1, 0 }));

    EXPECT_EQ(    0, y_buf.GetFP32(3, { 0, 0, 0 }));
    EXPECT_EQ(10010, y_buf.GetFP32(3, { 1, 0, 0 }));
    EXPECT_EQ(10011, y_buf.GetFP32(3, { 2, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(3, { 0, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(3, { 1, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(3, { 2, 1, 0 }));

    EXPECT_EQ(10010, y_buf.GetFP32(4, { 0, 0, 0 }));
    EXPECT_EQ(10011, y_buf.GetFP32(4, { 1, 0, 0 }));
    EXPECT_EQ(10012, y_buf.GetFP32(4, { 2, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(4, { 0, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(4, { 1, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(4, { 2, 1, 0 }));

    EXPECT_EQ(10011, y_buf.GetFP32(5, { 0, 0, 0 }));
    EXPECT_EQ(10012, y_buf.GetFP32(5, { 1, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(5, { 2, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(5, { 0, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(5, { 1, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(5, { 2, 1, 0 }));


    EXPECT_EQ(    0, y_buf.GetFP32(6, { 0, 0, 0 }));
    EXPECT_EQ(11000, y_buf.GetFP32(6, { 1, 0, 0 }));
    EXPECT_EQ(11001, y_buf.GetFP32(6, { 2, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(6, { 0, 1, 0 }));
    EXPECT_EQ(11010, y_buf.GetFP32(6, { 1, 1, 0 }));
    EXPECT_EQ(11011, y_buf.GetFP32(6, { 2, 1, 0 }));
    
    EXPECT_EQ(11000, y_buf.GetFP32(7, { 0, 0, 0 }));
    EXPECT_EQ(11001, y_buf.GetFP32(7, { 1, 0, 0 }));
    EXPECT_EQ(11002, y_buf.GetFP32(7, { 2, 0, 0 }));
    EXPECT_EQ(11010, y_buf.GetFP32(7, { 0, 1, 0 }));
    EXPECT_EQ(11011, y_buf.GetFP32(7, { 1, 1, 0 }));
    EXPECT_EQ(11012, y_buf.GetFP32(7, { 2, 1, 0 }));
    
    EXPECT_EQ(11001, y_buf.GetFP32(8, { 0, 0, 0 }));
    EXPECT_EQ(11002, y_buf.GetFP32(8, { 1, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(8, { 2, 0, 0 }));
    EXPECT_EQ(11011, y_buf.GetFP32(8, { 0, 1, 0 }));
    EXPECT_EQ(11012, y_buf.GetFP32(8, { 1, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(8, { 2, 1, 0 }));
    
    EXPECT_EQ(    0, y_buf.GetFP32(9, { 0, 0, 0 }));
    EXPECT_EQ(11010, y_buf.GetFP32(9, { 1, 0, 0 }));
    EXPECT_EQ(11011, y_buf.GetFP32(9, { 2, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(9, { 0, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(9, { 1, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(9, { 2, 1, 0 }));
    
    EXPECT_EQ(11010, y_buf.GetFP32(10, { 0, 0, 0 }));
    EXPECT_EQ(11011, y_buf.GetFP32(10, { 1, 0, 0 }));
    EXPECT_EQ(11012, y_buf.GetFP32(10, { 2, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(10, { 0, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(10, { 1, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(10, { 2, 1, 0 }));
    
    EXPECT_EQ(11011, y_buf.GetFP32(11, { 0, 0, 0 }));
    EXPECT_EQ(11012, y_buf.GetFP32(11, { 1, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(11, { 2, 0, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(11, { 0, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(11, { 1, 1, 0 }));
    EXPECT_EQ(    0, y_buf.GetFP32(11, { 2, 1, 0 }));

    EXPECT_EQ(    0, y_buf.GetFP32(6, { 0, 0, 1 }));
    EXPECT_EQ(11100, y_buf.GetFP32(6, { 1, 0, 1 }));
    EXPECT_EQ(11101, y_buf.GetFP32(6, { 2, 0, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(6, { 0, 1, 1 }));
    EXPECT_EQ(11110, y_buf.GetFP32(6, { 1, 1, 1 }));
    EXPECT_EQ(11111, y_buf.GetFP32(6, { 2, 1, 1 }));
    
    EXPECT_EQ(11100, y_buf.GetFP32(7, { 0, 0, 1 }));
    EXPECT_EQ(11101, y_buf.GetFP32(7, { 1, 0, 1 }));
    EXPECT_EQ(11102, y_buf.GetFP32(7, { 2, 0, 1 }));
    EXPECT_EQ(11110, y_buf.GetFP32(7, { 0, 1, 1 }));
    EXPECT_EQ(11111, y_buf.GetFP32(7, { 1, 1, 1 }));
    EXPECT_EQ(11112, y_buf.GetFP32(7, { 2, 1, 1 }));
    
    EXPECT_EQ(11101, y_buf.GetFP32(8, { 0, 0, 1 }));
    EXPECT_EQ(11102, y_buf.GetFP32(8, { 1, 0, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(8, { 2, 0, 1 }));
    EXPECT_EQ(11111, y_buf.GetFP32(8, { 0, 1, 1 }));
    EXPECT_EQ(11112, y_buf.GetFP32(8, { 1, 1, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(8, { 2, 1, 1 }));
    
    EXPECT_EQ(    0, y_buf.GetFP32(9, { 0, 0, 1 }));
    EXPECT_EQ(11110, y_buf.GetFP32(9, { 1, 0, 1 }));
    EXPECT_EQ(11111, y_buf.GetFP32(9, { 2, 0, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(9, { 0, 1, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(9, { 1, 1, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(9, { 2, 1, 1 }));
    
    EXPECT_EQ(11110, y_buf.GetFP32(10, { 0, 0, 1 }));
    EXPECT_EQ(11111, y_buf.GetFP32(10, { 1, 0, 1 }));
    EXPECT_EQ(11112, y_buf.GetFP32(10, { 2, 0, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(10, { 0, 1, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(10, { 1, 1, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(10, { 2, 1, 1 }));
    
    EXPECT_EQ(11111, y_buf.GetFP32(11, { 0, 0, 1 }));
    EXPECT_EQ(11112, y_buf.GetFP32(11, { 1, 0, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(11, { 2, 0, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(11, { 0, 1, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(11, { 1, 1, 1 }));
    EXPECT_EQ(    0, y_buf.GetFP32(11, { 2, 1, 1 }));

    // backward
    bb::FrameBuffer dy_buf(output_frame_size, output_shape, BB_TYPE_FP32);
    
    float dy_data[output_frame_size][output_c_size][output_h_size][output_w_size];

    dy_buf = dy_buf.Clone();
    for (bb::index_t f = 0; f < output_frame_size; ++f) {
        for (bb::index_t c = 0; c < output_c_size; ++c) {
            for (bb::index_t y = 0; y < output_h_size; ++y) {
                for (bb::index_t x = 0; x < output_w_size; ++x) {
                    dy_data[f][c][y][x] = (float)(1000 * f + 100 * c + 10 * y + x);
                    dy_buf.SetFP32(f, { x, y, c }, dy_data[f][c][y][x]);
                }
            }
        }
    }
    
    auto dx_buf = cnvim2col->Backward(dy_buf);
   
    EXPECT_EQ(input_frame_size, dx_buf.GetFrameSize());
    EXPECT_EQ(input_shape,      dx_buf.GetShape());

    for ( bb::index_t f = 0; f < 2; ++f ) {
        for ( bb::index_t c = 0; c < 2; ++c ) {
            EXPECT_EQ(dy_data[(f*6)+0][c][0][1]
                    + dy_data[(f*6)+1][c][0][0],
                    dx_buf.GetFP32(f, { 0, 0, c }));

            EXPECT_EQ(dy_data[(f*6)+0][c][0][2]
                    + dy_data[(f*6)+1][c][0][1]
                    + dy_data[(f*6)+2][c][0][0],
                    dx_buf.GetFP32(f, { 1, 0, c }));

            EXPECT_EQ(dy_data[(f*6)+1][c][0][2]
                    + dy_data[(f*6)+2][c][0][1],
                    dx_buf.GetFP32(f, { 2, 0, c }));

            EXPECT_EQ(dy_data[(f*6)+0][c][1][1]
                    + dy_data[(f*6)+1][c][1][0]
                    + dy_data[(f*6)+3][c][0][1]
                    + dy_data[(f*6)+4][c][0][0],
                    dx_buf.GetFP32(f, { 0, 1, c }));

            EXPECT_EQ(dy_data[(f*6)+0][c][1][2]
                    + dy_data[(f*6)+1][c][1][1]
                    + dy_data[(f*6)+2][c][1][0]
                    + dy_data[(f*6)+3][c][0][2]
                    + dy_data[(f*6)+4][c][0][1]
                    + dy_data[(f*6)+5][c][0][0],
                    dx_buf.GetFP32(f, { 1, 1, c }));

            EXPECT_EQ(dy_data[(f*6)+1][c][1][2]
                    + dy_data[(f*6)+2][c][1][1]
                    + dy_data[(f*6)+4][c][0][2]
                    + dy_data[(f*6)+5][c][0][1],
                    dx_buf.GetFP32(f, { 2, 1, c }));
        }
    }
}

