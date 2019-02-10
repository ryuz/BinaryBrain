#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/ConvolutionIm2Col.h"
// #include "bb/ConvolutionCol2Im.h"


TEST(ConvolutionIm2ColTest, testConvolutionIm2Col_xy)
{
	bb::ConvolutionIm2Col<> cnvim2col(2, 3);

    bb::FrameBuffer buf_x(16, {28, 28, 3}, BB_TYPE_FP32);
    bb::FrameBuffer buf_y = cnvim2col.Forward(buf_x);

}


TEST(ConvolutionIm2ColTest, testConvolutionIm2Col)
{
	bb::ConvolutionIm2Col<> cnvim2col(2, 3);
	
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

	auto buf_y = cnvim2col.Forward(buf_x);

//	buf_y.Reshape({ 3, 2, 2 });
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
	
	auto buf_dx = cnvim2col.Backward(buf_dy);

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


