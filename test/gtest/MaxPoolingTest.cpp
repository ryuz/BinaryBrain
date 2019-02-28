#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/MaxPooling.h"


TEST(MaxPoolingTest, testMaxPoolingTest)
{
    bb::index_t const frame_size = 2;
    bb::index_t const c_size = 3;
    bb::index_t const input_h_size  = 4;
    bb::index_t const input_w_size  = 6;
    bb::index_t const filter_h_size = 2;
    bb::index_t const filter_w_size = 3;
    bb::index_t const output_h_size = input_h_size / filter_h_size;
    bb::index_t const output_w_size = input_w_size / filter_w_size;

	auto maxpol = bb::MaxPooling<>::Create(filter_h_size, filter_w_size);

    bb::FrameBuffer    x_buf(BB_TYPE_FP32, frame_size, {input_w_size, input_h_size, c_size});

	for (bb::index_t f = 0; f < frame_size; ++f) {
		for (bb::index_t c = 0; c < c_size; ++c) {
			for (bb::index_t y = 0; y < input_h_size; ++y) {
				for (bb::index_t x = 0; x < input_w_size; ++x) {
					x_buf.SetFP32(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
				}
			}
		}
	}
	x_buf.SetFP32(0, { 1, 0, 0 }, 99);

	auto y_buf = maxpol->Forward(x_buf);
	
    EXPECT_EQ(bb::indices_t({2, 2, 3}), y_buf.GetShape());

	EXPECT_EQ(99, y_buf.GetFP32(0, { 0, 0, 0 }));
	EXPECT_EQ(15, y_buf.GetFP32(0, { 1, 0, 0 }));
	EXPECT_EQ(32, y_buf.GetFP32(0, { 0, 1, 0 }));
	EXPECT_EQ(35, y_buf.GetFP32(0, { 1, 1, 0 }));

	EXPECT_EQ(112, y_buf.GetFP32(0, { 0, 0, 1 }));
	EXPECT_EQ(115, y_buf.GetFP32(0, { 1, 0, 1 }));
	EXPECT_EQ(132, y_buf.GetFP32(0, { 0, 1, 1 }));
	EXPECT_EQ(135, y_buf.GetFP32(0, { 1, 1, 1 }));

	EXPECT_EQ(212, y_buf.GetFP32(0, { 0, 0, 2 }));
	EXPECT_EQ(215, y_buf.GetFP32(0, { 1, 0, 2 }));
	EXPECT_EQ(232, y_buf.GetFP32(0, { 0, 1, 2 }));
	EXPECT_EQ(235, y_buf.GetFP32(0, { 1, 1, 2 }));

	EXPECT_EQ(1012, y_buf.GetFP32(1, { 0, 0, 0 }));
	EXPECT_EQ(1015, y_buf.GetFP32(1, { 1, 0, 0 }));
	EXPECT_EQ(1032, y_buf.GetFP32(1, { 0, 1, 0 }));
	EXPECT_EQ(1035, y_buf.GetFP32(1, { 1, 1, 0 }));

	EXPECT_EQ(1112, y_buf.GetFP32(1, { 0, 0, 1 }));
	EXPECT_EQ(1115, y_buf.GetFP32(1, { 1, 0, 1 }));
	EXPECT_EQ(1132, y_buf.GetFP32(1, { 0, 1, 1 }));
	EXPECT_EQ(1135, y_buf.GetFP32(1, { 1, 1, 1 }));

	EXPECT_EQ(1212, y_buf.GetFP32(1, { 0, 0, 2 }));
	EXPECT_EQ(1215, y_buf.GetFP32(1, { 1, 0, 2 }));
	EXPECT_EQ(1232, y_buf.GetFP32(1, { 0, 1, 2 }));
	EXPECT_EQ(1235, y_buf.GetFP32(1, { 1, 1, 2 }));

	// backward

    bb::FrameBuffer dy_buf(BB_TYPE_FP32, frame_size, {output_w_size, output_h_size, c_size});

	for (bb::index_t f = 0; f < 2; ++f) {
		for (bb::index_t c = 0; c < 3; ++c) {
			for (bb::index_t y = 0; y < 2; ++y) {
				for (bb::index_t x = 0; x < 2; ++x) {
					dy_buf.SetFP32(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x + 1));
				}
			}
		}
	}

	auto dx_buf = maxpol->Backward(dy_buf);

  	EXPECT_EQ(bb::indices_t({input_w_size, input_h_size, c_size}), dx_buf.GetShape());

	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 0, 0, 0 }));
	EXPECT_EQ(1,  dx_buf.GetFP32(0, { 1, 0, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 2, 0, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 0, 1, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 1, 1, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 2, 1, 0 }));
                  
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 3, 0, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 4, 0, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 5, 0, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 3, 1, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 4, 1, 0 }));
	EXPECT_EQ(2,  dx_buf.GetFP32(0, { 5, 1, 0 }));

	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 0, 2, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 1, 2, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 2, 2, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 0, 3, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 1, 3, 0 }));
	EXPECT_EQ(11, dx_buf.GetFP32(0, { 2, 3, 0 }));

	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 3, 2, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 4, 2, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 5, 2, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 3, 3, 0 }));
	EXPECT_EQ(0,  dx_buf.GetFP32(0, { 4, 3, 0 }));
	EXPECT_EQ(12, dx_buf.GetFP32(0, { 5, 3, 0 }));

	//
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 0, 0, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 1, 0, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 2, 0, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 0, 1, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 1, 1, 1 }));
	EXPECT_EQ(101, dx_buf.GetFP32(0, { 2, 1, 1 }));

	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 3, 0, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 4, 0, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 5, 0, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 3, 1, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 4, 1, 1 }));
	EXPECT_EQ(102, dx_buf.GetFP32(0, { 5, 1, 1 }));

	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 0, 2, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 1, 2, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 2, 2, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 0, 3, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 1, 3, 1 }));
	EXPECT_EQ(111, dx_buf.GetFP32(0, { 2, 3, 1 }));

	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 3, 2, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 4, 2, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 5, 2, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 3, 3, 1 }));
	EXPECT_EQ(0,   dx_buf.GetFP32(0, { 4, 3, 1 }));
	EXPECT_EQ(112, dx_buf.GetFP32(0, { 5, 3, 1 }));


	//
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 0, 0, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 1, 0, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 0, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 0, 1, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 1, 1, 2 }));
	EXPECT_EQ(1201, dx_buf.GetFP32(1, { 2, 1, 2 }));

	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 3, 0, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 4, 0, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 5, 0, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 3, 1, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 4, 1, 2 }));
	EXPECT_EQ(1202, dx_buf.GetFP32(1, { 5, 1, 2 }));

	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 0, 2, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 1, 2, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 2, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 0, 3, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 1, 3, 2 }));
	EXPECT_EQ(1211, dx_buf.GetFP32(1, { 2, 3, 2 }));

	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 3, 2, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 4, 2, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 5, 2, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 3, 3, 2 }));
	EXPECT_EQ(0,    dx_buf.GetFP32(1, { 4, 3, 2 }));
	EXPECT_EQ(1212, dx_buf.GetFP32(1, { 5, 3, 2 }));

}

