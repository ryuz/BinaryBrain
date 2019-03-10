#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/ConvolutionCol2Im.h"



TEST(ConvolutionCol2ImTest, testConvolutionCol2ImTest)
{
   	auto cnvcol2im = bb::ConvolutionCol2Im<>::Create(3, 4);

    bb::FrameBuffer x_buf(BB_TYPE_FP32, 2*(3*4), 2);
    cnvcol2im->SetInputShape(x_buf.GetShape());

    {
	    auto x_ptr = x_buf.GetPtr<float>();
	    for (size_t f = 0; f < 2; ++f) {
		    for (size_t y = 0; y < 3; ++y) {
			    for (size_t x = 0; x < 4; ++x) {
				    for (size_t c = 0; c < 2; ++c) {
					    x_ptr.Set((f*3+y)*4+x, c, (float)(1000 * f + c * 100 + y * 10 + x));
				    }
			    }
		    }
	    }
    }

	auto y_buf = cnvcol2im->Forward(x_buf);

//	for (int f = 0; f < 2; ++f) {
//		for (int i = 0; i < 2 * 3 * 4; ++i) {
//			std::cout << out_sig_buf.GetReal(f, i) << std::endl;
//		}
//	}


    {
	    auto y_ptr = y_buf.LockConst<float>();
	    EXPECT_EQ(0,   y_ptr.Get(0, { 0, 0, 0 }));
	    EXPECT_EQ(1,   y_ptr.Get(0, { 1, 0, 0 }));
	    EXPECT_EQ(2,   y_ptr.Get(0, { 2, 0, 0 }));
	    EXPECT_EQ(3,   y_ptr.Get(0, { 3, 0, 0 }));
	    EXPECT_EQ(10,  y_ptr.Get(0, { 0, 1, 0 }));
	    EXPECT_EQ(11,  y_ptr.Get(0, { 1, 1, 0 }));
	    EXPECT_EQ(12,  y_ptr.Get(0, { 2, 1, 0 }));
	    EXPECT_EQ(13,  y_ptr.Get(0, { 3, 1, 0 }));
	    EXPECT_EQ(20,  y_ptr.Get(0, { 0, 2, 0 }));
	    EXPECT_EQ(21,  y_ptr.Get(0, { 1, 2, 0 }));
	    EXPECT_EQ(22,  y_ptr.Get(0, { 2, 2, 0 }));
	    EXPECT_EQ(23,  y_ptr.Get(0, { 3, 2, 0 }));
	    EXPECT_EQ(100, y_ptr.Get(0, { 0, 0, 1 }));
	    EXPECT_EQ(101, y_ptr.Get(0, { 1, 0, 1 }));
	    EXPECT_EQ(102, y_ptr.Get(0, { 2, 0, 1 }));
	    EXPECT_EQ(103, y_ptr.Get(0, { 3, 0, 1 }));
	    EXPECT_EQ(110, y_ptr.Get(0, { 0, 1, 1 }));
	    EXPECT_EQ(111, y_ptr.Get(0, { 1, 1, 1 }));
	    EXPECT_EQ(112, y_ptr.Get(0, { 2, 1, 1 }));
	    EXPECT_EQ(113, y_ptr.Get(0, { 3, 1, 1 }));
	    EXPECT_EQ(120, y_ptr.Get(0, { 0, 2, 1 }));
	    EXPECT_EQ(121, y_ptr.Get(0, { 1, 2, 1 }));
	    EXPECT_EQ(122, y_ptr.Get(0, { 2, 2, 1 }));
	    EXPECT_EQ(123, y_ptr.Get(0, { 3, 2, 1 }));

	    EXPECT_EQ(1000, y_buf.GetFP32(1, { 0, 0, 0 }));
	    EXPECT_EQ(1001, y_buf.GetFP32(1, { 1, 0, 0 }));
	    EXPECT_EQ(1002, y_buf.GetFP32(1, { 2, 0, 0 }));
	    EXPECT_EQ(1003, y_buf.GetFP32(1, { 3, 0, 0 }));
	    EXPECT_EQ(1010, y_buf.GetFP32(1, { 0, 1, 0 }));
	    EXPECT_EQ(1011, y_buf.GetFP32(1, { 1, 1, 0 }));
	    EXPECT_EQ(1012, y_buf.GetFP32(1, { 2, 1, 0 }));
	    EXPECT_EQ(1013, y_buf.GetFP32(1, { 3, 1, 0 }));
	    EXPECT_EQ(1020, y_buf.GetFP32(1, { 0, 2, 0 }));
	    EXPECT_EQ(1021, y_buf.GetFP32(1, { 1, 2, 0 }));
	    EXPECT_EQ(1022, y_buf.GetFP32(1, { 2, 2, 0 }));
	    EXPECT_EQ(1023, y_buf.GetFP32(1, { 3, 2, 0 }));
	    EXPECT_EQ(1100, y_buf.GetFP32(1, { 0, 0, 1 }));
	    EXPECT_EQ(1101, y_buf.GetFP32(1, { 1, 0, 1 }));
	    EXPECT_EQ(1102, y_buf.GetFP32(1, { 2, 0, 1 }));
	    EXPECT_EQ(1103, y_buf.GetFP32(1, { 3, 0, 1 }));
	    EXPECT_EQ(1110, y_buf.GetFP32(1, { 0, 1, 1 }));
	    EXPECT_EQ(1111, y_buf.GetFP32(1, { 1, 1, 1 }));
	    EXPECT_EQ(1112, y_buf.GetFP32(1, { 2, 1, 1 }));
	    EXPECT_EQ(1113, y_buf.GetFP32(1, { 3, 1, 1 }));
	    EXPECT_EQ(1120, y_buf.GetFP32(1, { 0, 2, 1 }));
	    EXPECT_EQ(1121, y_buf.GetFP32(1, { 1, 2, 1 }));
	    EXPECT_EQ(1122, y_buf.GetFP32(1, { 2, 2, 1 }));
	    EXPECT_EQ(1123, y_buf.GetFP32(1, { 3, 2, 1 }));
    }

	
    // backward
    bb::FrameBuffer dy_buf(BB_TYPE_FP32, 2, {4, 3, 2});

	for (bb::index_t f = 0; f < 2; ++f) {
		for (bb::index_t c = 0; c < 2; ++c) {
			for (bb::index_t y = 0; y < 3; ++y) {
				for (bb::index_t x = 0; x < 4; ++x) {
					float val = y_buf.GetFP32(f, { x, y, c });
					dy_buf.SetFP32(f, { x, y, c }, val + 10000);
				}
			}
		}
	}

	auto dx_buf = cnvcol2im->Backward(dy_buf);

	for (bb::index_t f = 0; f < 2; ++f) {
		for (bb::index_t y = 0; y < 3; ++y) {
			for (bb::index_t x = 0; x < 4; ++x) {
				for (bb::index_t c = 0; c < 2; ++c) {
					EXPECT_EQ(x_buf.GetFP32((f * 3 + y) * 4 + x, c) + 10000,
						dx_buf.GetFP32((f * 3 + y) * 4 + x, c));
				}
			}
		}
	}

}


