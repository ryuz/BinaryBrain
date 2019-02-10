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
	
    bb::FrameBuffer buf_x(16, {4, 3, 2}, BB_TYPE_FP32);

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

    /*
	// backward
	auto out_err_buf = cnvim2col.GetOutputErrorBuffer();
	auto in_err_buf = cnvim2col.GetInputErrorBuffer();

	out_err_buf.SetDimensions({ 3, 2, 2 });
	for (bb::INDEX f = 0; f < 8; ++f) {
		for (bb::INDEX c = 0; c < 2; ++c) {
			for (bb::INDEX y = 0; y < 2; ++y) {
				for (bb::INDEX x = 0; x < 3; ++x) {
					out_err_buf.SetReal(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
				}
			}
		}
	}
	
	cnvim2col.Backward();

//	for (int i = 0; i < 2 * 3 * 4; ++i) {
//		std::cout << in_err_buf.GetReal(0, i) << std::endl;
//	}

//	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 0, 0 }));
//	EXPECT_EQ(((0+1) + 0)+ ((1 + 2) + 1000), in_err_buf.GetReal(0, { 1, 0, 0 }));
//	EXPECT_EQ((1 + 2) + 1000, in_err_buf.GetReal(0, { 2, 0, 0 }));
*/
}



#if 0

#if 0

#include <chrono>

TEST(NeuralNetConvolutionIm2ColTest, testNeuralNetConvolutionIm2ColSpeed)
{
	// 実践的なサイズで速度比較
	bb::NeuralNetConvExpand<> cnvexp(100, 28, 28, 3, 3);
	bb::NeuralNetConvExpandM<100, 28, 28, 3, 3> cnvexpM;

	cnvexp.SetBatchSize(256);
	cnvexpM.SetBatchSize(256);
	testSetupLayerBuffer(cnvexp);
	testSetupLayerBuffer(cnvexpM);

	std::chrono::system_clock::time_point  start, end;

	start = std::chrono::system_clock::now();
	cnvexp.Forward();
	end = std::chrono::system_clock::now();

	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << elapsed << std::endl;


	start = std::chrono::system_clock::now();
	cnvexpM.Forward();
	end = std::chrono::system_clock::now();

	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << elapsed << std::endl;
}


#endif

#endif