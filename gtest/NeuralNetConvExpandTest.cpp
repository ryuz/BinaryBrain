#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/NeuralNetConvExpand.h"
#include "bb/NeuralNetConvExpandM.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetConvExpandTest, testNeuralNetConvExpand)
{
	bb::NeuralNetConvExpand<> cnvexp(2, 3, 4, 2, 3);
//	bb::NeuralNetConvExpandM<2, 3, 4, 2, 3> cnvexp;
	
	cnvexp.SetBatchSize(2);
	testSetupLayerBuffer(cnvexp);

	auto in_sig_buf = cnvexp.GetInputSignalBuffer();
	auto out_sig_buf = cnvexp.GetOutputSignalBuffer();

	EXPECT_EQ(2 * 3 * 4, cnvexp.GetInputNodeSize());
	EXPECT_EQ(2 * 2 * 3, cnvexp.GetOutputNodeSize());
	EXPECT_EQ(2, cnvexp.GetInputFrameSize());
	EXPECT_EQ(2 * 2 * 2, cnvexp.GetOutputFrameSize());

	in_sig_buf.SetDimensions({ 4, 3, 2 });
	for (size_t f = 0; f < 2; ++f) {
		for (size_t c = 0; c < 2; ++c) {
			for (size_t y = 0; y < 3; ++y) {
				for (size_t x = 0; x < 4; ++x) {
					in_sig_buf.SetReal(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
				}
			}
		}
	}

	cnvexp.Forward();

//	for (int i = 0; i < 2 * 2 * 3; ++i) {
//		std::cout << out_sig_buf.GetReal(0, i) << std::endl;
//	}

	out_sig_buf.SetDimensions({ 3, 2, 2 });
	EXPECT_EQ(0, out_sig_buf.GetReal(0,  { 0, 0, 0 }));
	EXPECT_EQ(1, out_sig_buf.GetReal(0,  { 1, 0, 0 }));
	EXPECT_EQ(2, out_sig_buf.GetReal(0,  { 2, 0, 0 }));
	EXPECT_EQ(10, out_sig_buf.GetReal(0, { 0, 1, 0 }));
	EXPECT_EQ(11, out_sig_buf.GetReal(0, { 1, 1, 0 }));
	EXPECT_EQ(12, out_sig_buf.GetReal(0, { 2, 1, 0 }));
	EXPECT_EQ(100, out_sig_buf.GetReal(0, { 0, 0, 1 }));
	EXPECT_EQ(101, out_sig_buf.GetReal(0, { 1, 0, 1 }));
	EXPECT_EQ(102, out_sig_buf.GetReal(0, { 2, 0, 1 }));
	EXPECT_EQ(110, out_sig_buf.GetReal(0, { 0, 1, 1 }));
	EXPECT_EQ(111, out_sig_buf.GetReal(0, { 1, 1, 1 }));
	EXPECT_EQ(112, out_sig_buf.GetReal(0, { 2, 1, 1 }));

	EXPECT_EQ(1, out_sig_buf.GetReal(1, { 0, 0, 0 }));
	EXPECT_EQ(2, out_sig_buf.GetReal(1, { 1, 0, 0 }));
	EXPECT_EQ(3, out_sig_buf.GetReal(1, { 2, 0, 0 }));
	EXPECT_EQ(11, out_sig_buf.GetReal(1, { 0, 1, 0 }));
	EXPECT_EQ(12, out_sig_buf.GetReal(1, { 1, 1, 0 }));
	EXPECT_EQ(13, out_sig_buf.GetReal(1, { 2, 1, 0 }));
	EXPECT_EQ(101, out_sig_buf.GetReal(1, { 0, 0, 1 }));
	EXPECT_EQ(102, out_sig_buf.GetReal(1, { 1, 0, 1 }));
	EXPECT_EQ(103, out_sig_buf.GetReal(1, { 2, 0, 1 }));
	EXPECT_EQ(111, out_sig_buf.GetReal(1, { 0, 1, 1 }));
	EXPECT_EQ(112, out_sig_buf.GetReal(1, { 1, 1, 1 }));
	EXPECT_EQ(113, out_sig_buf.GetReal(1, { 2, 1, 1 }));

	EXPECT_EQ(10, out_sig_buf.GetReal(2, { 0, 0, 0 }));
	EXPECT_EQ(11, out_sig_buf.GetReal(2, { 1, 0, 0 }));
	EXPECT_EQ(12, out_sig_buf.GetReal(2, { 2, 0, 0 }));
	EXPECT_EQ(20, out_sig_buf.GetReal(2, { 0, 1, 0 }));
	EXPECT_EQ(21, out_sig_buf.GetReal(2, { 1, 1, 0 }));
	EXPECT_EQ(22, out_sig_buf.GetReal(2, { 2, 1, 0 }));
	EXPECT_EQ(110, out_sig_buf.GetReal(2, { 0, 0, 1 }));
	EXPECT_EQ(111, out_sig_buf.GetReal(2, { 1, 0, 1 }));
	EXPECT_EQ(112, out_sig_buf.GetReal(2, { 2, 0, 1 }));
	EXPECT_EQ(120, out_sig_buf.GetReal(2, { 0, 1, 1 }));
	EXPECT_EQ(121, out_sig_buf.GetReal(2, { 1, 1, 1 }));
	EXPECT_EQ(122, out_sig_buf.GetReal(2, { 2, 1, 1 }));

	EXPECT_EQ(11, out_sig_buf.GetReal(3, { 0, 0, 0 }));
	EXPECT_EQ(12, out_sig_buf.GetReal(3, { 1, 0, 0 }));
	EXPECT_EQ(13, out_sig_buf.GetReal(3, { 2, 0, 0 }));
	EXPECT_EQ(21, out_sig_buf.GetReal(3, { 0, 1, 0 }));
	EXPECT_EQ(22, out_sig_buf.GetReal(3, { 1, 1, 0 }));
	EXPECT_EQ(23, out_sig_buf.GetReal(3, { 2, 1, 0 }));
	EXPECT_EQ(111, out_sig_buf.GetReal(3, { 0, 0, 1 }));
	EXPECT_EQ(112, out_sig_buf.GetReal(3, { 1, 0, 1 }));
	EXPECT_EQ(113, out_sig_buf.GetReal(3, { 2, 0, 1 }));
	EXPECT_EQ(121, out_sig_buf.GetReal(3, { 0, 1, 1 }));
	EXPECT_EQ(122, out_sig_buf.GetReal(3, { 1, 1, 1 }));
	EXPECT_EQ(123, out_sig_buf.GetReal(3, { 2, 1, 1 }));

	EXPECT_EQ(1010, out_sig_buf.GetReal(6, { 0, 0, 0 }));
	EXPECT_EQ(1011, out_sig_buf.GetReal(6, { 1, 0, 0 }));
	EXPECT_EQ(1012, out_sig_buf.GetReal(6, { 2, 0, 0 }));
	EXPECT_EQ(1020, out_sig_buf.GetReal(6, { 0, 1, 0 }));
	EXPECT_EQ(1021, out_sig_buf.GetReal(6, { 1, 1, 0 }));
	EXPECT_EQ(1022, out_sig_buf.GetReal(6, { 2, 1, 0 }));
	EXPECT_EQ(1110, out_sig_buf.GetReal(6, { 0, 0, 1 }));
	EXPECT_EQ(1111, out_sig_buf.GetReal(6, { 1, 0, 1 }));
	EXPECT_EQ(1112, out_sig_buf.GetReal(6, { 2, 0, 1 }));
	EXPECT_EQ(1120, out_sig_buf.GetReal(6, { 0, 1, 1 }));
	EXPECT_EQ(1121, out_sig_buf.GetReal(6, { 1, 1, 1 }));
	EXPECT_EQ(1122, out_sig_buf.GetReal(6, { 2, 1, 1 }));

	EXPECT_EQ(1011, out_sig_buf.GetReal(7, { 0, 0, 0 }));
	EXPECT_EQ(1012, out_sig_buf.GetReal(7, { 1, 0, 0 }));
	EXPECT_EQ(1013, out_sig_buf.GetReal(7, { 2, 0, 0 }));
	EXPECT_EQ(1021, out_sig_buf.GetReal(7, { 0, 1, 0 }));
	EXPECT_EQ(1022, out_sig_buf.GetReal(7, { 1, 1, 0 }));
	EXPECT_EQ(1023, out_sig_buf.GetReal(7, { 2, 1, 0 }));
	EXPECT_EQ(1111, out_sig_buf.GetReal(7, { 0, 0, 1 }));
	EXPECT_EQ(1112, out_sig_buf.GetReal(7, { 1, 0, 1 }));
	EXPECT_EQ(1113, out_sig_buf.GetReal(7, { 2, 0, 1 }));
	EXPECT_EQ(1121, out_sig_buf.GetReal(7, { 0, 1, 1 }));
	EXPECT_EQ(1122, out_sig_buf.GetReal(7, { 1, 1, 1 }));
	EXPECT_EQ(1123, out_sig_buf.GetReal(7, { 2, 1, 1 }));


	// backward
	auto out_err_buf = cnvexp.GetOutputErrorBuffer();
	auto in_err_buf = cnvexp.GetInputErrorBuffer();

	out_err_buf.SetDimensions({ 3, 2, 2 });
	for (size_t f = 0; f < 8; ++f) {
		for (size_t c = 0; c < 2; ++c) {
			for (size_t y = 0; y < 2; ++y) {
				for (size_t x = 0; x < 3; ++x) {
					out_err_buf.SetReal(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
				}
			}
		}
	}
	
	cnvexp.Backward();

//	for (int i = 0; i < 2 * 3 * 4; ++i) {
//		std::cout << in_err_buf.GetReal(0, i) << std::endl;
//	}

//	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 0, 0 }));
//	EXPECT_EQ(((0+1) + 0)+ ((1 + 2) + 1000), in_err_buf.GetReal(0, { 1, 0, 0 }));
//	EXPECT_EQ((1 + 2) + 1000, in_err_buf.GetReal(0, { 2, 0, 0 }));

}



#if 0

#include <chrono>

TEST(NeuralNetConvExpandTest, testNeuralNetConvExpandSpeed)
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
