#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/NeuralNetConvolutionCol2Im.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetConvolutionCol2ImTest, testNeuralNetConvCollapse)
{
	bb::NeuralNetConvolutionCol2Im<> cnvcol2im(2, 3, 4);

	cnvcol2im.SetBatchSize(2);
	testSetupLayerBuffer(cnvcol2im);

	auto in_sig_buf = cnvcol2im.GetInputSignalBuffer();
	auto out_sig_buf = cnvcol2im.GetOutputSignalBuffer();

	EXPECT_EQ(2, cnvcol2im.GetInputNodeSize());
	EXPECT_EQ(2 * 3 * 4, cnvcol2im.GetInputFrameSize());
	
	EXPECT_EQ(2 * 3 * 4, cnvcol2im.GetOutputNodeSize());
	EXPECT_EQ(2, cnvcol2im.GetOutputFrameSize());

	for (size_t f = 0; f < 2; ++f) {
		for (size_t y = 0; y < 3; ++y) {
			for (size_t x = 0; x < 4; ++x) {
				for (size_t c = 0; c < 2; ++c) {
					in_sig_buf.SetReal((f*3+y)*4+x, c, (float)(1000 * f + c * 100 + y * 10 + x));
				}
			}
		}
	}

	cnvcol2im.Forward();

//	for (int f = 0; f < 2; ++f) {
//		for (int i = 0; i < 2 * 3 * 4; ++i) {
//			std::cout << out_sig_buf.GetReal(f, i) << std::endl;
//		}
//	}

	out_sig_buf.SetDimensions({ 4, 3, 2 });
	EXPECT_EQ(0, out_sig_buf.GetReal(0, { 0, 0, 0 }));
	EXPECT_EQ(1, out_sig_buf.GetReal(0, { 1, 0, 0 }));
	EXPECT_EQ(2, out_sig_buf.GetReal(0, { 2, 0, 0 }));
	EXPECT_EQ(3, out_sig_buf.GetReal(0, { 3, 0, 0 }));
	EXPECT_EQ(10, out_sig_buf.GetReal(0, { 0, 1, 0 }));
	EXPECT_EQ(11, out_sig_buf.GetReal(0, { 1, 1, 0 }));
	EXPECT_EQ(12, out_sig_buf.GetReal(0, { 2, 1, 0 }));
	EXPECT_EQ(13, out_sig_buf.GetReal(0, { 3, 1, 0 }));
	EXPECT_EQ(20, out_sig_buf.GetReal(0, { 0, 2, 0 }));
	EXPECT_EQ(21, out_sig_buf.GetReal(0, { 1, 2, 0 }));
	EXPECT_EQ(22, out_sig_buf.GetReal(0, { 2, 2, 0 }));
	EXPECT_EQ(23, out_sig_buf.GetReal(0, { 3, 2, 0 }));
	EXPECT_EQ(100, out_sig_buf.GetReal(0, { 0, 0, 1 }));
	EXPECT_EQ(101, out_sig_buf.GetReal(0, { 1, 0, 1 }));
	EXPECT_EQ(102, out_sig_buf.GetReal(0, { 2, 0, 1 }));
	EXPECT_EQ(103, out_sig_buf.GetReal(0, { 3, 0, 1 }));
	EXPECT_EQ(110, out_sig_buf.GetReal(0, { 0, 1, 1 }));
	EXPECT_EQ(111, out_sig_buf.GetReal(0, { 1, 1, 1 }));
	EXPECT_EQ(112, out_sig_buf.GetReal(0, { 2, 1, 1 }));
	EXPECT_EQ(113, out_sig_buf.GetReal(0, { 3, 1, 1 }));
	EXPECT_EQ(120, out_sig_buf.GetReal(0, { 0, 2, 1 }));
	EXPECT_EQ(121, out_sig_buf.GetReal(0, { 1, 2, 1 }));
	EXPECT_EQ(122, out_sig_buf.GetReal(0, { 2, 2, 1 }));
	EXPECT_EQ(123, out_sig_buf.GetReal(0, { 3, 2, 1 }));

	EXPECT_EQ(1000, out_sig_buf.GetReal(1, { 0, 0, 0 }));
	EXPECT_EQ(1001, out_sig_buf.GetReal(1, { 1, 0, 0 }));
	EXPECT_EQ(1002, out_sig_buf.GetReal(1, { 2, 0, 0 }));
	EXPECT_EQ(1003, out_sig_buf.GetReal(1, { 3, 0, 0 }));
	EXPECT_EQ(1010, out_sig_buf.GetReal(1, { 0, 1, 0 }));
	EXPECT_EQ(1011, out_sig_buf.GetReal(1, { 1, 1, 0 }));
	EXPECT_EQ(1012, out_sig_buf.GetReal(1, { 2, 1, 0 }));
	EXPECT_EQ(1013, out_sig_buf.GetReal(1, { 3, 1, 0 }));
	EXPECT_EQ(1020, out_sig_buf.GetReal(1, { 0, 2, 0 }));
	EXPECT_EQ(1021, out_sig_buf.GetReal(1, { 1, 2, 0 }));
	EXPECT_EQ(1022, out_sig_buf.GetReal(1, { 2, 2, 0 }));
	EXPECT_EQ(1023, out_sig_buf.GetReal(1, { 3, 2, 0 }));
	EXPECT_EQ(1100, out_sig_buf.GetReal(1, { 0, 0, 1 }));
	EXPECT_EQ(1101, out_sig_buf.GetReal(1, { 1, 0, 1 }));
	EXPECT_EQ(1102, out_sig_buf.GetReal(1, { 2, 0, 1 }));
	EXPECT_EQ(1103, out_sig_buf.GetReal(1, { 3, 0, 1 }));
	EXPECT_EQ(1110, out_sig_buf.GetReal(1, { 0, 1, 1 }));
	EXPECT_EQ(1111, out_sig_buf.GetReal(1, { 1, 1, 1 }));
	EXPECT_EQ(1112, out_sig_buf.GetReal(1, { 2, 1, 1 }));
	EXPECT_EQ(1113, out_sig_buf.GetReal(1, { 3, 1, 1 }));
	EXPECT_EQ(1120, out_sig_buf.GetReal(1, { 0, 2, 1 }));
	EXPECT_EQ(1121, out_sig_buf.GetReal(1, { 1, 2, 1 }));
	EXPECT_EQ(1122, out_sig_buf.GetReal(1, { 2, 2, 1 }));
	EXPECT_EQ(1123, out_sig_buf.GetReal(1, { 3, 2, 1 }));


	// backward
	auto out_err_buf = cnvcol2im.GetOutputErrorBuffer();
	auto in_err_buf = cnvcol2im.GetInputErrorBuffer();

	out_sig_buf.SetDimensions({ 4, 3, 2 });
	out_err_buf.SetDimensions({ 4, 3, 2 });
	for (bb::INDEX f = 0; f < 2; ++f) {
		for (bb::INDEX c = 0; c < 2; ++c) {
			for (bb::INDEX y = 0; y < 3; ++y) {
				for (bb::INDEX x = 0; x < 4; ++x) {
					float val = out_sig_buf.GetReal(f, { x, y, c });
					out_err_buf.SetReal(f, { x, y, c }, val + 10000);
				}
			}
		}
	}

	cnvcol2im.Backward();

	for (size_t f = 0; f < 2; ++f) {
		for (size_t y = 0; y < 3; ++y) {
			for (size_t x = 0; x < 4; ++x) {
				for (size_t c = 0; c < 2; ++c) {
					EXPECT_EQ(in_sig_buf.GetReal((f * 3 + y) * 4 + x, c) + 10000,
						in_err_buf.GetReal((f * 3 + y) * 4 + x, c));
				}
			}
		}
	}

//	for (int i = 0; i < 2 * 3 * 4; ++i) {
//		std::cout << in_err_buf.GetReal(0, i) << std::endl;
//	}

//	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 0, 0 }));
//	EXPECT_EQ(((0+1) + 0)+ ((1 + 2) + 1000), in_err_buf.GetReal(0, { 1, 0, 0 }));
//	EXPECT_EQ((1 + 2) + 1000, in_err_buf.GetReal(0, { 2, 0, 0 }));

}


