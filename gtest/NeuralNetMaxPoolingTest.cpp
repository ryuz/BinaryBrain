#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/NeuralNetMaxPooling.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}



TEST(NeuralNetMaxPoolingTest, testNeuralNetMaxPoolingTest)
{
	bb::NeuralNetMaxPooling<> maxpol(3, 4, 6, 2, 3);
	maxpol.SetBatchSize(2);

	testSetupLayerBuffer(maxpol);
	auto in_sig_buf = maxpol.GetInputSignalBuffer();
	auto out_sig_buf = maxpol.GetOutputSignalBuffer();

	in_sig_buf.SetDimensions({ 6, 4, 3 });
	for (size_t f = 0; f < 2; ++f) {
		for (size_t c = 0; c < 3; ++c) {
			for (size_t y = 0; y < 4; ++y) {
				for (size_t x = 0; x < 6; ++x) {
					in_sig_buf.SetReal(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
				}
			}
		}
	}
	in_sig_buf.SetReal(0, { 1, 0, 0 }, 99);

	maxpol.Forward();
	
	out_sig_buf.SetDimensions({ 2, 2, 3 });
	EXPECT_EQ(99, out_sig_buf.GetReal(0, { 0, 0, 0 }));
	EXPECT_EQ(15, out_sig_buf.GetReal(0, { 1, 0, 0 }));
	EXPECT_EQ(32, out_sig_buf.GetReal(0, { 0, 1, 0 }));
	EXPECT_EQ(35, out_sig_buf.GetReal(0, { 1, 1, 0 }));

	EXPECT_EQ(112, out_sig_buf.GetReal(0, { 0, 0, 1 }));
	EXPECT_EQ(115, out_sig_buf.GetReal(0, { 1, 0, 1 }));
	EXPECT_EQ(132, out_sig_buf.GetReal(0, { 0, 1, 1 }));
	EXPECT_EQ(135, out_sig_buf.GetReal(0, { 1, 1, 1 }));

	EXPECT_EQ(212, out_sig_buf.GetReal(0, { 0, 0, 2 }));
	EXPECT_EQ(215, out_sig_buf.GetReal(0, { 1, 0, 2 }));
	EXPECT_EQ(232, out_sig_buf.GetReal(0, { 0, 1, 2 }));
	EXPECT_EQ(235, out_sig_buf.GetReal(0, { 1, 1, 2 }));

	EXPECT_EQ(1012, out_sig_buf.GetReal(1, { 0, 0, 0 }));
	EXPECT_EQ(1015, out_sig_buf.GetReal(1, { 1, 0, 0 }));
	EXPECT_EQ(1032, out_sig_buf.GetReal(1, { 0, 1, 0 }));
	EXPECT_EQ(1035, out_sig_buf.GetReal(1, { 1, 1, 0 }));

	EXPECT_EQ(1112, out_sig_buf.GetReal(1, { 0, 0, 1 }));
	EXPECT_EQ(1115, out_sig_buf.GetReal(1, { 1, 0, 1 }));
	EXPECT_EQ(1132, out_sig_buf.GetReal(1, { 0, 1, 1 }));
	EXPECT_EQ(1135, out_sig_buf.GetReal(1, { 1, 1, 1 }));

	EXPECT_EQ(1212, out_sig_buf.GetReal(1, { 0, 0, 2 }));
	EXPECT_EQ(1215, out_sig_buf.GetReal(1, { 1, 0, 2 }));
	EXPECT_EQ(1232, out_sig_buf.GetReal(1, { 0, 1, 2 }));
	EXPECT_EQ(1235, out_sig_buf.GetReal(1, { 1, 1, 2 }));


	// backward
	auto in_err_buf = maxpol.GetInputErrorBuffer();
	auto out_err_buf = maxpol.GetOutputErrorBuffer();

	out_err_buf.SetDimensions({ 2, 2, 3 });
	in_err_buf.SetDimensions({ 6, 4, 3 });

	for (size_t f = 0; f < 2; ++f) {
		for (size_t c = 0; c < 3; ++c) {
			for (size_t y = 0; y < 2; ++y) {
				for (size_t x = 0; x < 2; ++x) {
					out_err_buf.SetReal(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x + 1));
				}
			}
		}
	}

	maxpol.Backward();

	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 0, 0 }));
	EXPECT_EQ(1, in_err_buf.GetReal(0, { 1, 0, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 2, 0, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 1, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 1, 1, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 2, 1, 0 }));

	EXPECT_EQ(0, in_err_buf.GetReal(0, { 3, 0, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 4, 0, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 5, 0, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 3, 1, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 4, 1, 0 }));
	EXPECT_EQ(2, in_err_buf.GetReal(0, { 5, 1, 0 }));

	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 2, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 1, 2, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 2, 2, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 3, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 1, 3, 0 }));
	EXPECT_EQ(11, in_err_buf.GetReal(0, { 2, 3, 0 }));

	EXPECT_EQ(0, in_err_buf.GetReal(0, { 3, 2, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 4, 2, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 5, 2, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 3, 3, 0 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 4, 3, 0 }));
	EXPECT_EQ(12, in_err_buf.GetReal(0, { 5, 3, 0 }));

	//
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 0, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 1, 0, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 2, 0, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 1, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 1, 1, 1 }));
	EXPECT_EQ(101, in_err_buf.GetReal(0, { 2, 1, 1 }));

	EXPECT_EQ(0, in_err_buf.GetReal(0, { 3, 0, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 4, 0, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 5, 0, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 3, 1, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 4, 1, 1 }));
	EXPECT_EQ(102, in_err_buf.GetReal(0, { 5, 1, 1 }));

	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 2, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 1, 2, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 2, 2, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 3, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 1, 3, 1 }));
	EXPECT_EQ(111, in_err_buf.GetReal(0, { 2, 3, 1 }));

	EXPECT_EQ(0, in_err_buf.GetReal(0, { 3, 2, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 4, 2, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 5, 2, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 3, 3, 1 }));
	EXPECT_EQ(0, in_err_buf.GetReal(0, { 4, 3, 1 }));
	EXPECT_EQ(112, in_err_buf.GetReal(0, { 5, 3, 1 }));


	//
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 0, 0, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 1, 0, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 2, 0, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 0, 1, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 1, 1, 2 }));
	EXPECT_EQ(1201, in_err_buf.GetReal(1, { 2, 1, 2 }));

	EXPECT_EQ(0, in_err_buf.GetReal(1, { 3, 0, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 4, 0, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 5, 0, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 3, 1, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 4, 1, 2 }));
	EXPECT_EQ(1202, in_err_buf.GetReal(1, { 5, 1, 2 }));

	EXPECT_EQ(0, in_err_buf.GetReal(1, { 0, 2, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 1, 2, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 2, 2, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 0, 3, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 1, 3, 2 }));
	EXPECT_EQ(1211, in_err_buf.GetReal(1, { 2, 3, 2 }));

	EXPECT_EQ(0, in_err_buf.GetReal(1, { 3, 2, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 4, 2, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 5, 2, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 3, 3, 2 }));
	EXPECT_EQ(0, in_err_buf.GetReal(1, { 4, 3, 2 }));
	EXPECT_EQ(1212, in_err_buf.GetReal(1, { 5, 3, 2 }));
}

