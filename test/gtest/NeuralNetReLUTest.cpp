#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/NeuralNetReLU.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}



TEST(NeuralNetReLUTest, testNeuralNetReLUTest)
{
	bb::NeuralNetReLU<> relu(3);
	relu.SetBatchSize(2);

	testSetupLayerBuffer(relu);
	auto in_sig_buf = relu.GetInputSignalBuffer();
	auto out_sig_buf = relu.GetOutputSignalBuffer();

	in_sig_buf.SetReal(0, 0, -1);
	in_sig_buf.SetReal(0, 1, 0);
	in_sig_buf.SetReal(0, 2, 1);
	in_sig_buf.SetReal(1, 0, 2);
	in_sig_buf.SetReal(1, 1, 1);
	in_sig_buf.SetReal(1, 2, -2);
	relu.Forward();
	
	EXPECT_EQ(0, out_sig_buf.GetReal(0, 0));
	EXPECT_EQ(0, out_sig_buf.GetReal(0, 1));
	EXPECT_EQ(1, out_sig_buf.GetReal(0, 2));
	EXPECT_EQ(2, out_sig_buf.GetReal(1, 0));
	EXPECT_EQ(1, out_sig_buf.GetReal(1, 1));
	EXPECT_EQ(0, out_sig_buf.GetReal(1, 2));

	// backward
	auto out_err_buf = relu.GetOutputErrorBuffer();
	auto in_err_buf = relu.GetInputErrorBuffer();

	out_err_buf.SetReal(0, 0, 1);
	out_err_buf.SetReal(0, 1, 2);
	out_err_buf.SetReal(0, 2, 3);
	out_err_buf.SetReal(1, 0, 4);
	out_err_buf.SetReal(1, 1, 5);
	out_err_buf.SetReal(1, 2, 6);

	relu.Backward();

	EXPECT_EQ(0, in_err_buf.GetReal(0, 0));
	EXPECT_EQ(0, in_err_buf.GetReal(0, 1));
	EXPECT_EQ(3, in_err_buf.GetReal(0, 2));
	EXPECT_EQ(4, in_err_buf.GetReal(1, 0));
	EXPECT_EQ(5, in_err_buf.GetReal(1, 1));
	EXPECT_EQ(0, in_err_buf.GetReal(1, 2));

}

