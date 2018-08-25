#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/NeuralNetConvolution.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputValueBuffer (net.CreateInputValueBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputValueBuffer(net.CreateOutputValueBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetConvolutionTest, testNeuralNetConvolution)
{
	bb::NeuralNetConvolution<> cnv(1, 3, 3, 1, 2, 2);
	testSetupLayerBuffer(cnv);
	auto in_val = cnv.GetInputValueBuffer();
	auto out_val = cnv.GetOutputValueBuffer();

	EXPECT_EQ(9, cnv.GetInputNodeSize());
	EXPECT_EQ(4, cnv.GetOutputNodeSize());

	in_val.SetReal(0, 3 * 0 + 0, 0.1f);
	in_val.SetReal(0, 3 * 0 + 1, 0.2f);
	in_val.SetReal(0, 3 * 0 + 2, 0.3f);
	in_val.SetReal(0, 3 * 1 + 0, 0.4f);
	in_val.SetReal(0, 3 * 1 + 1, 0.5f);
	in_val.SetReal(0, 3 * 1 + 2, 0.6f);
	in_val.SetReal(0, 3 * 2 + 0, 0.7f);
	in_val.SetReal(0, 3 * 2 + 1, 0.8f);
	in_val.SetReal(0, 3 * 2 + 2, 0.9f);

	cnv.W(0, 0, 0, 0) = 0.1f;
	cnv.W(0, 0, 0, 1) = 0.2f;
	cnv.W(0, 0, 1, 0) = 0.3f;
	cnv.W(0, 0, 1, 1) = 0.4f;

	cnv.b(0) = 0.321f;

	cnv.Forward();

	float exp00 = 0.321f
		+ 0.1f * 0.1f
		+ 0.2f * 0.2f
		+ 0.4f * 0.3f
		+ 0.5f * 0.4f;

	float exp01 = 0.321f
		+ 0.2f * 0.1f
		+ 0.3f * 0.2f
		+ 0.5f * 0.3f
		+ 0.6f * 0.4f;

	float exp10 = 0.321f
		+ 0.4f * 0.1f
		+ 0.5f * 0.2f
		+ 0.7f * 0.3f
		+ 0.8f * 0.4f;

	float exp11 = 0.321f
		+ 0.5f * 0.1f
		+ 0.6f * 0.2f
		+ 0.8f * 0.3f
		+ 0.9f * 0.4f;

	EXPECT_EQ(exp00, out_val.GetReal(0, 0));
	EXPECT_EQ(exp01, out_val.GetReal(0, 1));
	EXPECT_EQ(exp10, out_val.GetReal(0, 2));
	EXPECT_EQ(exp11, out_val.GetReal(0, 3));
}



