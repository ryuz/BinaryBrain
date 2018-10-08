#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetBinaryToReal.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetBinaryToRealTest, testNeuralNetUnbinarize)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int frame_size = 1;

	bb::NeuralNetBinaryToReal<> bin2real(node_size, node_size);
	testSetupLayerBuffer(bin2real);

	bin2real.SetMuxSize(mux_size);
	bin2real.SetBatchSize(1);

	EXPECT_EQ(2, bin2real.GetInputFrameSize());
	EXPECT_EQ(1, bin2real.GetOutputFrameSize());


	auto in_val = bin2real.GetInputSignalBuffer();
	auto out_val = bin2real.GetOutputSignalBuffer();
	in_val.SetBinary(0, 0, true);
	in_val.SetBinary(1, 0, true);
	in_val.SetBinary(0, 1, false);
	in_val.SetBinary(1, 1, false);
	in_val.SetBinary(0, 2, false);
	in_val.SetBinary(1, 2, true);

	bin2real.Forward();
	

	EXPECT_EQ(1.0, out_val.GetReal(0, 0));
	EXPECT_EQ(0.0, out_val.GetReal(0, 1));
	EXPECT_EQ(0.5, out_val.GetReal(0, 2));
	

	auto out_err = bin2real.GetOutputErrorBuffer();
	auto in_err = bin2real.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 0.0f);
	out_err.SetReal(0, 1, 1.0f);
	out_err.SetReal(0, 2, 0.5f);

	bin2real.Backward();

	EXPECT_EQ(0.0f, in_err.GetReal(0, 0));
	EXPECT_EQ(0.0f, in_err.GetReal(1, 0));
	EXPECT_EQ(1.0f, in_err.GetReal(0, 1));
	EXPECT_EQ(1.0f, in_err.GetReal(1, 1));
	EXPECT_EQ(0.5f, in_err.GetReal(0, 2));
	EXPECT_EQ(0.5f, in_err.GetReal(1, 2));
}



