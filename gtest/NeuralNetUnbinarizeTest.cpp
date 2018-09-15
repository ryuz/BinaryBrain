#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetUnbinarize.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetUnbinarizeTest, testNeuralNetUnbinarize)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int frame_size = 1;

	bb::NeuralNetUnbinarize<> unbinarize(node_size, node_size);
	testSetupLayerBuffer(unbinarize);

	unbinarize.SetMuxSize(mux_size);
	unbinarize.SetBatchSize(1);

	EXPECT_EQ(2, unbinarize.GetInputFrameSize());
	EXPECT_EQ(1, unbinarize.GetOutputFrameSize());


	auto in_val = unbinarize.GetInputSignalBuffer();
	auto out_val = unbinarize.GetOutputSignalBuffer();
	in_val.SetBinary(0, 0, true);
	in_val.SetBinary(1, 0, true);
	in_val.SetBinary(0, 1, false);
	in_val.SetBinary(1, 1, false);
	in_val.SetBinary(0, 2, false);
	in_val.SetBinary(1, 2, true);

	unbinarize.Forward();
	

	EXPECT_EQ(1.0, out_val.GetReal(0, 0));
	EXPECT_EQ(0.0, out_val.GetReal(0, 1));
	EXPECT_EQ(0.5, out_val.GetReal(0, 2));
	

	auto out_err = unbinarize.GetOutputErrorBuffer();
	auto in_err = unbinarize.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 0.0f);
	out_err.SetReal(0, 1, 1.0f);
	out_err.SetReal(0, 2, 0.5f);

	unbinarize.Backward();

//	EXPECT_EQ(false, in_err.GetBinary(0, 0));
//	EXPECT_EQ(false, in_err.GetBinary(1, 0));
//	EXPECT_EQ(true,  in_err.GetBinary(0, 1));
//	EXPECT_EQ(true,  in_err.GetBinary(1, 1));
}



