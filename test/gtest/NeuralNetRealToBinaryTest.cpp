#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetRealToBinary.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}



TEST(NeuralNetRealToBinaryTest, testRealToBinary)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int frame_size = 1;

	bb::NeuralNetRealToBinary<> real2bin(node_size, node_size);
	testSetupLayerBuffer(real2bin);

	real2bin.SetMuxSize(mux_size);
	real2bin.SetBatchSize(1);

	EXPECT_EQ(1, real2bin.GetInputFrameSize());
	EXPECT_EQ(2, real2bin.GetOutputFrameSize());

	auto in_val = real2bin.GetInputSignalBuffer();
	auto out_val = real2bin.GetOutputSignalBuffer();
	in_val.SetReal(0, 0, 0.0f);
	in_val.SetReal(0, 1, 1.0f);
	in_val.SetReal(0, 2, 0.5f);

	real2bin.Forward();

	EXPECT_EQ(false, out_val.GetBinary(0, 0));
	EXPECT_EQ(false, out_val.GetBinary(1, 0));
	EXPECT_EQ(true, out_val.GetBinary(0, 1));
	EXPECT_EQ(true, out_val.GetBinary(1, 1));

	auto out_err = real2bin.GetOutputErrorBuffer();
	auto in_err = real2bin.GetInputErrorBuffer();

	out_err.SetReal(0, 0, 0);
	out_err.SetReal(1, 0, 0);
	out_err.SetReal(0, 1, 1);
	out_err.SetReal(1, 1, 1);
	out_err.SetReal(0, 2, 2);
	out_err.SetReal(1, 2, 2);
	
	real2bin.Backward();
	
	EXPECT_EQ(0.0f, in_err.GetReal(0, 0));
	EXPECT_EQ(2.0f, in_err.GetReal(0, 1));
	EXPECT_EQ(4.0f, in_err.Get<float>(0, 2));
}


TEST(NeuralNetRealToBinaryTest, testNeuralNetRealToBinaryBatch)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int batch_size = 2;

	bb::NeuralNetRealToBinary<> real2bin(node_size, node_size);
	testSetupLayerBuffer(real2bin);

	real2bin.SetMuxSize(mux_size);
	real2bin.SetBatchSize(batch_size);

	EXPECT_EQ(batch_size, real2bin.GetInputFrameSize());
	EXPECT_EQ(batch_size*mux_size, real2bin.GetOutputFrameSize());

	auto in_val = real2bin.GetInputSignalBuffer();
	auto out_val = real2bin.GetOutputSignalBuffer();
	in_val.SetReal(0, 0, 0.0f);
	in_val.SetReal(0, 1, 1.0f);
	in_val.SetReal(0, 2, 0.5f);
	in_val.SetReal(1, 0, 1.0f);
	in_val.SetReal(1, 1, 0.5f);
	in_val.SetReal(1, 2, 0.0f);
	
	real2bin.Forward();

	EXPECT_EQ(true, out_val.GetBinary(2, 0));
	EXPECT_EQ(true, out_val.Get<bool>(3, 0));
	EXPECT_EQ(false, out_val.GetBinary(2, 2));
	EXPECT_EQ(false, out_val.Get<bool>(3, 2));


	auto out_err = real2bin.GetOutputErrorBuffer();
	auto in_err = real2bin.GetInputErrorBuffer();

	out_err.SetReal(0, 0, 0);
	out_err.SetReal(1, 0, 0);
	out_err.SetReal(0, 1, 1);
	out_err.SetReal(1, 1, 2);
	out_err.SetReal(0, 2, 3);
	out_err.SetReal(1, 2, 4);

	out_err.SetReal(2, 0, 5);
	out_err.SetReal(3, 0, 6);
	out_err.SetReal(2, 1, 7);
	out_err.SetReal(3, 1, 8);
	out_err.SetReal(2, 2, 9);
	out_err.SetReal(3, 2, 10);
	
	real2bin.Backward();
	
	EXPECT_EQ(0.0f, in_err.GetReal(0, 0));
	EXPECT_EQ(3.0f, in_err.GetReal(0, 1));
	EXPECT_EQ(7.0f, in_err.GetReal(0, 2));

	EXPECT_EQ(11.0f, in_err.GetReal(1, 0));
	EXPECT_EQ(15.0f, in_err.GetReal(1, 1));
	EXPECT_EQ(19.0f, in_err.GetReal(1, 2));
}


