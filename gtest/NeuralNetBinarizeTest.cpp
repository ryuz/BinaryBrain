#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetBinarize.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputValueBuffer (net.CreateInputValueBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputValueBuffer(net.CreateOutputValueBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}



TEST(NeuralNetBinarizeTest, testNeuralNetBinarize)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int frame_size = 1;

	bb::NeuralNetBinarize<> binarize(node_size, node_size);
	testSetupLayerBuffer(binarize);

	binarize.SetMuxSize(mux_size);
	binarize.SetBatchSize(1);

	EXPECT_EQ(1, binarize.GetInputFrameSize());
	EXPECT_EQ(2, binarize.GetOutputFrameSize());

	auto in_val = binarize.GetInputValueBuffer();
	auto out_val = binarize.GetOutputValueBuffer();
	in_val.SetReal(0, 0, 0.0f);
	in_val.SetReal(0, 1, 1.0f);
	in_val.SetReal(0, 2, 0.5f);

	binarize.Forward();

	EXPECT_EQ(false, out_val.GetBinary(0, 0));
	EXPECT_EQ(false, out_val.GetBinary(1, 0));
	EXPECT_EQ(true, out_val.GetBinary(0, 1));
	EXPECT_EQ(true, out_val.GetBinary(1, 1));

	auto out_err = binarize.GetOutputErrorBuffer();
	auto in_err = binarize.GetInputErrorBuffer();

	out_err.SetBinary(0, 0, false);
	out_err.SetBinary(1, 0, false);
	out_err.SetBinary(0, 1, true);
	out_err.SetBinary(1, 1, true);
	out_err.SetBinary(0, 2, true);
	out_err.SetBinary(1, 2, false);


	binarize.Backward();
	
//	EXPECT_EQ(0.0f, in_err.GetReal(0, 0));
//	EXPECT_EQ(1.0f, in_err.GetReal(0, 1));
//	EXPECT_EQ(0.5f, in_err.Get<float>(0, 2));
}


TEST(NeuralNetBinarizeTest, testNeuralNetBinarizeBatch)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int batch_size = 2;

	bb::NeuralNetBinarize<> binarize(node_size, node_size);
	testSetupLayerBuffer(binarize);

	binarize.SetMuxSize(mux_size);
	binarize.SetBatchSize(batch_size);

	EXPECT_EQ(batch_size, binarize.GetInputFrameSize());
	EXPECT_EQ(batch_size*mux_size, binarize.GetOutputFrameSize());

	auto in_val = binarize.GetInputValueBuffer();
	auto out_val = binarize.GetOutputValueBuffer();
	in_val.SetReal(0, 0, 0.0f);
	in_val.SetReal(0, 1, 1.0f);
	in_val.SetReal(0, 2, 0.5f);
	in_val.SetReal(1, 0, 1.0f);
	in_val.SetReal(1, 1, 0.5f);
	in_val.SetReal(1, 2, 0.0f);
	
	binarize.Forward();

	EXPECT_EQ(true, out_val.GetBinary(2, 0));
	EXPECT_EQ(true, out_val.Get<bool>(3, 0));
	EXPECT_EQ(false, out_val.GetBinary(2, 2));
	EXPECT_EQ(false, out_val.Get<bool>(3, 2));


	auto out_err = binarize.GetOutputErrorBuffer();
	auto in_err = binarize.GetInputErrorBuffer();

	out_err.SetBinary(0, 0, false);
	out_err.SetBinary(1, 0, false);
	out_err.SetBinary(0, 1, true);
	out_err.SetBinary(1, 1, true);
	out_err.SetBinary(0, 2, true);
	out_err.SetBinary(1, 2, false);

	out_err.SetBinary(2, 0, true);
	out_err.SetBinary(3, 0, false);
	out_err.SetBinary(2, 1, false);
	out_err.SetBinary(3, 1, false);
	out_err.SetBinary(2, 2, true);
	out_err.SetBinary(3, 2, true);
	
	binarize.Backward();
	
//	EXPECT_EQ(0.0f, in_err.GetReal(0, 0));
//	EXPECT_EQ(1.0f, in_err.GetReal(0, 1));
//	EXPECT_EQ(0.5f, in_err.GetReal(0, 2));

//	EXPECT_EQ(0.5f, in_err.GetReal(1, 0));
//	EXPECT_EQ(0.0f, in_err.GetReal(1, 1));
//	EXPECT_EQ(1.0f, in_err.GetReal(1, 2));
}


