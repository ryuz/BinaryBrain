#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetSigmoid.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}



TEST(NeuralNetSigmoidTest, testSigmoid)
{
	bb::NeuralNetSigmoid<> sigmoid(2);
	testSetupLayerBuffer(sigmoid);


	auto in_val = sigmoid.GetInputSignalBuffer();
	auto out_val = sigmoid.GetOutputSignalBuffer();
	in_val.SetReal(0, 0, 1);
	in_val.SetReal(0, 1, 2);

	sigmoid.Forward();
	EXPECT_EQ(1.0f / (1.0f + exp(-1.0f)), out_val.GetReal(0, 0));
	EXPECT_EQ(1.0f / (1.0f + exp(-2.0f)), out_val.GetReal(0, 1));

	auto out_err = sigmoid.GetOutputErrorBuffer();
	auto in_err = sigmoid.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 2);
	out_err.SetReal(0, 1, 3);


	sigmoid.Backward();


	EXPECT_EQ(out_err.GetReal(0, 0) * (1.0f - out_val.GetReal(0, 0)) * out_val.GetReal(0, 0), in_err.GetReal(0, 0));
	EXPECT_EQ(out_err.GetReal(0, 1) * (1.0f - out_val.GetReal(0, 1)) * out_val.GetReal(0, 1), in_err.GetReal(0, 1));
}


TEST(NeuralNetSigmoidTest, testSigmoidBatch)
{
	bb::NeuralNetSigmoid<> sigmoid(2);
	testSetupLayerBuffer(sigmoid);

	sigmoid.SetBatchSize(2);

	auto in_val = sigmoid.GetInputSignalBuffer();
	auto out_val = sigmoid.GetOutputSignalBuffer();
	in_val.SetReal(0, 0, 1);
	in_val.SetReal(1, 0, 2);
	in_val.SetReal(0, 1, 3);
	in_val.SetReal(1, 1, 4);

	sigmoid.Forward();
	EXPECT_EQ(1.0f / (1.0f + exp(-1.0f)), out_val.GetReal(0, 0));
	EXPECT_EQ(1.0f / (1.0f + exp(-2.0f)), out_val.GetReal(1, 0));
	EXPECT_EQ(1.0f / (1.0f + exp(-3.0f)), out_val.GetReal(0, 1));
	EXPECT_EQ(1.0f / (1.0f + exp(-4.0f)), out_val.GetReal(1, 1));


	auto out_err = sigmoid.GetOutputErrorBuffer();
	auto in_err = sigmoid.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 2);
	out_err.SetReal(1, 0, 3);
	out_err.SetReal(0, 1, 4);
	out_err.SetReal(1, 1, -5);


	sigmoid.Backward();

	EXPECT_EQ(out_err.GetReal(0, 0) * (1.0f - out_val.GetReal(0, 0)) * out_val.GetReal(0, 0), in_err.GetReal(0, 0));
	EXPECT_EQ(out_err.GetReal(1, 0) * (1.0f - out_val.GetReal(1, 0)) * out_val.GetReal(1, 0), in_err.GetReal(1, 0));
	EXPECT_EQ(out_err.GetReal(0, 1) * (1.0f - out_val.GetReal(0, 1)) * out_val.GetReal(0, 1), in_err.GetReal(0, 1));
	EXPECT_EQ(out_err.GetReal(1, 1) * (1.0f - out_val.GetReal(1, 1)) * out_val.GetReal(1, 1), in_err.GetReal(1, 1));
}


