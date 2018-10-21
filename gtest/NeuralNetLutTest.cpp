#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetLut.h"
#include "bb/NeuralNetOptimizerSgd.h"


inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetLutTest, testLut)
{
	bb::NeuralNetLut<> lut(2, 3);

	lut.SetBatchSize(12);
	testSetupLayerBuffer(lut);

	auto in_val = lut.GetInputSignalBuffer();
	auto out_val = lut.GetOutputSignalBuffer();

	lut.SetOptimizer(&bb::NeuralNetOptimizerSgd<>(0.01f));

	lut.Forward();
	auto in_err = lut.GetInputErrorBuffer();
	auto out_err = lut.GetOutputErrorBuffer();

	lut.Backward();
	lut.Update();
}

