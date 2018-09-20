#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetSparseBinaryAffine.h"


inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetSparseBinaryAffineTest, testAffine)
{
	bb::NeuralNetSparseBinaryAffine<2> affine(2, 3);
	affine.SetBatchSize(1);
	testSetupLayerBuffer(affine);
	
	auto in_val = affine.GetInputSignalBuffer();
	auto out_val = affine.GetOutputSignalBuffer();

	affine.Forward();
	affine.Backward();
	affine.Update();
}


