#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetSparseBinaryAffine.h"


inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputValueBuffer (net.CreateInputValueBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputValueBuffer(net.CreateOutputValueBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetSparseBinaryAffineTest, testAffine)
{
	bb::NeuralNetSparseBinaryAffine<2> affine(2, 3);
	testSetupLayerBuffer(affine);
	
	auto in_val = affine.GetInputValueBuffer();
	auto out_val = affine.GetOutputValueBuffer();

	affine.Forward();
	affine.Backward();
	affine.Update(0.1);
}


