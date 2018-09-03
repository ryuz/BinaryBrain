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

	in_val.SetDimensions({ 3, 3, 1 });
	EXPECT_EQ(0.1f, in_val.GetReal(0, { 0 , 0, 0 }));
	EXPECT_EQ(0.2f, in_val.GetReal(0, { 1 , 0, 0 }));
	EXPECT_EQ(0.3f, in_val.GetReal(0, { 2 , 0, 0 }));
	EXPECT_EQ(0.4f, in_val.GetReal(0, { 0 , 1, 0 }));
	EXPECT_EQ(0.5f, in_val.GetReal(0, { 1 , 1, 0 }));
	EXPECT_EQ(0.6f, in_val.GetReal(0, { 2 , 1, 0 }));
	EXPECT_EQ(0.7f, in_val.GetReal(0, { 0 , 2, 0 }));
	EXPECT_EQ(0.8f, in_val.GetReal(0, { 1 , 2, 0 }));
	EXPECT_EQ(0.9f, in_val.GetReal(0, { 2 , 2, 0 }));

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
	
//	std::cout << exp00 << std::endl;
//	std::cout << exp01 << std::endl;
//	std::cout << exp10 << std::endl;
//	std::cout << exp11 << std::endl;

	EXPECT_EQ(exp00, out_val.GetReal(0, 0));
	EXPECT_EQ(exp01, out_val.GetReal(0, 1));
	EXPECT_EQ(exp10, out_val.GetReal(0, 2));
	EXPECT_EQ(exp11, out_val.GetReal(0, 3));
}



TEST(NeuralNetConvolutionTest, testNeuralNetConvolution2)
{
	bb::NeuralNetConvolution<> cnv(3, 4, 5, 2, 3, 3);
	cnv.SetBatchSize(2);

	testSetupLayerBuffer(cnv);
	auto in_val = cnv.GetInputValueBuffer();
	auto out_val = cnv.GetOutputValueBuffer();

	in_val.SetDimensions({ 5, 4, 3 });
	int index = 0;
	for (size_t f = 0; f < 2; ++f) {
		for (size_t c = 0; c < 3; ++c) {
			for (size_t y = 0; y < 4; ++y) {
				for (size_t x = 0; x < 5; ++x) {
					in_val.SetReal(f, { x, y, c }, (float)(index++));
				}
			}
		}
	}
	EXPECT_EQ(120, index);

	EXPECT_EQ(0, in_val.GetReal(0, { 0, 0, 0 }));
	EXPECT_EQ(23, in_val.GetReal(0, { 3, 0, 1 }));
	EXPECT_EQ(71, in_val.GetReal(1, { 1, 2, 0 }));
	EXPECT_EQ(116, in_val.GetReal(1, { 1, 3, 2 }));
	EXPECT_EQ(119, in_val.GetReal(1, { 4, 3, 2 }));


	index = 10;
	for (size_t n = 0; n < 2; ++n) {
		for (size_t c = 0; c < 3; ++c) {
			for (size_t y = 0; y < 3; ++y) {
				for (size_t x = 0; x < 3; ++x) {
					cnv.W(n, c, y, x) = (float)(index++);
				}
			}
		}
	}
	EXPECT_EQ(64, index);

	index = 100;
	for (size_t n = 0; n < 2; ++n) {
		cnv.b(n) = (float)(index++);
	}

	cnv.Forward();

	std::cout << out_val.GetReal(0, 0) << std::endl;
}

