#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetBatchNormalization.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}



TEST(NeuralNetBatchNormalizationTest, testBatchNormalization)
{
	bb::NeuralNetBatchNormalization<> batch_norm(2);
	batch_norm.SetBatchSize(8);
	testSetupLayerBuffer(batch_norm);
	
	auto in_val = batch_norm.GetInputSignalBuffer();
	auto out_val = batch_norm.GetOutputSignalBuffer();
	in_val.SetReal(0, 0, 1);
	in_val.SetReal(1, 0, 2);
	in_val.SetReal(2, 0, 3);
	in_val.SetReal(3, 0, 4);
	in_val.SetReal(4, 0, 5);
	in_val.SetReal(5, 0, 6);
	in_val.SetReal(6, 0, 7);
	in_val.SetReal(7, 0, 8);

	in_val.SetReal(0, 1, 10);
	in_val.SetReal(1, 1, 30);
	in_val.SetReal(2, 1, 20);
	in_val.SetReal(3, 1, 15);
	in_val.SetReal(4, 1, 11);
	in_val.SetReal(5, 1, 34);
	in_val.SetReal(6, 1, 27);
	in_val.SetReal(7, 1, 16);

	batch_norm.Forward(true);
//	batch_norm.Forward(false);

	/*
	[-1.52752510, -1.23359570],
	[-1.09108940, +1.14442010],
	[-0.65465360, -0.04458780],
	[-0.21821786, -0.63909180],
	[+0.21821786, -1.11469500],
	[+0.65465360, +1.62002340],
	[+1.09108940, +0.78771776],
	[+1.52752510, -0.52019095]
	*/

	EXPECT_TRUE(abs(out_val.GetReal(0, 0) - -1.52752510) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(1, 0) - -1.09108940) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(2, 0) - -0.65465360) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(3, 0) - -0.21821786) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(4, 0) - +0.21821786) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(5, 0) - +0.65465360) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(6, 0) - +1.09108940) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(7, 0) - +1.52752510) < 0.00001);

	EXPECT_TRUE(abs(out_val.GetReal(0, 1) - -1.23359570) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(1, 1) - +1.14442010) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(2, 1) - -0.04458780) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(3, 1) - -0.63909180) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(4, 1) - -1.11469500) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(5, 1) - +1.62002340) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(6, 1) - +0.78771776) < 0.00001);
	EXPECT_TRUE(abs(out_val.GetReal(7, 1) - -0.52019095) < 0.00001);



	auto out_err = batch_norm.GetOutputErrorBuffer();
	auto in_err = batch_norm.GetInputErrorBuffer();
	
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 8; j++) {
			out_err.SetReal(j, i, out_val.GetReal(j, i));
		}
	}

	out_err.SetReal(0, 0, 8);
	out_err.SetReal(1, 0, 6);
	out_err.SetReal(2, 0, 3);
	out_err.SetReal(3, 0, 4);
	out_err.SetReal(4, 0, 5);
	out_err.SetReal(5, 0, 4);
	out_err.SetReal(6, 0, 6);
	out_err.SetReal(7, 0, 1);
	out_err.SetReal(0, 1, 20);
	out_err.SetReal(1, 1, 70);
	out_err.SetReal(2, 1, 40);
	out_err.SetReal(3, 1, 15);
	out_err.SetReal(4, 1, 31);
	out_err.SetReal(5, 1, 54);
	out_err.SetReal(6, 1, 37);
	out_err.SetReal(7, 1, 26);

	batch_norm.Backward();

	/*
		[+0.65465380, +0.08798742],
		[+0.01558709, +2.05285700],
		[-1.05991530, +0.47591877],
		[-0.38967478, -1.50155930],
		[+0.28056574, +1.19688750],
		[+0.07793474, -0.64558935],
		[+1.18461110, -1.27384350],
		[-0.76376295, -0.39265870]]
	*/

	EXPECT_TRUE(abs(in_err.GetReal(0, 0) - +0.65465380) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(1, 0) - +0.01558709) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(2, 0) - -1.05991530) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(3, 0) - -0.38967478) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(4, 0) - +0.28056574) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(5, 0) - +0.07793474) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(6, 0) - +1.18461110) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(7, 0) - -0.76376295) < 0.00001);

	EXPECT_TRUE(abs(in_err.GetReal(0, 1) - +0.08798742) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(1, 1) - +2.05285700) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(2, 1) - +0.47591877) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(3, 1) - -1.50155930) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(4, 1) - +1.19688750) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(5, 1) - -0.64558935) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(6, 1) - -1.27384350) < 0.00001);
	EXPECT_TRUE(abs(in_err.GetReal(7, 1) - -0.39265870) < 0.00001);



}

