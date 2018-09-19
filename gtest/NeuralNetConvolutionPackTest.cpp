#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/NeuralNetAffine.h"
#include "bb/NeuralNetGroup.h"
#include "bb/NeuralNetConvolutionPack.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetConvolutionPackTest, testNeuralNetConvolutionPack)
{
	bb::NeuralNetAffine<>	sub_affine(1 * 2 * 2, 1);
	bb::NeuralNetGroup<>	sub_net;
	sub_net.AddLayer(&sub_affine);
	bb::NeuralNetConvolutionPack<>	cnv(&sub_net, 1, 3, 3, 1, 2, 2);

	cnv.SetBatchSize(1);
	testSetupLayerBuffer(cnv);
	auto in_val = cnv.GetInputSignalBuffer();
	auto out_val = cnv.GetOutputSignalBuffer();

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

//	cnv.W(0, 0, 0, 0) = 0.1f;
//	cnv.W(0, 0, 0, 1) = 0.2f;
//	cnv.W(0, 0, 1, 0) = 0.3f;
//	cnv.W(0, 0, 1, 1) = 0.4f;

//	cnv.b(0) = 0.321f;

	sub_affine.W(0, 0) = 0.1f;
	sub_affine.W(1, 0) = 0.2f;
	sub_affine.W(2, 0) = 0.3f;
	sub_affine.W(3, 0) = 0.4f;
	sub_affine.b(0) = 0.321f;


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

	EXPECT_TRUE(abs(exp00 - out_val.GetReal(0, 0)) < 0.000001);
	EXPECT_TRUE(abs(exp01 - out_val.GetReal(0, 1)) < 0.000001);
	EXPECT_TRUE(abs(exp10 - out_val.GetReal(0, 2)) < 0.000001);
	EXPECT_TRUE(abs(exp11 - out_val.GetReal(0, 3)) < 0.000001);
}


#if 1

TEST(NeuralNetConvolutionPackTest, testNeuralNetConvolutionPack2)
{
	bb::NeuralNetAffine<>	sub_affine(3 * 3 * 3, 2);
	bb::NeuralNetGroup<>	sub_net;
	sub_net.AddLayer(&sub_affine);
	bb::NeuralNetConvolutionPack<>	cnv(&sub_net, 3, 4, 5, 2, 3, 3);

//	bb::NeuralNetConvolution<> cnv(3, 4, 5, 2, 3, 3);
	cnv.SetMuxSize(1);
	cnv.SetBatchSize(2);

	testSetupLayerBuffer(cnv);
	auto in_val = cnv.GetInputSignalBuffer();
	auto out_val = cnv.GetOutputSignalBuffer();

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
//	for (size_t n = 0; n < 2; ++n) {
//		for (size_t c = 0; c < 3; ++c) {
//			for (size_t y = 0; y < 3; ++y) {
//				for (size_t x = 0; x < 3; ++x) {
//					cnv.W(n, c, y, x) = (float)(index++);
//				}
//			}
//		}
//	}
	for (size_t n = 0; n < 2; ++n) {
		for (size_t i = 0; i < 3 * 3 * 3; ++i) {
			sub_affine.W(i, n) = (float)(index++);
		}
	}

	EXPECT_EQ(64, index);

	index = 100;
	for (size_t n = 0; n < 2; ++n) {
		sub_affine.b(n) = (float)(index++);
	}

	cnv.Forward();
	
	EXPECT_EQ(19774, out_val.GetReal(0, 0));
	EXPECT_EQ(20395, out_val.GetReal(0, 1));
	EXPECT_EQ(21016, out_val.GetReal(0, 2));
	EXPECT_EQ(22879, out_val.GetReal(0, 3));
	EXPECT_EQ(23500, out_val.GetReal(0, 4));
	EXPECT_EQ(24121, out_val.GetReal(0, 5));
	EXPECT_EQ(38729, out_val.GetReal(0, 6));
	EXPECT_EQ(40079, out_val.GetReal(0, 7));
	EXPECT_EQ(41429, out_val.GetReal(0, 8));
	EXPECT_EQ(45479, out_val.GetReal(0, 9));
	EXPECT_EQ(46829, out_val.GetReal(0, 10));
	EXPECT_EQ(48179, out_val.GetReal(0, 11));
	EXPECT_EQ(57034, out_val.GetReal(1, 0));
	EXPECT_EQ(57655, out_val.GetReal(1, 1));
	EXPECT_EQ(58276, out_val.GetReal(1, 2));
	EXPECT_EQ(60139, out_val.GetReal(1, 3));
	EXPECT_EQ(60760, out_val.GetReal(1, 4));
	EXPECT_EQ(61381, out_val.GetReal(1, 5));
	EXPECT_EQ(119729, out_val.GetReal(1, 6));
	EXPECT_EQ(121079, out_val.GetReal(1, 7));
	EXPECT_EQ(122429, out_val.GetReal(1, 8));
	EXPECT_EQ(126479, out_val.GetReal(1, 9));
	EXPECT_EQ(127829, out_val.GetReal(1, 10));
	EXPECT_EQ(129179, out_val.GetReal(1, 11));
	

	// backward

	auto in_err = cnv.GetInputErrorBuffer();
	auto out_err = cnv.GetOutputErrorBuffer();

	in_err.SetDimensions({ 5, 4, 3 });
	out_err.SetDimensions({ 3, 2, 2 });

	index = 30;
	for (size_t f = 0; f < 2; ++f) {
		for (size_t c = 0; c < 2; ++c) {
			for (size_t y = 0; y < 2; ++y) {
				for (size_t x = 0; x < 3; ++x) {
					out_err.SetReal(f, { x, y, c }, (float)(index++));
				}
			}
		}
	}
	cnv.Backward();

#if 0
	for (size_t n = 0; n < 2; ++n) {
		for (size_t c = 0; c < 3; ++c) {
			for (size_t y = 0; y < 3; ++y) {
				for (size_t x = 0; x < 3; ++x) {
					std::cout << cnv.dW(n, c, y, x) << ", ";
				}
				std::cout << std::endl;
			}
		}
	}
#endif

	/*
	array([[[[17690., 18152., 18614.],
         [20000., 20462., 20924.],
         [22310., 22772., 23234.]],

        [[26930., 27392., 27854.],
         [29240., 29702., 30164.],
         [31550., 32012., 32474.]],

        [[36170., 36632., 37094.],
         [38480., 38942., 39404.],
         [40790., 41252., 41714.]]],


       [[[20102., 20636., 21170.],
         [22772., 23306., 23840.],
         [25442., 25976., 26510.]],

        [[30782., 31316., 31850.],
         [33452., 33986., 34520.],
         [36122., 36656., 37190.]],

        [[41462., 41996., 42530.],
         [44132., 44666., 45200.],
         [46802., 47336., 47870.]]]])
	 */
	
//	EXPECT_EQ(17690, sub_affine.dW(0, 0, 0, 0));
//	EXPECT_EQ(29702, sub_affine.dW(0, 1, 1, 1));
//	EXPECT_EQ(26510, sub_affine.dW(1, 0, 2, 2));
//	EXPECT_EQ(33452, sub_affine.dW(1, 1, 1, 0));
//	EXPECT_EQ(47870, sub_affine.dW(1, 2, 2, 2));

	EXPECT_EQ(462, sub_affine.db(0));
	EXPECT_EQ(534, sub_affine.db(1));

#if 0
	for (size_t f = 0; f < 2; ++f) {
		for (size_t c = 0; c < 3; ++c) {
			for (size_t y = 0; y < 4; ++y) {
				for (size_t x = 0; x < 5; ++x) {
					std::cout << in_err.GetReal(f, { x, y, c }) << ", ";
				}
				std::cout << std::endl;
			}
		}
	}
#endif

	/*
	array([[[[1632., 3377., 5237., 3611., 1866.],
		[3603., 7444., 11527., 7936., 4095.],
		[4017., 8284., 12805., 8800., 4533.],
		[2205., 4541., 7010., 4811., 2475.]],

		[[2226., 4583., 7073., 4853., 2496.],
		[4845., 9964., 15361., 10528., 5409.],
		[5259., 10804., 16639., 11392., 5847.],
		[2853., 5855., 9008., 6161., 3159.]],

		[[2820., 5789., 8909., 6095., 3126.],
		[6087., 12484., 19195., 13120., 6723.],
		[6501., 13324., 20473., 13984., 7161.],
		[3501., 7169., 11006., 7511., 3843.]]],


		[[[2196., 4529., 7001., 4811., 2478.],
		[4803., 9892., 15271., 10480., 5391.],
		[5361., 11020., 16981., 11632., 5973.],
		[2913., 5981., 9206., 6299., 3231.]],

		[[3006., 6167., 9485., 6485., 3324.],
		[6477., 13276., 20401., 13936., 7137.],
		[7035., 14404., 22111., 15088., 7719.],
		[3777., 7727., 11852., 8081., 4131.]],

		[[3816., 7805., 11969., 8159., 4170.],
		[8151., 16660., 25531., 17392., 8883.],
		[8709., 17788., 27241., 18544., 9465.],
		[4641., 9473., 14498., 9863., 5031.]]]])
	*/

	EXPECT_EQ(1632, in_err.GetReal(0, { 0, 0, 0 }));
	EXPECT_EQ(7444, in_err.GetReal(0, { 1, 1, 0 }));
	EXPECT_EQ(3816,  in_err.GetReal(1, { 0, 0, 2 }));
	EXPECT_EQ(5031, in_err.GetReal(1, { 4, 3, 2 }));

}

#endif
