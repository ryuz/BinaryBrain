#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetAffine.h"
#include "bb/NeuralNetLutPre.h"


inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}

#if 1
TEST(NeuralNetLutPreTest, testNeuralNetLutPre)
{
	bb::NeuralNetLutPre<2, 1> affine(2, 3);
	testSetupLayerBuffer(affine);
	
	// 接続を通常Affineと同一にする
	for (size_t node = 0; node < affine.GetOutputNodeSize(); ++node) {
		for (int i = 0; i < affine.GetNodeInputSize(node); ++i) {
			affine.SetNodeInput(node, i, i);
		}
	}

	auto in_val = affine.GetInputSignalBuffer();
	auto out_val = affine.GetOutputSignalBuffer();

	auto p0 = (float*)in_val.GetPtr(0);
	auto p1 = (float*)in_val.GetPtr(1);

	in_val.SetReal(0, 0, 1);
	in_val.SetReal(0, 1, 2);
	EXPECT_EQ(1, in_val.GetReal(0, 0));
	EXPECT_EQ(2, in_val.GetReal(0, 1));

	affine.W(0, 0, 0) = 1;
	affine.W(0, 0, 1) = 2;
	affine.W(1, 0, 0) = 10;
	affine.W(1, 0, 1) = 20;
	affine.W(2, 0, 0) = 100;
	affine.W(2, 0, 1) = 200;
	affine.b(0, 0) = 1000;
	affine.b(1, 0) = 2000;
	affine.b(2, 0) = 3000;
	affine.Forward();
	EXPECT_EQ(1 * 1 + 2 * 2 + 1000, out_val.GetReal(0, 0));
	EXPECT_EQ(1 * 10 + 2 * 20 + 2000, out_val.GetReal(0, 1));
	EXPECT_EQ(1 * 100 + 2 * 200 + 3000, out_val.GetReal(0, 2));

	auto in_err = affine.GetInputErrorBuffer();
	auto out_err = affine.GetOutputErrorBuffer();
	out_err.SetReal(0, 0, 998);
	out_err.SetReal(0, 1, 2042);
	out_err.SetReal(0, 2, 3491);

	affine.Backward();
	EXPECT_EQ(370518, in_err.GetReal(0, 0));
	EXPECT_EQ(741036, in_err.GetReal(0, 1));

	EXPECT_EQ(998,  affine.dW(0, 0, 0));
	EXPECT_EQ(2042, affine.dW(1, 0, 0));
	EXPECT_EQ(3491, affine.dW(2, 0, 0));
	EXPECT_EQ(1996, affine.dW(0, 0, 1));
	EXPECT_EQ(4084, affine.dW(1, 0, 1));
	EXPECT_EQ(6982, affine.dW(2, 0, 1));
	
	affine.Update();
}
#endif



#if 0
TEST(NeuralNetSparseAffineTest, testAffineInput)
{
	bb::NeuralNetSparseAffine<2> affine(2, 3);
	testSetupLayerBuffer(affine);

	// 接続を変える
	affine.SetNodeInput(0, 0, 1);
	affine.SetNodeInput(0, 1, 0);
	affine.SetNodeInput(1, 0, 1);
	affine.SetNodeInput(1, 1, 0);
	affine.SetNodeInput(2, 0, 1);
	affine.SetNodeInput(2, 1, 0);

	auto in_val = affine.GetInputSignalBuffer();
	auto out_val = affine.GetOutputSignalBuffer();

	in_val.SetReal(0, 0, 2);
	in_val.SetReal(0, 1, 1);
	EXPECT_EQ(2, in_val.GetReal(0, 0));
	EXPECT_EQ(1, in_val.GetReal(0, 1));

	affine.W(0, 0) = 1;
	affine.W(1, 0) = 2;
	affine.W(0, 1) = 10;
	affine.W(1, 1) = 20;
	affine.W(0, 2) = 100;
	affine.W(1, 2) = 200;
	affine.b(0) = 1000;
	affine.b(1) = 2000;
	affine.b(2) = 3000;
	affine.Forward();
	EXPECT_EQ(1 * 1   + 2 * 2 + 1000, out_val.GetReal(0, 0));
	EXPECT_EQ(1 * 10  + 2 * 20 + 2000, out_val.GetReal(0, 1));
	EXPECT_EQ(1 * 100 + 2 * 200 + 3000, out_val.GetReal(0, 2));

	auto in_err = affine.GetInputErrorBuffer();
	auto out_err = affine.GetOutputErrorBuffer();
	out_err.SetReal(0, 0, 998);
	out_err.SetReal(0, 1, 2042);
	out_err.SetReal(0, 2, 3491);

	affine.Backward();
	EXPECT_EQ(370518, in_err.GetReal(0, 1));
	EXPECT_EQ(741036, in_err.GetReal(0, 0));

	EXPECT_EQ(998,  affine.dW(0, 0));
	EXPECT_EQ(2042, affine.dW(0, 1));
	EXPECT_EQ(3491, affine.dW(0, 2));
	EXPECT_EQ(1996, affine.dW(1, 0));
	EXPECT_EQ(4084, affine.dW(1, 1));
	EXPECT_EQ(6982, affine.dW(1, 2));

	affine.Update();
}
#endif


#if 0
TEST(NeuralNetSparseAffineTest, testAffineCompare)
{
	bb::NeuralNetAffine<>	  	  affineOrg(6, 5);
	bb::NeuralNetSparseAffine<6>  affineLim(6, 5);

	affineOrg.SetBatchSize(23);
	affineOrg.SetBatchSize(23);

	testSetupLayerBuffer(affineOrg);
	testSetupLayerBuffer(affineLim);
	
	// 制限付きAffineの接続を通常Affineと同一にする
	for (size_t node = 0; node < affineLim.GetOutputNodeSize(); ++node) {
		for (int i = 0; i < affineLim.GetNodeInputSize(node); ++i) {
			affineLim.SetNodeInput(node, i, i);
		}
	}

	auto org_in_val = affineOrg.GetInputSignalBuffer();
	auto org_out_val = affineOrg.GetOutputSignalBuffer();
	auto org_in_err = affineOrg.GetInputErrorBuffer();
	auto org_out_err = affineOrg.GetOutputErrorBuffer();

	auto lim_in_val = affineLim.GetInputSignalBuffer();
	auto lim_out_val = affineLim.GetOutputSignalBuffer();
	auto lim_in_err = affineLim.GetInputErrorBuffer();
	auto lim_out_err = affineLim.GetOutputErrorBuffer();
	
	auto input_size = affineOrg.GetInputNodeSize();
	auto node_size = affineOrg.GetOutputNodeSize();
	auto frame_size = affineOrg.GetOutputNodeSize();

	std::mt19937_64 mt(1);
	std::uniform_real_distribution<float> uniform_dist(0, 1);
	std::normal_distribution<float> normal_dist(0, 1);

	// 内部係数を同じ乱数で統一
	for (size_t node = 0; node < node_size; ++node) {
		for (size_t input = 0; input < input_size; ++input) {
			float r = normal_dist(mt);
			affineOrg.W(input, node) = r;
			affineLim.W(input, node) = r;
		}
		float r = normal_dist(mt);
		affineOrg.b(node) = r;
		affineLim.b(node) = r;
	}

	for (int loop = 0; loop < 3; ++loop)
	{
		// 入力データを同じ乱数で統一
		for (size_t input = 0; input < input_size; ++input) {
			for (size_t frame = 0; frame < frame_size; ++frame) {
				float r = uniform_dist(mt);
				org_in_val.SetReal(frame, input, r);
				lim_in_val.SetReal(frame, input, r);
			}
		}

		// forward
		affineOrg.Forward();
		affineLim.Forward();

		// 出力比較
		for (size_t node = 0; node < node_size; ++node) {
			for (size_t frame = 0; frame < frame_size; ++frame) {
				EXPECT_TRUE(abs(org_out_val.GetReal(frame, node) - lim_out_val.GetReal(frame, node)) < 0.00001);
			}
		}

		// 誤差入力データを同じ乱数で統一
		std::uniform_real_distribution<float> uniform_dist(0, 1);
		for (size_t node = 0; node < node_size; ++node) {
			for (size_t frame = 0; frame < frame_size; ++frame) {
				float r = normal_dist(mt);
				org_out_err.SetReal(frame, node_size, r);
				lim_out_err.SetReal(frame, node_size, r);
			}
		}

		// backward
		affineOrg.Backward();
		affineLim.Backward();

		// 誤差を比較
		for (size_t input = 0; input < input_size; ++input) {
			for (size_t frame = 0; frame < frame_size; ++frame) {
		//		std::cout << org_in_err.GetReal(frame, input) << std::endl;
		//		std::cout << lim_in_err.GetReal(frame, input) << std::endl;
		//		EXPECT_EQ(org_in_err.GetReal(frame, input), lim_in_err.GetReal(frame, input));
				EXPECT_TRUE(abs(org_in_err.GetReal(frame, input) - lim_in_err.GetReal(frame, input)) < 0.00001);
			}
		}

		// update
		affineOrg.Update();
		affineLim.Update();
		
		// 学習係数比較
		for (size_t node = 0; node < node_size; ++node) {
			for (size_t input = 0; input < input_size; ++input) {
	//			EXPECT_EQ(affineOrg.W(input, node), affineLim.W(input, node));
				EXPECT_TRUE(abs(affineOrg.W(input, node) - affineLim.W(input, node)) < 0.00001);
			}
	//		EXPECT_EQ(affineOrg.b(node), affineLim.b(node));
			EXPECT_TRUE(abs(affineOrg.b(node) - affineLim.b(node)) < 0.00001);
		}
	}
}





#if 0 

TEST(NeuralNetAffineTest, testAffineBatch)
{
	bb::NeuralNetAffine<> affine(2, 3);
	testSetupLayerBuffer(affine);

	affine.SetBatchSize(2);

	auto in_val = affine.GetInputSignalBuffer();
	auto out_val = affine.GetOutputSignalBuffer();
	in_val.SetReal(0, 0, 1);
	in_val.SetReal(0, 1, 2);
	in_val.SetReal(1, 0, 3);
	in_val.SetReal(1, 1, 4);


	affine.W(0, 0) = 1;
	affine.W(1, 0) = 2;
	affine.W(0, 1) = 10;
	affine.W(1, 1) = 20;
	affine.W(0, 2) = 100;
	affine.W(1, 2) = 200;
	affine.b(0) = 1000;
	affine.b(1) = 2000;
	affine.b(2) = 3000;
	affine.Forward();
		
	EXPECT_EQ(1 * 1 + 2 * 2 + 1000, out_val.GetReal(0, 0));
	EXPECT_EQ(1 * 10 + 2 * 20 + 2000, out_val.GetReal(0, 1));
	EXPECT_EQ(1 * 100 + 2 * 200 + 3000, out_val.GetReal(0, 2));
	EXPECT_EQ(3 * 1 + 4 * 2 + 1000, out_val.GetReal(1, 0));
	EXPECT_EQ(3 * 10 + 4 * 20 + 2000, out_val.GetReal(1, 1));
	EXPECT_EQ(3 * 100 + 4 * 200 + 3000, out_val.GetReal(1, 2));
	

	auto out_err = affine.GetOutputErrorBuffer();
	auto in_err = affine.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 998);
	out_err.SetReal(1, 0, 1004);
	out_err.SetReal(0, 1, 2042);
	out_err.SetReal(1, 1, 2102);
	out_err.SetReal(0, 2, 3491);
	out_err.SetReal(1, 2, 4091);

	affine.Backward();

	EXPECT_EQ(370518, in_err.GetReal(0, 0));
	EXPECT_EQ(741036, in_err.GetReal(0, 1));
	EXPECT_EQ(431124, in_err.GetReal(1, 0));
	EXPECT_EQ(862248, in_err.GetReal(1, 1));
}


#endif

#endif