#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetDenseAffine.h"
#include "bb/NeuralNetSparseAffine.h"


inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetSparseAffineTest, testAffine)
{
	bb::NeuralNetSparseAffine<2> affine(2, 3);
	testSetupLayerBuffer(affine);
	
	// 接続を通常Affineと同一にする
	for (bb::INDEX node = 0; node < affine.GetOutputNodeSize(); ++node) {
		for (int i = 0; i < affine.GetNodeInputSize(node); ++i) {
			affine.SetNodeInput(node, i, i);
		}
	}

	auto in_val = affine.GetInputSignalBuffer();
	auto out_val = affine.GetOutputSignalBuffer();

	auto p0 = (float*)in_val.Lock(0);
	auto p1 = (float*)in_val.Lock(1);

	in_val.SetReal(0, 0, 1);
	in_val.SetReal(0, 1, 2);
	EXPECT_EQ(1, in_val.GetReal(0, 0));
	EXPECT_EQ(2, in_val.GetReal(0, 1));

	affine.W(0, 0) = 1;
	affine.W(0, 1) = 2;
	affine.W(1, 0) = 10;
	affine.W(1, 1) = 20;
	affine.W(2, 0) = 100;
	affine.W(2, 1) = 200;
	affine.b(0) = 1000;
	affine.b(1) = 2000;
	affine.b(2) = 3000;
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

	EXPECT_EQ(998,  affine.dW(0, 0));
	EXPECT_EQ(2042, affine.dW(1, 0));
	EXPECT_EQ(3491, affine.dW(2, 0));
	EXPECT_EQ(1996, affine.dW(0, 1));
	EXPECT_EQ(4084, affine.dW(1, 1));
	EXPECT_EQ(6982, affine.dW(2, 1));
	
	affine.Update();
}


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
	affine.W(0, 1) = 2;
	affine.W(1, 0) = 10;
	affine.W(1, 1) = 20;
	affine.W(2, 0) = 100;
	affine.W(2, 1) = 200;
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
	EXPECT_EQ(2042, affine.dW(1, 0));
	EXPECT_EQ(3491, affine.dW(2, 0));
	EXPECT_EQ(1996, affine.dW(0, 1));
	EXPECT_EQ(4084, affine.dW(1, 1));
	EXPECT_EQ(6982, affine.dW(2, 1));

	affine.Update();
}


TEST(NeuralNetSparseAffineTest, testAffineCompare)
{
	bb::NeuralNetDenseAffine<>	   dense_affine(6, 17);
	bb::NeuralNetSparseAffine<6>  sparse_affine(6, 17);

	dense_affine.SetBatchSize(67);
	sparse_affine.SetBatchSize(67);

	testSetupLayerBuffer(dense_affine);
	testSetupLayerBuffer(sparse_affine);
	
	// 制限付きAffineの接続を通常Affineと同一にする
	for (bb::INDEX node = 0; node < sparse_affine.GetOutputNodeSize(); ++node) {
		for (int i = 0; i < sparse_affine.GetNodeInputSize(node); ++i) {
			sparse_affine.SetNodeInput(node, i, i);
		}
	}

	auto dense_in_val = dense_affine.GetInputSignalBuffer();
	auto dense_out_val = dense_affine.GetOutputSignalBuffer();
	auto dense_in_err = dense_affine.GetInputErrorBuffer();
	auto dense_out_err = dense_affine.GetOutputErrorBuffer();

	auto sparse_in_val = sparse_affine.GetInputSignalBuffer();
	auto sparse_out_val = sparse_affine.GetOutputSignalBuffer();
	auto sparse_in_err = sparse_affine.GetInputErrorBuffer();
	auto sparse_out_err = sparse_affine.GetOutputErrorBuffer();
	
	auto input_size = dense_affine.GetInputNodeSize();
	auto node_size = dense_affine.GetOutputNodeSize();
	auto frame_size = dense_affine.GetOutputNodeSize();

	std::mt19937_64 mt(1);
	std::uniform_real_distribution<float> uniform_dist(0, 1);
	std::normal_distribution<float> normal_dist(0, 1);

	// 内部係数を同じ乱数で統一
	for (bb::INDEX node = 0; node < node_size; ++node) {
		for (bb::INDEX input = 0; input < input_size; ++input) {
			float r = normal_dist(mt);
			dense_affine.W(node, input) = r;
			sparse_affine.W(node, input) = r;
		}
		float r = normal_dist(mt);
		dense_affine.b(node) = r;
		sparse_affine.b(node) = r;
	}

	for (int loop = 0; loop < 3; ++loop)
	{
		// 入力データを同じ乱数で統一
		for (bb::INDEX input = 0; input < input_size; ++input) {
			for (bb::INDEX frame = 0; frame < frame_size; ++frame) {
				float r = uniform_dist(mt);
				dense_in_val.SetReal(frame, input, r);
				sparse_in_val.SetReal(frame, input, r);
			}
		}

		// forward
		dense_affine.Forward();
		sparse_affine.Forward();

		// 出力比較
		for (bb::INDEX node = 0; node < node_size; ++node) {
			for (bb::INDEX frame = 0; frame < frame_size; ++frame) {
				EXPECT_TRUE(abs(dense_out_val.GetReal(frame, node) - sparse_out_val.GetReal(frame, node)) < 0.00001);
			}
		}

		// 誤差入力データを同じ乱数で統一
		std::uniform_real_distribution<float> uniform_dist(0, 1);
		for (bb::INDEX node = 0; node < node_size; ++node) {
			for (bb::INDEX frame = 0; frame < frame_size; ++frame) {
				float r = normal_dist(mt);
				dense_out_err.SetReal(frame, node_size, r);
				sparse_out_err.SetReal(frame, node_size, r);
			}
		}

		// backward
		dense_affine.Backward();
		sparse_affine.Backward();

		// 誤差を比較
		for (bb::INDEX input = 0; input < input_size; ++input) {
			for (bb::INDEX frame = 0; frame < frame_size; ++frame) {
		//		std::cout << dense_in_err.GetReal(frame, input) << std::endl;
		//		std::cout << sparse_in_err.GetReal(frame, input) << std::endl;
		//		EXPECT_EQ(dense_in_err.GetReal(frame, input), sparse_in_err.GetReal(frame, input));
				EXPECT_TRUE(abs(dense_in_err.GetReal(frame, input) - sparse_in_err.GetReal(frame, input)) < 0.00001);
			}
		}

		// update
		dense_affine.Update();
		sparse_affine.Update();
		
		// 学習係数比較
		for (bb::INDEX node = 0; node < node_size; ++node) {
			for (bb::INDEX input = 0; input < input_size; ++input) {
	//			EXPECT_EQ(dense_affine.W(node, input), sparse_affine.W(node, input));
				EXPECT_TRUE(abs(dense_affine.W(node, input) - sparse_affine.W(node, input)) < 0.00001);
			}
	//		EXPECT_EQ(dense_affine.b(node), sparse_affine.b(node));
			EXPECT_TRUE(abs(dense_affine.b(node) - sparse_affine.b(node)) < 0.00001);
		}


		dense_affine.Backward();
		sparse_affine.Backward();
		dense_affine.Update();
		sparse_affine.Update();
		for (bb::INDEX node = 0; node < node_size; ++node) {
			for (bb::INDEX input = 0; input < input_size; ++input) {
				EXPECT_TRUE(abs(dense_affine.W(node, input) - sparse_affine.W(node, input)) < 0.00001);
			}
			EXPECT_TRUE(abs(dense_affine.b(node) - sparse_affine.b(node)) < 0.00001);
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
	affine.W(0, 1) = 2;
	affine.W(1, 0) = 10;
	affine.W(1, 1) = 20;
	affine.W(2, 0) = 100;
	affine.W(2, 1) = 200;
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
