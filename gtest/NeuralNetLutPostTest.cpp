#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetLutPost.h"
#include "bb/NeuralNetSparseAffine.h"


inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetLutPostTest, testNeuralNetLutPost)
{
	const int M         = 64;
	const int output_node_size = 17;
	const int input_node_size = output_node_size * M;
	const int frame_size = 37;

	bb::NeuralNetLutPost<M>			lut_post(output_node_size);
	bb::NeuralNetSparseAffine<M>	affine(input_node_size, output_node_size);

	lut_post.SetBatchSize(frame_size);
	affine.SetBatchSize(frame_size);
	testSetupLayerBuffer(lut_post);
	testSetupLayerBuffer(affine);

	std::mt19937_64	mt(1);
	std::uniform_int_distribution<int> rand_dist(-10, 10);

	// 入力設定
	{
		auto lut_in_sig_buf = lut_post.GetInputSignalBuffer();
		auto aff_in_sig_buf = affine.GetInputSignalBuffer();
		for (size_t frame = 0; frame < frame_size; ++frame) {
			for (size_t node = 0; node < input_node_size; ++node) {
				float sig = (float)rand_dist(mt);
				lut_in_sig_buf.SetReal(frame, node, sig);
				aff_in_sig_buf.SetReal(frame, node, sig);
			}
		}
	}

	// 接続を同一にする
	for (size_t node = 0; node < output_node_size; ++node) {
		for (int i = 0; i < M; ++i) {
			affine.SetNodeInput(node, i, node*M+i);
		}
	}

	// パラメーターを揃える
	for (size_t node = 0; node < output_node_size; ++node) {
		for (int i = 0; i < M; ++i) {
			float W = (float)rand_dist(mt);
			lut_post.W(node, i) = W;
			affine.W(node, i) = W;
		}
		float b = (float)rand_dist(mt);
		lut_post.b(node) = b;
		affine.b(node) = b;
	}

	lut_post.Forward();
	affine.Forward();

	// 結果比較
	{
		auto lut_out_sig_buf = lut_post.GetOutputSignalBuffer();
		auto aff_out_sig_buf = affine.GetOutputSignalBuffer();
		for (size_t frame = 0; frame < frame_size; ++frame) {
			for (size_t node = 0; node < output_node_size; ++node) {
				float sig = (float)rand_dist(mt);
				float lut_sig = lut_out_sig_buf.GetReal(frame, node);
				float aff_sig = aff_out_sig_buf.GetReal(frame, node);
				EXPECT_EQ(aff_sig, lut_sig);
			}
		}
	}


	// 誤差設定
	{
		auto lut_out_err_buf = lut_post.GetOutputErrorBuffer();
		auto aff_out_err_buf = affine.GetOutputErrorBuffer();
		for (size_t frame = 0; frame < frame_size; ++frame) {
			for (size_t node = 0; node < output_node_size; ++node) {
				float err = (float)rand_dist(mt);
				lut_out_err_buf.SetReal(frame, node, err);
				aff_out_err_buf.SetReal(frame, node, err);
			}
		}
	}

	lut_post.Backward();
	affine.Backward();

	// 結果比較
	{
		auto lut_in_err_buf = lut_post.GetInputErrorBuffer();
		auto aff_in_err_buf = affine.GetInputErrorBuffer();
		for (size_t frame = 0; frame < frame_size; ++frame) {
			for (size_t node = 0; node < output_node_size; ++node) {
				float sig = (float)rand_dist(mt);
				float lut_err = lut_in_err_buf.GetReal(frame, node);
				float aff_err = aff_in_err_buf.GetReal(frame, node);
				EXPECT_EQ(aff_err, lut_err);
			}
		}
	}
	
	// パラメーター勾配を比較
	for (size_t node = 0; node < output_node_size; ++node) {
		for (int i = 0; i < M; ++i) {
			float lut_dW = lut_post.dW(node, i);
			float aff_dW = affine.dW(node, i);
			EXPECT_EQ(lut_dW, aff_dW);
//			EXPECT_TRUE(abs(lut_dW - aff_dW) < 0.00001);
		}
		float lut_db = lut_post.db(node);
		float aff_db = affine.db(node);
		EXPECT_EQ(lut_db, aff_db);
//		EXPECT_TRUE(abs(lut_db - aff_db) < 0.00001);
	}

	lut_post.Update();
	affine.Update();

	// パラメーターを比較
	for (size_t node = 0; node < output_node_size; ++node) {
		for (int i = 0; i < M; ++i) {
			float lut_W = lut_post.W(node, i);
			float aff_W = affine.W(node, i);
			EXPECT_EQ(lut_W, aff_W);
//			EXPECT_TRUE(abs(lut_W - aff_W) < 0.00001);
		}
		float lut_b = lut_post.b(node);
		float aff_b = affine.b(node);
		EXPECT_EQ(lut_b, aff_b);
//		EXPECT_TRUE(abs(lut_b - aff_b) < 0.00001);
	}
}



