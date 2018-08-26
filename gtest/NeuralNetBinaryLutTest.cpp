#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <valarray>

#include "gtest/gtest.h"

#include "cereal/types/array.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/json.hpp"

#include "bb/NeuralNetAffine.h"
#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetSoftmax.h"
#include "bb/NeuralNetBinarize.h"
#include "bb/NeuralNetUnbinarize.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryLutN.h"
#include "bb/NeuralNetConvolution.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputValueBuffer (net.CreateInputValueBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputValueBuffer(net.CreateOutputValueBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}

template <typename T>
inline T vector_product(const std::vector<T>& vec)
{
	T prod = (T)1;
	for (auto v : vec) {
		prod *= v;
	}
	return prod;
}

template <typename T>
inline std::vector<T> vector_add(const std::vector<T>& vec0, const std::vector<T>& vec1)
{
	std::vector<T> vec(vec0.size());
	for (size_t i = 0; i < vec.size(); ++i ) {
		vec[i] = vec0[i] + vec1[i];
	}
	return vec;
}


TEST(NeuralNetBinaryLut, testNeuralNetBinaryLut6)
{
	bb::NeuralNetBinaryLut6<> lut(16, 2);
	testSetupLayerBuffer(lut);

	auto in_val = lut.GetInputValueBuffer();
	auto out_val = lut.GetOutputValueBuffer();
	in_val.SetBinary(0, 0, false);
	in_val.SetBinary(0, 1, true);
	in_val.SetBinary(0, 2, true);
	in_val.SetBinary(0, 3, false);
	in_val.SetBinary(0, 4, false);
	in_val.SetBinary(0, 5, true);
	in_val.SetBinary(0, 6, true);
	in_val.SetBinary(0, 7, true);

	// 0x1d
	lut.SetLutInput(0, 0, 6);	// 1
	lut.SetLutInput(0, 1, 4);	// 0
	lut.SetLutInput(0, 2, 1);	// 1
	lut.SetLutInput(0, 3, 7);	// 1
	lut.SetLutInput(0, 4, 2);	// 1
	lut.SetLutInput(0, 5, 3);	// 0

	// 0x1c
	lut.SetLutInput(1, 0, 0);	// 0
	lut.SetLutInput(1, 1, 4);	// 0
	lut.SetLutInput(1, 2, 1);	// 1
	lut.SetLutInput(1, 3, 5);	// 1
	lut.SetLutInput(1, 4, 6);	// 1
	lut.SetLutInput(1, 5, 3);	// 0

	for (int i = 0; i < 64; i++) {
		lut.SetLutTable(0, i, i == 0x1d);
		lut.SetLutTable(0, i, i != 0x1c);
	}
	
	lut.Forward();

	EXPECT_EQ(true, out_val.GetBinary(0, 0));
	EXPECT_EQ(false, out_val.Get<bool>(0, 1));
}



TEST(NeuralNetBinaryLut6, testNeuralNetBinaryLut6Batch)
{
	bb::NeuralNetBinaryLut6<> lut(16, 2, 2);
	testSetupLayerBuffer(lut);

	auto in_val = lut.GetInputValueBuffer();
	auto out_val = lut.GetOutputValueBuffer();
	in_val.SetBinary(0, 0, false);
	in_val.SetBinary(0, 1, true);
	in_val.SetBinary(0, 2, true);
	in_val.SetBinary(0, 3, false);
	in_val.SetBinary(0, 4, false);
	in_val.SetBinary(0, 5, true);
	in_val.SetBinary(0, 6, true);
	in_val.SetBinary(0, 7, true);

	in_val.SetBinary(1, 0, true);
	in_val.SetBinary(1, 1, false);
	in_val.SetBinary(1, 2, false);
	in_val.SetBinary(1, 3, true);
	in_val.SetBinary(1, 4, true);
	in_val.SetBinary(1, 5, false);
	in_val.SetBinary(1, 6, false);
	in_val.SetBinary(1, 7, false);

	// 0x1d
	lut.SetLutInput(0, 0, 6);	// 1
	lut.SetLutInput(0, 1, 4);	// 0
	lut.SetLutInput(0, 2, 1);	// 1
	lut.SetLutInput(0, 3, 7);	// 1
	lut.SetLutInput(0, 4, 2);	// 1
	lut.SetLutInput(0, 5, 3);	// 0

	// 0x1c
	lut.SetLutInput(1, 0, 0);	// 0
	lut.SetLutInput(1, 1, 4);	// 0
	lut.SetLutInput(1, 2, 1);	// 1
	lut.SetLutInput(1, 3, 5);	// 1
	lut.SetLutInput(1, 4, 6);	// 1
	lut.SetLutInput(1, 5, 3);	// 0

	for (int i = 0; i < 64; i++) {
		lut.SetLutTable(0, i, i == 0x1d || i == 0x22 );
		lut.SetLutTable(1, i, i != 0x1c && i != 0x23 );
	}

	lut.Forward();

	EXPECT_EQ(true, out_val.GetBinary(0, 0));
	EXPECT_EQ(false, out_val.Get<bool>(0, 1));
	EXPECT_EQ(true, out_val.GetBinary(1, 0));
	EXPECT_EQ(false, out_val.Get<bool>(1, 1));
}


TEST(NeuralNetBinaryLut, testNeuralNetBinaryLut6Compare)
{
	// 各数値を適当に設定
	const std::vector<size_t> in_index_size = { 3, 2, 1 };
	const std::vector<size_t> out_index_size = { 1, 3, 2 };
	const size_t input_node_size  = vector_product(in_index_size);
	const size_t output_node_size = vector_product(out_index_size);
	const size_t mux_size = 7;
	const size_t batch_size = 257;
	const size_t frame_size = mux_size * batch_size;
	const int lut_input_size = 6;
	const int lut_table_size = 64;

	std::mt19937_64	mt(123);
	std::uniform_int<size_t>	rand_input(0, input_node_size - 1);
	std::uniform_int<int>		rand_bin(0, 1);

	bb::NeuralNetBinaryLut6<>  lut0(input_node_size, output_node_size, 1);
	bb::NeuralNetBinaryLutN<6> lut1(input_node_size, output_node_size, 1);

	lut0.SetMuxSize(mux_size);
	lut1.SetMuxSize(mux_size);
	lut0.SetBatchSize(batch_size);
	lut1.SetBatchSize(batch_size);

#if 0

	// そのままのサイズ
	testSetupLayerBuffer(lut0);
	testSetupLayerBuffer(lut1);

	auto in_val0 = lut0.GetInputValueBuffer();
	auto in_val1 = lut1.GetInputValueBuffer();
	auto out_val0 = lut0.GetOutputValueBuffer();
	auto out_val1 = lut1.GetOutputValueBuffer();
	lut1.SetInputValueBuffer(in_val0);		// 入力バッファ共通化

#else

	// ROIテストのためサイズの異なるバッファを作る
	std::vector<size_t> in0_front_blank  = { 1, 2, 3 };
	std::vector<size_t> in0_back_blank   = { 9, 8, 7 };
	std::vector<size_t> in1_front_blank  = { 2, 0, 4 };
	std::vector<size_t> in1_back_blank   = { 7, 3, 0 };
	std::vector<size_t> out0_front_blank = { 2, 0, 4 };
	std::vector<size_t> out0_back_blank  = { 0, 5, 0 };
	std::vector<size_t> out1_front_blank = { 3, 0, 3 };
	std::vector<size_t> out1_back_blank  = { 0, 4, 5 };

	auto in0_addr_size  = vector_add(in_index_size, vector_add(in0_front_blank, in0_back_blank));
	auto in1_addr_size  = vector_add(in_index_size, vector_add(in1_front_blank, in1_back_blank));
	auto out0_addr_size = vector_add(out_index_size, vector_add(out0_front_blank, out0_back_blank));
	auto out1_addr_size = vector_add(out_index_size, vector_add(out1_front_blank, out1_back_blank));
	
	bb::NeuralNetBuffer<>	in_val0(frame_size, vector_product(in0_addr_size), lut0.GetInputValueDataType());
	bb::NeuralNetBuffer<>	in_val1(frame_size, vector_product(in1_addr_size), lut0.GetInputValueDataType());
	bb::NeuralNetBuffer<>	out_val0(frame_size, vector_product(out0_addr_size), lut0.GetInputValueDataType());
	bb::NeuralNetBuffer<>	out_val1(frame_size, vector_product(out1_addr_size), lut0.GetInputValueDataType());
	in_val0.SetDimensions(in0_addr_size);
	in_val1.SetDimensions(in1_addr_size);
	out_val0.SetDimensions(out0_addr_size);
	out_val1.SetDimensions(out1_addr_size);

	// ROIサイズは揃える
	in_val0.SetRoi(in0_front_blank, in_index_size);
	in_val1.SetRoi(in1_front_blank, in_index_size);
	out_val0.SetRoi(out0_front_blank, out_index_size);
	out_val1.SetRoi(out1_front_blank, out_index_size);

	// バッファ設定
	lut0.SetInputValueBuffer(in_val0);
	lut1.SetInputValueBuffer(in_val1);
	lut0.SetOutputValueBuffer(out_val0);
	lut1.SetOutputValueBuffer(out_val1);

#endif

	// 基本パラメータ確認
	EXPECT_EQ(input_node_size, lut0.GetInputNodeSize());
	EXPECT_EQ(output_node_size, lut0.GetOutputNodeSize());
	EXPECT_EQ(frame_size, lut0.GetInputFrameSize());
	EXPECT_EQ(frame_size, lut0.GetOutputFrameSize());
	EXPECT_EQ(lut_input_size, lut0.GetLutInputSize());
	EXPECT_EQ(lut_table_size, lut0.GetLutTableSize());
	EXPECT_EQ(lut0.GetInputNodeSize() , lut1.GetInputNodeSize());
	EXPECT_EQ(lut0.GetOutputNodeSize(), lut1.GetOutputNodeSize());
	EXPECT_EQ(lut0.GetInputFrameSize(), lut1.GetInputFrameSize());
	EXPECT_EQ(lut0.GetOutputFrameSize(), lut1.GetOutputFrameSize());
	EXPECT_EQ(lut0.GetLutInputSize(), lut1.GetLutInputSize());
	EXPECT_EQ(lut0.GetLutTableSize(), lut1.GetLutTableSize());

	EXPECT_EQ(lut0.GetInputNodeSize(), in_val0.GetNodeSize());
	EXPECT_EQ(lut0.GetInputNodeSize(), in_val0.GetNodeSize());
	EXPECT_EQ(lut1.GetInputNodeSize(), in_val1.GetNodeSize());
	EXPECT_EQ(lut1.GetInputNodeSize(), in_val1.GetNodeSize());
	EXPECT_EQ(lut0.GetOutputNodeSize(), out_val0.GetNodeSize());
	EXPECT_EQ(lut0.GetOutputNodeSize(), out_val0.GetNodeSize());
	EXPECT_EQ(lut1.GetOutputNodeSize(), out_val1.GetNodeSize());
	EXPECT_EQ(lut1.GetOutputNodeSize(), out_val1.GetNodeSize());

	// 設定
	for (size_t node = 0; node < output_node_size; ++node) {
		for (int lut_input = 0; lut_input < lut_input_size; ++lut_input) {
			size_t input_node = rand_input(mt);
			lut0.SetLutInput(node, lut_input, input_node);
			lut1.SetLutInput(node, lut_input, input_node);
		}

		for (int bit = 0; bit < lut_table_size; ++bit) {
			bool table_value = (rand_bin(mt) != 0);
			lut0.SetLutTable(node, bit, table_value);
			lut1.SetLutTable(node, bit, table_value);
		}
	}
	
	// データ設定
	for (size_t frame = 0; frame < frame_size; ++frame) {
		for (int node = 0; node < input_node_size; ++node) {
			bool input_value = (rand_bin(mt) != 0);
			in_val0.SetBinary(frame, node, input_value);
			in_val1.SetBinary(frame, node, input_value);
		}
	}

	// 出力バッファを壊しておく(Debug版だと同じ初期値が埋まるので)
	for (size_t frame = 0; frame < frame_size; ++frame) {
		for (int node = 0; node < output_node_size; ++node) {
			out_val0.SetBinary(frame, node, (rand_bin(mt) != 0));
			out_val1.SetBinary(frame, node, (rand_bin(mt) != 0));
		}
	}

	lut0.Forward();
	lut1.Forward();

	for (size_t frame = 0; frame < frame_size; ++frame) {
		for (int node = 0; node < output_node_size; ++node) {
			EXPECT_EQ(out_val0.GetBinary(frame, node), out_val1.GetBinary(frame, node));
		}
	}
}



TEST(NeuralNetBinaryLut, testNeuralNetBinaryLutFeedback)
{
	const size_t lut_input_size = 6;
	const size_t lut_table_size = (1 << lut_input_size);
	const size_t mux_size   = 3;
	const size_t batch_size = lut_table_size;
	const size_t frame_size = mux_size * batch_size;
	const size_t node_size = lut_table_size;
	bb::NeuralNetBinaryLutN<lut_input_size> lut(lut_input_size, lut_table_size);
	testSetupLayerBuffer(lut);

	lut.SetMuxSize(mux_size);
	lut.SetBatchSize(batch_size);

	auto in_val = lut.GetInputValueBuffer();
	auto out_val = lut.GetOutputValueBuffer();

	for (size_t frame = 0; frame < frame_size; frame++) {
		for (int bit = 0; bit < lut_input_size; bit++) {
			in_val.Set<bool>(frame, bit, (frame & ((size_t)1 << bit)) != 0);
		}
	}

	for (size_t node = 0; node < node_size; node++) {
		for (int i = 0; i < lut_input_size; i++) {
			lut.SetLutInput(node, i, i);
		}
	}

	lut.Forward();


	std::vector<double> vec_loss(frame_size);
	std::vector<float> vec_out(frame_size);
	for (int loop = 0; loop < 1; loop++) {
		do {
			for (size_t frame = 0; frame < frame_size; frame++) {
				vec_loss[frame] = 0;
				for (size_t node = 0; node < node_size; node++) {
					bool val = out_val.Get<bool>(frame, node);
					if (frame % node_size == node) {
						vec_loss[frame] += !val ? +1.0f : -1.0f;
					}
					else {
						vec_loss[frame] += val ? +0.01f : -0.01f;
					}
				}
			}
		} while (lut.Feedback(vec_loss));
		std::cout << loop << std::endl;
	}

	for (size_t node = 0; node < node_size; node++) {
		for (int i = 0; i < lut_table_size; i++) {
			EXPECT_EQ(node == i, lut.GetLutTable(node, i));
		}
	}
}

#if 0
TEST(NeuralNetBinaryLut, testNeuralNetBinaryLutFeedbackBitwise)
{
	const size_t lut_input_size = 6;
	const size_t lut_table_size = (1 << lut_input_size);
	const size_t mux_size = 3;
	const size_t batch_size = lut_table_size;
	const size_t frame_size = mux_size * batch_size;
	const size_t node_size = lut_table_size;
	bb::NeuralNetBinaryLutN<lut_input_size, true> lut(lut_input_size, lut_table_size);
	testSetupLayerBuffer(lut);

	lut.SetMuxSize(mux_size);
	lut.SetBatchSize(batch_size);

	auto in_val = lut.GetInputValueBuffer();
	auto out_val = lut.GetOutputValueBuffer();

	for (size_t frame = 0; frame < frame_size; frame++) {
		for (int bit = 0; bit < lut_input_size; bit++) {
			in_val.Set<bool>(frame, bit, (frame & ((size_t)1 << bit)) != 0);
		}
	}

	for (size_t node = 0; node < node_size; node++) {
		for (int i = 0; i < lut_input_size; i++) {
			lut.SetLutInput(node, i, i);
		}
	}

	lut.Forward();


	std::vector<double> vec_loss(frame_size);
	std::vector<float> vec_out(frame_size);
	for (int loop = 0; loop < 1; loop++) {
		do {
			for (size_t frame = 0; frame < frame_size; frame++) {
				vec_loss[frame] = 0;
				for (size_t node = 0; node < node_size; node++) {
					bool val = out_val.Get<bool>(frame, node);
					if (frame % node_size == node) {
						vec_loss[frame] += !val ? +1.0f : -1.0f;
					}
					else {
						vec_loss[frame] += val ? +0.01f : -0.01f;
					}
				}
			}
		} while (lut.Feedback(vec_loss));
		std::cout << loop << std::endl;
	}

	for (size_t node = 0; node < node_size; node++) {
		for (int i = 0; i < lut_table_size; i++) {
			EXPECT_EQ(node == i, lut.GetLutTable(node, i));
		}
	}
}

#endif


TEST(NeuralNetBinaryLut, testNeuralNetBinaryLutFeedbackSerialize)
{
	std::string fname("testNeuralNetBinaryLutFeedbackSerialize.json");
	size_t input_node_size  = 20;
	size_t output_node_size = 2;
	size_t lut_input_size = 6;
	size_t lut_table_size = 64;

	bb::NeuralNetBinaryLut6<>  lut0(input_node_size, output_node_size);
	lut0.SetLutInput(0, 0, 9);
	lut0.SetLutInput(0, 1, 8);
	lut0.SetLutInput(0, 2, 7);
	lut0.SetLutInput(0, 3, 6);
	lut0.SetLutInput(0, 4, 5);
	lut0.SetLutInput(0, 5, 4);
	lut0.SetLutInput(1, 0, 12);
	lut0.SetLutInput(1, 1, 18);
	lut0.SetLutInput(1, 2, 14);
	lut0.SetLutInput(1, 3, 16);
	lut0.SetLutInput(1, 4, 17);
	lut0.SetLutInput(1, 5, 14);

	lut0.SetLutTable(0, 0,  true);
	lut0.SetLutTable(0, 1,  false);
	lut0.SetLutTable(0, 2,  true);
	lut0.SetLutTable(0, 3,  true);
	lut0.SetLutTable(0, 63, false);
	lut0.SetLutTable(0, 62, false);
	lut0.SetLutTable(0, 61, true);
	lut0.SetLutTable(0, 60, true);

	lut0.SetLutTable(1, 0, false);
	lut0.SetLutTable(1, 1, false);
	lut0.SetLutTable(1, 2, true);
	lut0.SetLutTable(1, 3, false);
	lut0.SetLutTable(1, 63, true);
	lut0.SetLutTable(1, 62, false);
	lut0.SetLutTable(1, 61, false);
	lut0.SetLutTable(1, 60, false);

	// save
	{
		std::ofstream ofs(fname);
		cereal::JSONOutputArchive o_archive(ofs);
//		o_archive(cereal::make_nvp("layer_lut", lut0));
		lut0.Save(o_archive);
	}


	// load
	bb::NeuralNetBinaryLutN<6> lut1(input_node_size, output_node_size);
	{
		std::ifstream ifs(fname);
		cereal::JSONInputArchive i_archive(ifs);
	//	i_archive(cereal::make_nvp("layer_lut", lut1));
		lut1.Load(i_archive);
	}

	// compare
	for (size_t node = 0; node < output_node_size; ++node) {
		for (int i = 0; i < lut_input_size; ++i) {
			EXPECT_EQ(lut0.GetLutInput(node, i), lut1.GetLutInput(node, i));
		}

		for (int i = 0; i < lut_table_size; ++i) {
			EXPECT_EQ(lut0.GetLutTable(node, i), lut1.GetLutTable(node, i));
		}
	}

	EXPECT_EQ(lut1.GetLutInput(0,  0), 9);
	EXPECT_EQ(lut1.GetLutInput(0,  1), 8);
	EXPECT_EQ(lut1.GetLutInput(0,  2), 7);
	EXPECT_EQ(lut1.GetLutInput(0,  3), 6);
	EXPECT_EQ(lut1.GetLutInput(0,  4), 5);
	EXPECT_EQ(lut1.GetLutInput(0,  5), 4);
	EXPECT_EQ(lut1.GetLutInput(1,  0), 12);
	EXPECT_EQ(lut1.GetLutInput(1,  1), 18);
	EXPECT_EQ(lut1.GetLutInput(1,  2), 14);
	EXPECT_EQ(lut1.GetLutInput(1,  3), 16);
	EXPECT_EQ(lut1.GetLutInput(1,  4), 17);
	EXPECT_EQ(lut1.GetLutInput(1,  5), 14);
	EXPECT_EQ(lut1.GetLutTable(0,  0), true);
	EXPECT_EQ(lut1.GetLutTable(0,  1), false);
	EXPECT_EQ(lut1.GetLutTable(0,  2), true);
	EXPECT_EQ(lut1.GetLutTable(0,  3), true);
	EXPECT_EQ(lut1.GetLutTable(0, 63), false);
	EXPECT_EQ(lut1.GetLutTable(0, 62), false);
	EXPECT_EQ(lut1.GetLutTable(0, 61), true);
	EXPECT_EQ(lut1.GetLutTable(0, 60), true);
	EXPECT_EQ(lut1.GetLutTable(1,  0), false);
	EXPECT_EQ(lut1.GetLutTable(1,  1), false);
	EXPECT_EQ(lut1.GetLutTable(1,  2), true);
	EXPECT_EQ(lut1.GetLutTable(1,  3), false);
	EXPECT_EQ(lut1.GetLutTable(1, 63), true);
	EXPECT_EQ(lut1.GetLutTable(1, 62), false);
	EXPECT_EQ(lut1.GetLutTable(1, 61), false);
	EXPECT_EQ(lut1.GetLutTable(1, 60), false);
}

