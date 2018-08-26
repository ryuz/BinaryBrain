#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"
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



TEST(NeuralNetBufferTest, testNeuralNetBufferTest)
{
	bb::NeuralNetBuffer<> buf(10, 2 * 3 * 4, BB_TYPE_REAL32);

	for (int i = 0; i < 2 * 3 * 4; ++i) {
		buf.Set<float>(0, i, (float)i);
	}

	// 多次元配列構成
	buf.SetDimensions({ 2, 3, 4 });
	EXPECT_EQ(0, *(float *)buf.GetPtr3(0, 0, 0));
	EXPECT_EQ(1, *(float *)buf.GetPtr3(0, 0, 1));
	EXPECT_EQ(2, *(float *)buf.GetPtr3(0, 1, 0));
	EXPECT_EQ(3, *(float *)buf.GetPtr3(0, 1, 1));
	EXPECT_EQ(4, *(float *)buf.GetPtr3(0, 2, 0));
	EXPECT_EQ(5, *(float *)buf.GetPtr3(0, 2, 1));
	EXPECT_EQ(6, *(float *)buf.GetPtr3(1, 0, 0));
	EXPECT_EQ(7, *(float *)buf.GetPtr3(1, 0, 1));
	EXPECT_EQ(8, *(float *)buf.GetPtr3(1, 1, 0));
	EXPECT_EQ(9, *(float *)buf.GetPtr3(1, 1, 1));
	EXPECT_EQ(10, *(float *)buf.GetPtr3(1, 2, 0));
	EXPECT_EQ(11, *(float *)buf.GetPtr3(1, 2, 1));
	EXPECT_EQ(12, *(float *)buf.GetPtr3(2, 0, 0));
	EXPECT_EQ(13, *(float *)buf.GetPtr3(2, 0, 1));
	EXPECT_EQ(14, *(float *)buf.GetPtr3(2, 1, 0));
	EXPECT_EQ(15, *(float *)buf.GetPtr3(2, 1, 1));
	EXPECT_EQ(16, *(float *)buf.GetPtr3(2, 2, 0));
	EXPECT_EQ(17, *(float *)buf.GetPtr3(2, 2, 1));
	EXPECT_EQ(18, *(float *)buf.GetPtr3(3, 0, 0));
	EXPECT_EQ(19, *(float *)buf.GetPtr3(3, 0, 1));
	EXPECT_EQ(20, *(float *)buf.GetPtr3(3, 1, 0));
	EXPECT_EQ(21, *(float *)buf.GetPtr3(3, 1, 1));
	EXPECT_EQ(22, *(float *)buf.GetPtr3(3, 2, 0));
	EXPECT_EQ(23, *(float *)buf.GetPtr3(3, 2, 1));

	// シーケンシャルアクセス確認
	{
		int i = 0;
		buf.ResetPtr();
		while (!buf.IsEnd()) {
			EXPECT_EQ((float)i, *(float *)buf.NextPtr());
			i++;
		}
		EXPECT_EQ(i, 24);
	}

	// オフセットのみのROI
	buf.SetRoi({ 0, 1, 0 });
	EXPECT_EQ(2, *(float *)buf.GetPtr3(0, 0, 0));
	EXPECT_EQ(3, *(float *)buf.GetPtr3(0, 0, 1));
	EXPECT_EQ(4, *(float *)buf.GetPtr3(0, 1, 0));
	EXPECT_EQ(5, *(float *)buf.GetPtr3(0, 1, 1));
	EXPECT_EQ(8, *(float *)buf.GetPtr3(1, 0, 0));
	EXPECT_EQ(9, *(float *)buf.GetPtr3(1, 0, 1));
	EXPECT_EQ(10, *(float *)buf.GetPtr3(1, 1, 0));
	EXPECT_EQ(11, *(float *)buf.GetPtr3(1, 1, 1));
	EXPECT_EQ(14, *(float *)buf.GetPtr3(2, 0, 0));
	EXPECT_EQ(15, *(float *)buf.GetPtr3(2, 0, 1));
	EXPECT_EQ(16, *(float *)buf.GetPtr3(2, 1, 0));
	EXPECT_EQ(17, *(float *)buf.GetPtr3(2, 1, 1));
	EXPECT_EQ(20, *(float *)buf.GetPtr3(3, 0, 0));
	EXPECT_EQ(21, *(float *)buf.GetPtr3(3, 0, 1));
	EXPECT_EQ(22, *(float *)buf.GetPtr3(3, 1, 0));
	EXPECT_EQ(23, *(float *)buf.GetPtr3(3, 1, 1));

	buf.ResetPtr();
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(2, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(3, *(float *)buf.NextPtr());
	EXPECT_EQ(4, *(float *)buf.NextPtr());
	EXPECT_EQ(5, *(float *)buf.NextPtr());
	EXPECT_EQ(8, *(float *)buf.NextPtr());
	EXPECT_EQ(9, *(float *)buf.NextPtr());
	EXPECT_EQ(10, *(float *)buf.NextPtr());
	EXPECT_EQ(11, *(float *)buf.NextPtr());
	EXPECT_EQ(14, *(float *)buf.NextPtr());
	EXPECT_EQ(15, *(float *)buf.NextPtr());
	EXPECT_EQ(16, *(float *)buf.NextPtr());
	EXPECT_EQ(17, *(float *)buf.NextPtr());
	EXPECT_EQ(20, *(float *)buf.NextPtr());
	EXPECT_EQ(21, *(float *)buf.NextPtr());
	EXPECT_EQ(22, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(23, *(float *)buf.NextPtr());
	EXPECT_EQ(true, buf.IsEnd());

	buf.SetRoi({ 0, 0, 2 });
	EXPECT_EQ(14, *(float *)buf.GetPtr3(0, 0, 0));
	EXPECT_EQ(15, *(float *)buf.GetPtr3(0, 0, 1));
	EXPECT_EQ(16, *(float *)buf.GetPtr3(0, 1, 0));
	EXPECT_EQ(17, *(float *)buf.GetPtr3(0, 1, 1));
	EXPECT_EQ(20, *(float *)buf.GetPtr3(1, 0, 0));
	EXPECT_EQ(21, *(float *)buf.GetPtr3(1, 0, 1));
	EXPECT_EQ(22, *(float *)buf.GetPtr3(1, 1, 0));
	EXPECT_EQ(23, *(float *)buf.GetPtr3(1, 1, 1));

	EXPECT_EQ(14, *(float *)buf.GetPtr(0));
	EXPECT_EQ(15, *(float *)buf.GetPtr(1));
	EXPECT_EQ(16, *(float *)buf.GetPtr(2));
	EXPECT_EQ(17, *(float *)buf.GetPtr(3));
	EXPECT_EQ(20, *(float *)buf.GetPtr(4));
	EXPECT_EQ(21, *(float *)buf.GetPtr(5));
	EXPECT_EQ(22, *(float *)buf.GetPtr(6));
	EXPECT_EQ(23, *(float *)buf.GetPtr(7));

	buf.ResetPtr();
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(14, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(15, *(float *)buf.NextPtr());
	EXPECT_EQ(16, *(float *)buf.NextPtr());
	EXPECT_EQ(17, *(float *)buf.NextPtr());
	EXPECT_EQ(20, *(float *)buf.NextPtr());
	EXPECT_EQ(21, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(22, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(23, *(float *)buf.NextPtr());
	EXPECT_EQ(true, buf.IsEnd());


	// ROI解除
	buf.ClearRoi();

	// シーケンシャルアクセス確認
	{
		int i = 0;
		buf.ResetPtr();
		while (!buf.IsEnd()) {
			EXPECT_EQ((float)i, *(float *)buf.NextPtr());
			i++;
		}
		EXPECT_EQ(i, 24);
	}


	// 範囲付きROI
	buf.SetRoi({ 0, 1, 1 }, { 1, 2, 2 });

	EXPECT_EQ(8, *(float *)buf.GetPtr(0));	// (1, 1, 0) : 8
	EXPECT_EQ(10, *(float *)buf.GetPtr(1));	// (1, 2, 0) : 8
	EXPECT_EQ(14, *(float *)buf.GetPtr(2));	// (2, 1, 0) : 14
	EXPECT_EQ(16, *(float *)buf.GetPtr(3));	// (2, 2, 0) : 16

	buf.ResetPtr();
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(8, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(10, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(14, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(16, *(float *)buf.NextPtr());
	EXPECT_EQ(true, buf.IsEnd());
}


TEST(NeuralNetBufferTest, testNeuralNetBufferTest2)
{
	bb::NeuralNetBuffer<> base_buf(2, 2 * 3 * 4, BB_TYPE_REAL32);
	
	// 入力データ作成
	for (size_t node = 0; node < 24; node++) {
		base_buf.SetReal(0, node, (float)node);
		base_buf.SetReal(1, node, (float)node + 1000);
	}
	
	auto buf = base_buf;
	buf.SetDimensions({ 4, 3, 2 });
	
	//  0  1  2  3
	//  4  5  6  7
	//  8  9 10 11
	//
	// 12 13 14 15
	// 16 17 18 19
	// 20 21 22 23

	
	buf.SetRoi({ 0, 0, 0 }, { 2, 2, 2 });

	EXPECT_EQ(0, ((float*)buf.GetPtr(0))[0]);
	EXPECT_EQ(1, ((float*)buf.GetPtr(1))[0]);
	EXPECT_EQ(4, ((float*)buf.GetPtr(2))[0]);
	EXPECT_EQ(5, ((float*)buf.GetPtr(3))[0]);
	EXPECT_EQ(12, ((float*)buf.GetPtr(4))[0]);
	EXPECT_EQ(13, ((float*)buf.GetPtr(5))[0]);
	EXPECT_EQ(16, ((float*)buf.GetPtr(6))[0]);
	EXPECT_EQ(17, ((float*)buf.GetPtr(7))[0]);

	EXPECT_EQ(0, buf.GetReal(0, 0));
	EXPECT_EQ(1, buf.GetReal(0, 1));
	EXPECT_EQ(4, buf.GetReal(0, 2));
	EXPECT_EQ(5, buf.GetReal(0, 3));
	EXPECT_EQ(12, buf.GetReal(0, 4));
	EXPECT_EQ(13, buf.GetReal(0, 5));
	EXPECT_EQ(16, buf.GetReal(0, 6));
	EXPECT_EQ(17, buf.GetReal(0, 7));

	EXPECT_EQ(0+1000, buf.GetReal(1, 0));
	EXPECT_EQ(1+1000, buf.GetReal(1, 1));
	EXPECT_EQ(4+1000, buf.GetReal(1, 2));
	EXPECT_EQ(5+1000, buf.GetReal(1, 3));
	EXPECT_EQ(12+1000, buf.GetReal(1, 4));
	EXPECT_EQ(13+1000, buf.GetReal(1, 5));
	EXPECT_EQ(16+1000, buf.GetReal(1, 6));
	EXPECT_EQ(17+1000, buf.GetReal(1, 7));

	buf.ClearRoi();
	buf.SetRoi({ 1, 0, 0 }, { 2, 2, 2 });
	EXPECT_EQ(1, buf.GetReal(0, 0));
	EXPECT_EQ(2, buf.GetReal(0, 1));
	EXPECT_EQ(5, buf.GetReal(0, 2));
	EXPECT_EQ(6, buf.GetReal(0, 3));
	EXPECT_EQ(13, buf.GetReal(0, 4));
	EXPECT_EQ(14, buf.GetReal(0, 5));
	EXPECT_EQ(17, buf.GetReal(0, 6));
	EXPECT_EQ(18, buf.GetReal(0, 7));

	buf.ClearRoi();
	buf.SetRoi({ 1, 1, 0 }, { 2, 2, 2 });
	EXPECT_EQ(5, buf.GetReal(0, 0));
	EXPECT_EQ(6, buf.GetReal(0, 1));
	EXPECT_EQ(9, buf.GetReal(0, 2));
	EXPECT_EQ(10, buf.GetReal(0, 3));
	EXPECT_EQ(17, buf.GetReal(0, 4));
	EXPECT_EQ(18, buf.GetReal(0, 5));
	EXPECT_EQ(21, buf.GetReal(0, 6));
	EXPECT_EQ(22, buf.GetReal(0, 7));

	buf.ClearRoi();
	buf.SetRoi({ 1, 1, 1 }, { 2, 2, 1 });
	EXPECT_EQ(17, buf.GetReal(0, 0));
	EXPECT_EQ(18, buf.GetReal(0, 1));
	EXPECT_EQ(21, buf.GetReal(0, 2));
	EXPECT_EQ(22, buf.GetReal(0, 3));


}


static void image_show(std::string name, bb::NeuralNetBuffer<> buf, size_t f, size_t h, size_t w)
{
	cv::Mat img((int)h, (int)w, CV_8U);
	for (size_t y = 0; y < h; y++) {
		for (size_t x = 0; x < w; x++) {
			img.at<uchar>((int)y, (int)x) = buf.GetBinary(f, y*w + x) ? 255 : 0;
		}
	}
	cv::imshow(name, img);
}

TEST(NeuralNetBufferTest, testNeuralNetBufferTest3)
{
	size_t input_c_size = 1;
	size_t input_h_size = 15;
	size_t input_w_size = 14;
	size_t output_c_size = 1;
	size_t output_h_size = 9;
	size_t output_w_size = 8;
	size_t y_step = 1;
	size_t x_step = 1;
	size_t filter_h_size = 5;
	size_t filter_w_size = 5;

	size_t input_node_size = input_c_size * input_h_size * input_w_size;
	size_t output_node_size = output_c_size * output_h_size * output_w_size;

	bb::NeuralNetBuffer<> in_buf(1, input_node_size, BB_TYPE_BINARY);
	bb::NeuralNetBuffer<> out_buf(1, output_node_size, BB_TYPE_BINARY);

	// 入力データ作成
	std::mt19937_64 mt(1);
	for (size_t node = 0; node < input_node_size; node++) {
		in_buf.SetBinary(0, node, mt() % 2 != 0);
	}

	auto in_val = in_buf;
	auto out_val = out_buf;
	in_val.SetDimensions({ input_w_size, input_h_size, input_c_size });
	out_val.SetDimensions({ output_w_size, output_h_size, output_c_size });

	size_t in_y = 0;
	for (size_t out_y = 0; out_y < output_h_size; out_y++) {
		size_t in_x = 0;
		for (size_t out_x = 0; out_x < output_w_size; out_x++) {
			in_val.ClearRoi();
			in_val.SetRoi({ in_x, in_y, 0}, { filter_w_size , filter_h_size , input_c_size });
			out_val.ClearRoi();
			out_val.SetRoi({ out_x, out_y, 0}, { 1, 1, output_c_size });

			out_val.SetBinary(0, 0, in_val.GetBinary(0, 0));
			in_x += x_step;
		}
		in_y += y_step;
	}

	for (size_t y = 0; y < output_h_size; y++) {
		for (size_t x = 0; x < output_w_size; x++) {
			EXPECT_EQ(in_buf.GetBinary(0, y*input_w_size+x), out_buf.GetBinary(0, y*output_w_size + x));
		}
	}

	//	image_show("out_buf", out_buf, 0, output_h_size, output_w_size);
	//	image_show("in_buf", in_buf, 0, input_h_size, input_w_size);
	//	cv::waitKey();
}


