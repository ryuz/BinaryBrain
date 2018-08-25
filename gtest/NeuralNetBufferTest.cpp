#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
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
	bb::NeuralNetBuffer<> buf(10, 2*3*4, BB_TYPE_REAL32);
	
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
	EXPECT_EQ(2,  *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(3,  *(float *)buf.NextPtr());
	EXPECT_EQ(4,  *(float *)buf.NextPtr());
	EXPECT_EQ(5,  *(float *)buf.NextPtr());
	EXPECT_EQ(8,  *(float *)buf.NextPtr());
	EXPECT_EQ(9,  *(float *)buf.NextPtr());
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
	bb::NeuralNetBuffer<> buf(10, 2 * 3 * 4, BB_TYPE_REAL32);

	for (int i = 0; i < 2 * 3 * 4; ++i) {
		buf.Set<float>(0, i, (float)i);
	}


}
