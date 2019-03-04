#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/BinaryToReal.h"


TEST(BinaryToRealTest, testBinaryToReal)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int frame_size = 1;

    auto bin2real = bb::BinaryToReal<bb::Bit, float, float>::Create(bb::indices_t({node_size}), mux_size);
    
//	bb::NeuralNetBinaryToReal<> bin2real(node_size, node_size);
//	testSetupLayerBuffer(bin2real);

//	bin2real.SetMuxSize(mux_size);
//	bin2real.SetBatchSize(1);

    bb::FrameBuffer x_buf(BB_TYPE_BIT, frame_size*mux_size, node_size);

    bin2real->SetInputShape(x_buf.GetShape());

   	x_buf.SetBit(0, 0, true);
	x_buf.SetBit(1, 0, true);
	x_buf.SetBit(0, 1, false);
	x_buf.SetBit(1, 1, false);
	x_buf.SetBit(0, 2, false);
	x_buf.SetBit(1, 2, true);

    auto y_buf = bin2real->Forward(x_buf);

	EXPECT_EQ(2, x_buf.GetFrameSize());
	EXPECT_EQ(1, y_buf.GetFrameSize());

	EXPECT_EQ(1.0, y_buf.GetFP32(0, 0));
	EXPECT_EQ(0.0, y_buf.GetFP32(0, 1));
	EXPECT_EQ(0.5, y_buf.GetFP32(0, 2));
	

    // backward
//	auto out_err = bin2real.GetOutputErrorBuffer();
//	auto in_err = bin2real.GetInputErrorBuffer();

    bb::FrameBuffer dy_buf(BB_TYPE_FP32, frame_size, node_size);

	dy_buf.SetFP32(0, 0, 0.0f);
	dy_buf.SetFP32(0, 1, 1.0f);
	dy_buf.SetFP32(0, 2, 0.5f);

	auto dx_buf = bin2real->Backward(dy_buf);

	EXPECT_EQ(0.0f, dx_buf.GetFP32(0, 0));
	EXPECT_EQ(0.0f, dx_buf.GetFP32(1, 0));
	EXPECT_EQ(1.0f, dx_buf.GetFP32(0, 1));
	EXPECT_EQ(1.0f, dx_buf.GetFP32(1, 1));
	EXPECT_EQ(0.5f, dx_buf.GetFP32(0, 2));
	EXPECT_EQ(0.5f, dx_buf.GetFP32(1, 2));
}


