#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Sigmoid.h"


TEST(SigmoidTest, testSigmoid)
{
  	auto sigmoid = bb::Sigmoid<>::Create();
	
    bb::FrameBuffer x_buf(BB_TYPE_FP32, 1, 2);

	x_buf.SetFP32(0, 0, 1);
	x_buf.SetFP32(0, 1, 2);

	auto y_buf = sigmoid->Forward(x_buf);

	EXPECT_EQ(1.0f / (1.0f + exp(-1.0f)), y_buf.GetFP32(0, 0));
	EXPECT_EQ(1.0f / (1.0f + exp(-2.0f)), y_buf.GetFP32(0, 1));

    bb::FrameBuffer dy_buf(BB_TYPE_FP32, 1, 2);

    dy_buf.SetFP32(0, 0, 2);
	dy_buf.SetFP32(0, 1, 3);
    
	auto dx_buf = sigmoid->Backward(dy_buf);
    
	EXPECT_EQ(dy_buf.GetFP32(0, 0) * (1.0f - y_buf.GetFP32(0, 0)) * y_buf.GetFP32(0, 0), dx_buf.GetFP32(0, 0));
	EXPECT_EQ(dy_buf.GetFP32(0, 1) * (1.0f - y_buf.GetFP32(0, 1)) * y_buf.GetFP32(0, 1), dx_buf.GetFP32(0, 1));
}


TEST(SigmoidTest, testSigmoidBatch)
{
    auto sigmoid = bb::Sigmoid<>::Create();
	
    // forward
    bb::FrameBuffer x_buf(BB_TYPE_FP32, 2, 2);
      
    x_buf.SetFP32(0, 0, 1);
	x_buf.SetFP32(1, 0, 2);
	x_buf.SetFP32(0, 1, 3);
	x_buf.SetFP32(1, 1, 4);

	auto y_buf = sigmoid->Forward(x_buf);

	EXPECT_EQ(1.0f / (1.0f + exp(-1.0f)), y_buf.GetFP32(0, 0));
	EXPECT_EQ(1.0f / (1.0f + exp(-2.0f)), y_buf.GetFP32(1, 0));
	EXPECT_EQ(1.0f / (1.0f + exp(-3.0f)), y_buf.GetFP32(0, 1));
	EXPECT_EQ(1.0f / (1.0f + exp(-4.0f)), y_buf.GetFP32(1, 1));



    // backward
    bb::FrameBuffer dy_buf(BB_TYPE_FP32, 2, 2);

	dy_buf.SetFP32(0, 0, 2);
	dy_buf.SetFP32(1, 0, 3);
	dy_buf.SetFP32(0, 1, 4);
	dy_buf.SetFP32(1, 1, -5);

	auto dx_buf = sigmoid->Backward(dy_buf);

	EXPECT_EQ(dy_buf.GetFP32(0, 0) * (1.0f - y_buf.GetFP32(0, 0)) * y_buf.GetFP32(0, 0), dx_buf.GetFP32(0, 0));
	EXPECT_EQ(dy_buf.GetFP32(1, 0) * (1.0f - y_buf.GetFP32(1, 0)) * y_buf.GetFP32(1, 0), dx_buf.GetFP32(1, 0));
	EXPECT_EQ(dy_buf.GetFP32(0, 1) * (1.0f - y_buf.GetFP32(0, 1)) * y_buf.GetFP32(0, 1), dx_buf.GetFP32(0, 1));
	EXPECT_EQ(dy_buf.GetFP32(1, 1) * (1.0f - y_buf.GetFP32(1, 1)) * y_buf.GetFP32(1, 1), dx_buf.GetFP32(1, 1));
}


