#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/ReLU.h"



TEST(ReLUTest, testReLU_test0)
{
	auto relu = bb::ReLU<>::Create();
	
    bb::FrameBuffer x(BB_TYPE_FP32, 2, 3);

	x.SetFP32(0, 0, -1);
	x.SetFP32(0, 1, 0);
	x.SetFP32(0, 2, 1);
	x.SetFP32(1, 0, 2);
	x.SetFP32(1, 1, 1);
	x.SetFP32(1, 2, -2);
	auto y = relu->Forward(x);
	
	EXPECT_EQ(0, y.GetFP32(0, 0));
	EXPECT_EQ(0, y.GetFP32(0, 1));
	EXPECT_EQ(1, y.GetFP32(0, 2));
	EXPECT_EQ(2, y.GetFP32(1, 0));
	EXPECT_EQ(1, y.GetFP32(1, 1));
	EXPECT_EQ(0, y.GetFP32(1, 2));

	// backward
    bb::FrameBuffer dy(BB_TYPE_FP32, 2, 3);

	dy.SetFP32(0, 0, 1);
	dy.SetFP32(0, 1, 2);
	dy.SetFP32(0, 2, 3);
	dy.SetFP32(1, 0, 4);
	dy.SetFP32(1, 1, 5);
	dy.SetFP32(1, 2, 6);

	auto dx = relu->Backward(dy);

	EXPECT_EQ(0, dx.GetFP32(0, 0));
	
    EXPECT_TRUE(dx.GetFP32(0, 1) == 0 || dx.GetFP32(0, 1) == 2);    // 境界値は両方許容

	EXPECT_EQ(3, dx.GetFP32(0, 2));
	EXPECT_EQ(4, dx.GetFP32(1, 0));
	EXPECT_EQ(5, dx.GetFP32(1, 1));
	EXPECT_EQ(0, dx.GetFP32(1, 2));
}

