#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/RealLut4.h"



TEST(RealLut4, testRealLut4_test0)
{
    auto lut = bb::RealLut4<>::Create(1);
	
    bb::FrameBuffer x(BB_TYPE_FP32, 16, 4);
    lut->SetInputShape(x.GetShape());

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            float val = ((i >> j) & 1) ? 1.0f : 0.0f;
            x.SetFP32(i, j, val);
        }
    }

    {
        auto W_ptr = lut->lock_W();
        for (int i = 0; i < 16; ++i) {
            float val = (i == 2) ? 1.0f : 0.0f;
            W_ptr(0, i) = val;
        }
    }

	auto y = lut->Forward(x);

    for (int i = 0; i < 16; ++i) {
        float exp = (i == 2) ? 1.0f : 0.0f;
    	EXPECT_FLOAT_EQ(exp, y.GetFP32(i, 0));
    }
}

