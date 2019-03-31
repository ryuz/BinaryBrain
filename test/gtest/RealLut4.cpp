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


    bb::FrameBuffer dy(BB_TYPE_FP32, 16, 1);
    for (int i = 0; i < 16; ++i) {
        float val = (i == 2) ? -0.1f : 0.0f;
        dy.SetFP32(i, 0, val);
    }
	auto dx = lut->Backward(dy);




    {
        std::cout << "[W] : ";
        auto W_ptr = lut->lock_W_const();
        for (int i = 0; i < 16; ++i) {
            std::cout << W_ptr(0, i) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "[x] : " << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << i << " : ";
        for (int j = 0; j < 4; ++j) {
            std::cout << x.GetFP32(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "[y] : ";
    for (int i = 0; i < 16; ++i) {
        std::cout << y.GetFP32(i, 0) << " ";
    }
    std::cout << std::endl;

    std::cout << "[dy] : ";
    for (int i = 0; i < 16; ++i) {
        std::cout << dy.GetFP32(i, 0) << " ";
    }
    std::cout << std::endl;

    std::cout << "[dx] : " << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << i << " : ";
        for (int j = 0; j < 4; ++j) {
            std::cout << dx.GetFP32(i, j) << " ";
        }
        std::cout << std::endl;
    }

    {
        std::cout << "[dW] : ";
        auto dW_ptr = lut->lock_dW_const();
        for (int i = 0; i < 16; ++i) {
            std::cout << dW_ptr(0, i) << " ";
        }
        std::cout << std::endl;
    }


}

