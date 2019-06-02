#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Binarize.h"


TEST(BinarizeTest, testBinarize_test0)
{
    auto bin = bb::Binarize<>::Create();
    
    bb::FrameBuffer x_buf(BB_TYPE_FP32, 2, 3);
    bin->SetInputShape(x_buf.GetShape());

    x_buf.SetFP32(0, 0, -1.01f);
    x_buf.SetFP32(0, 1, -0.9f); 
    x_buf.SetFP32(0, 2, -0.5f); 
    x_buf.SetFP32(1, 0, -0.1f); 
    x_buf.SetFP32(1, 1, +0.9f); 
    x_buf.SetFP32(1, 2, +1.01f);

    auto y_buf = bin->Forward(x_buf);
    
    EXPECT_EQ(0, y_buf.GetFP32(0, 0));
    EXPECT_EQ(0, y_buf.GetFP32(0, 1));
    EXPECT_EQ(0, y_buf.GetFP32(0, 2));
    EXPECT_EQ(0, y_buf.GetFP32(1, 0));
    EXPECT_EQ(1, y_buf.GetFP32(1, 1));
    EXPECT_EQ(1, y_buf.GetFP32(1, 2));

    // backward
    bb::FrameBuffer dy_buf(BB_TYPE_FP32, 2, 3);
    dy_buf.SetFP32(0, 0, 1);
    dy_buf.SetFP32(0, 1, 2);
    dy_buf.SetFP32(0, 2, 3);
    dy_buf.SetFP32(1, 0, 4);
    dy_buf.SetFP32(1, 1, 5);
    dy_buf.SetFP32(1, 2, 6);

    auto dx_buf = bin->Backward(dy_buf);

    EXPECT_EQ(0, dx_buf.GetFP32(0, 0));
    EXPECT_EQ(2, dx_buf.GetFP32(0, 1));   // 未定義
    EXPECT_EQ(3, dx_buf.GetFP32(0, 2));
    EXPECT_EQ(4, dx_buf.GetFP32(1, 0));
    EXPECT_EQ(5, dx_buf.GetFP32(1, 1));   // 未定義
    EXPECT_EQ(0, dx_buf.GetFP32(1, 2));
}


TEST(BinarizeTest, testBinarize_bit_test)
{
    auto bin = bb::Binarize<bb::Bit>::Create();
    
    bb::FrameBuffer x_buf(BB_TYPE_FP32, 3, 2);
    bin->SetInputShape(x_buf.GetShape());

    x_buf.SetFP32(0, 0, -1.01f);
    x_buf.SetFP32(1, 0, +0.99f); 
    x_buf.SetFP32(2, 0, -0.5f); 
    x_buf.SetFP32(0, 1, -0.99f); 
    x_buf.SetFP32(1, 1,  0.01f); 
    x_buf.SetFP32(2, 1, +1.01f);

    auto y_buf = bin->Forward(x_buf);
    
    EXPECT_EQ(0, (int)y_buf.GetBit(0, 0));
    EXPECT_EQ(1, (int)y_buf.GetBit(1, 0));
    EXPECT_EQ(0, (int)y_buf.GetBit(2, 0));
    EXPECT_EQ(0, (int)y_buf.GetBit(0, 1));
    EXPECT_EQ(1, (int)y_buf.GetBit(1, 1));
    EXPECT_EQ(1, (int)y_buf.GetBit(2, 1));

    // backward
    bb::FrameBuffer dy_buf(BB_TYPE_FP32, 3, 2);
    dy_buf.SetFP32(0, 0, 1);
    dy_buf.SetFP32(1, 0, 2);
    dy_buf.SetFP32(2, 0, 3);
    dy_buf.SetFP32(0, 1, 4);
    dy_buf.SetFP32(1, 1, 5);
    dy_buf.SetFP32(2, 1, 6);

    auto dx_buf = bin->Backward(dy_buf);

    EXPECT_EQ(0, dx_buf.GetFP32(0, 0)); // 0
    EXPECT_EQ(2, dx_buf.GetFP32(1, 0));
    EXPECT_EQ(3, dx_buf.GetFP32(2, 0));
    EXPECT_EQ(4, dx_buf.GetFP32(0, 1));
    EXPECT_EQ(5, dx_buf.GetFP32(1, 1));
    EXPECT_EQ(0, dx_buf.GetFP32(2, 1)); // 0
}



#if 0 

TEST(BinarizeTest, testBinarize_comp)
{
    int const node_size = 10;
    int const frame_size = 10;

    auto bin_cpu = bb::Binarize<>::Create();
    auto bin_gpu = bb::Binarize<>::Create();
    
    bb::FrameBuffer x_cpu(BB_TYPE_FP32, 2, 3, true);
    bb::FrameBuffer x_gpu(BB_TYPE_FP32, 2, 3);

    x_cpu.SetFP32(0, 0, -1);    x_gpu.SetFP32(0, 0, -1);
    x_cpu.SetFP32(0, 1, 0);     x_gpu.SetFP32(0, 1, 0);
    x_cpu.SetFP32(0, 2, 1);     x_gpu.SetFP32(0, 2, 1);
    x_cpu.SetFP32(1, 0, 2);     x_gpu.SetFP32(1, 0, 2);
    x_cpu.SetFP32(1, 1, 1);     x_gpu.SetFP32(1, 1, 1);
    x_cpu.SetFP32(1, 2, -2);    x_gpu.SetFP32(1, 2, -2);

    auto y_cpu = bin_cpu->Forward(x_cpu);
    auto y_gpu = bin_cpu->Forward(x_gpu);
    
    EXPECT_EQ(0, y_cpu.GetFP32(0, 0));
    EXPECT_EQ(0, y_cpu.GetFP32(0, 1));
    EXPECT_EQ(1, y_cpu.GetFP32(0, 2));
    EXPECT_EQ(1, y_cpu.GetFP32(1, 0));
    EXPECT_EQ(1, y_cpu.GetFP32(1, 1));
    EXPECT_EQ(0, y_cpu.GetFP32(1, 2));

    EXPECT_EQ(0, y_gpu.GetFP32(0, 0));
    EXPECT_EQ(0, y_gpu.GetFP32(0, 1));
    EXPECT_EQ(1, y_gpu.GetFP32(0, 2));
    EXPECT_EQ(1, y_gpu.GetFP32(1, 0));
    EXPECT_EQ(1, y_gpu.GetFP32(1, 1));
    EXPECT_EQ(0, y_gpu.GetFP32(1, 2));

    /*
    // backward
    bb::FrameBuffer dy_cpu(BB_TYPE_FP32, 2, 3);

    dy_cpu.SetFP32(0, 0, 1);
    dy_cpu.SetFP32(0, 1, 2);
    dy_cpu.SetFP32(0, 2, 3);
    dy_cpu.SetFP32(1, 0, 4);
    dy_cpu.SetFP32(1, 1, 5);
    dy_cpu.SetFP32(1, 2, 6);

    auto dx = relu->Backward(dy);

    EXPECT_EQ(0, dx.GetFP32(0, 0));
    
    EXPECT_EQ(0, dx.GetFP32(0, 1));    // 境界値は両方許容

    EXPECT_EQ(3, dx.GetFP32(0, 2));
    EXPECT_EQ(4, dx.GetFP32(1, 0));
    EXPECT_EQ(5, dx.GetFP32(1, 1));
    EXPECT_EQ(0, dx.GetFP32(1, 2));
    */
}

#endif
