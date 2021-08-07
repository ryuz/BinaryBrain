#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Binarize.h"


TEST(BinarizeTest, testBinarize_test0)
{
    auto bin = bb::Binarize<>::Create();
    
    bb::FrameBuffer x_buf(2, {3}, BB_TYPE_FP32);
    bin->SetInputShape(x_buf.GetShape());

    x_buf.SetFP32(0, 0, -1.01f);
    x_buf.SetFP32(0, 1, -0.9f); 
    x_buf.SetFP32(0, 2, -0.5f); 
    x_buf.SetFP32(1, 0, -0.1f); 
    x_buf.SetFP32(1, 1, +0.9f); 
    x_buf.SetFP32(1, 2, +1.01f);

    auto y_buf = bin->Forward(x_buf);
    
    EXPECT_EQ(-1, y_buf.GetFP32(0, 0));
    EXPECT_EQ(-1, y_buf.GetFP32(0, 1));
    EXPECT_EQ(-1, y_buf.GetFP32(0, 2));
    EXPECT_EQ(-1, y_buf.GetFP32(1, 0));
    EXPECT_EQ(+1, y_buf.GetFP32(1, 1));
    EXPECT_EQ(+1, y_buf.GetFP32(1, 2));

    // backward
    bb::FrameBuffer dy_buf(2, {3}, BB_TYPE_FP32);
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
    
    bb::FrameBuffer x_buf(3, {2}, BB_TYPE_FP32);
    bin->SetInputShape(x_buf.GetShape());

    x_buf.SetFP32(0, 0, -1.01f);
    x_buf.SetFP32(1, 0, +0.99f); 
    x_buf.SetFP32(2, 0, -0.5f); 
    x_buf.SetFP32(0, 1, -0.99f); 
    x_buf.SetFP32(1, 1,  0.01f); 
    x_buf.SetFP32(2, 1, +1.01f);

    auto y_buf = bin->Forward(x_buf);

    EXPECT_EQ(false, y_buf.GetBit(0, 0));
    EXPECT_EQ(true,  y_buf.GetBit(1, 0));
    EXPECT_EQ(false, y_buf.GetBit(2, 0));
    EXPECT_EQ(false, y_buf.GetBit(0, 1));
    EXPECT_EQ(true,  y_buf.GetBit(1, 1));
    EXPECT_EQ(true,  y_buf.GetBit(2, 1));

    // backward
    bb::FrameBuffer dy_buf(3, {2}, BB_TYPE_FP32);
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


TEST(BinarizeTest, testBinarize_cast)
{
    bb::Bit bit_0  = false;
    bb::Bit bit_1  = true;
    float   fp32_0 = -1;
    float   fp32_1 = +1;
    int     i32_0  = -1;
    float   i32_1  = +1;

    EXPECT_EQ(-1.0,  (double       )bit_0);
    EXPECT_EQ(-1.0f, (float        )bit_0);
    EXPECT_EQ(-1,    (std::int64_t )bit_0);
    EXPECT_EQ(-1,    (std::int32_t )bit_0);
    EXPECT_EQ(-1,    (std::int16_t )bit_0);
    EXPECT_EQ(-1,    (std::int8_t  )bit_0);
    EXPECT_EQ(-1,    (std::int64_t )bit_0);
    EXPECT_EQ(-1,    (std::int32_t )bit_0);
    EXPECT_EQ(-1,    (std::int16_t )bit_0);
    EXPECT_EQ(-1,    (std::int8_t  )bit_0);
    EXPECT_EQ( 0,    (std::uint64_t)bit_0);
    EXPECT_EQ( 0,    (std::uint32_t)bit_0);
    EXPECT_EQ( 0,    (std::uint16_t)bit_0);
    EXPECT_EQ( 0,    (std::uint8_t )bit_0);
    EXPECT_EQ(false, (bool)bit_0);
    EXPECT_EQ(false, bit_0 ? true : false);
    
    EXPECT_EQ(+1.0,  (double       )bit_1);
    EXPECT_EQ(+1.0f, (float        )bit_1);
    EXPECT_EQ(+1,    (std::int64_t )bit_1);
    EXPECT_EQ(+1,    (std::int32_t )bit_1);
    EXPECT_EQ(+1,    (std::int16_t )bit_1);
    EXPECT_EQ(+1,    (std::int8_t  )bit_1);
    EXPECT_EQ(+1,    (std::int64_t )bit_1);
    EXPECT_EQ(+1,    (std::int32_t )bit_1);
    EXPECT_EQ(+1,    (std::int16_t )bit_1);
    EXPECT_EQ(+1,    (std::int8_t  )bit_1);
    EXPECT_EQ( 1,    (std::uint64_t)bit_1);
    EXPECT_EQ( 1,    (std::uint32_t)bit_1);
    EXPECT_EQ( 1,    (std::uint16_t)bit_1);
    EXPECT_EQ( 1,    (std::uint8_t )bit_1);
    EXPECT_EQ(true, (bool)bit_1);
    EXPECT_EQ(true, bit_1 ? true : false);


    EXPECT_EQ(-1.0,  (double       )fp32_0);
    EXPECT_EQ(-1.0f, (float        )fp32_0);
    EXPECT_EQ(-1,    (std::int64_t )fp32_0);
    EXPECT_EQ(-1,    (std::int32_t )fp32_0);
    EXPECT_EQ(-1,    (std::int16_t )fp32_0);
    EXPECT_EQ(-1,    (std::int8_t  )fp32_0);
    EXPECT_EQ(-1,    (std::int64_t )fp32_0);
    EXPECT_EQ(-1,    (std::int32_t )fp32_0);
    EXPECT_EQ(-1,    (std::int16_t )fp32_0);
    EXPECT_EQ(-1,    (std::int8_t  )fp32_0);
    EXPECT_EQ(false, (bb::Bit)      fp32_0);
    
    EXPECT_EQ(+1.0,  (double       )fp32_1);
    EXPECT_EQ(+1.0f, (float        )fp32_1);
    EXPECT_EQ(+1,    (std::int64_t )fp32_1);
    EXPECT_EQ(+1,    (std::int32_t )fp32_1);
    EXPECT_EQ(+1,    (std::int16_t )fp32_1);
    EXPECT_EQ(+1,    (std::int8_t  )fp32_1);
    EXPECT_EQ(+1,    (std::int64_t )fp32_1);
    EXPECT_EQ(+1,    (std::int32_t )fp32_1);
    EXPECT_EQ(+1,    (std::int16_t )fp32_1);
    EXPECT_EQ(+1,    (std::int8_t  )fp32_1);
    EXPECT_EQ(true,  (bb::Bit)      fp32_1);
}


#if 0 

TEST(BinarizeTest, testBinarize_comp)
{
    int const node_size = 10;
    int const frame_size = 10;

    auto bin_cpu = bb::Binarize<>::Create();
    auto bin_gpu = bb::Binarize<>::Create();
    
    bb::FrameBuffer x_cpu(2, 3, BB_TYPE_FP32, true);
    bb::FrameBuffer x_gpu(2, 3, BB_TYPE_FP32, );

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
    bb::FrameBuffer dy_cpu(2, 3, BB_TYPE_FP32);

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
