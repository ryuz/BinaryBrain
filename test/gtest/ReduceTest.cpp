#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Reduce.h"
#include "bb/NormalDistributionGenerator.h"


TEST(ReduceTest, testReduce_test0)
{
    auto reduce = bb::Reduce<>::Create(3);
    
    bb::FrameBuffer x_buf(BB_TYPE_FP32, 2, 9);
    reduce->SetInputShape(x_buf.GetShape());

    x_buf.SetFP32(0, 0, 1);
    x_buf.SetFP32(0, 1, 2);
    x_buf.SetFP32(0, 2, 3);
    x_buf.SetFP32(0, 3, 4);
    x_buf.SetFP32(0, 4, 5);
    x_buf.SetFP32(0, 5, 6);
    x_buf.SetFP32(0, 6, 7);
    x_buf.SetFP32(0, 7, 8);
    x_buf.SetFP32(0, 8, 9);
    x_buf.SetFP32(1, 0, 11);
    x_buf.SetFP32(1, 1, 12);
    x_buf.SetFP32(1, 2, 13);
    x_buf.SetFP32(1, 3, 14);
    x_buf.SetFP32(1, 4, 15);
    x_buf.SetFP32(1, 5, 16);
    x_buf.SetFP32(1, 6, 17);
    x_buf.SetFP32(1, 7, 18);
    x_buf.SetFP32(1, 8, 19);

    auto y_buf = reduce->Forward(x_buf);
    
    EXPECT_FLOAT_EQ((1 + 4 + 7)/3.0f, y_buf.GetFP32(0, 0));
    EXPECT_FLOAT_EQ((2 + 5 + 8)/3.0f, y_buf.GetFP32(0, 1));
    EXPECT_FLOAT_EQ((3 + 6 + 9)/3.0f, y_buf.GetFP32(0, 2));

    EXPECT_FLOAT_EQ((11 + 14 + 17)/3.0f, y_buf.GetFP32(1, 0));
    EXPECT_FLOAT_EQ((12 + 15 + 18)/3.0f, y_buf.GetFP32(1, 1));
    EXPECT_FLOAT_EQ((13 + 16 + 19)/3.0f, y_buf.GetFP32(1, 2));

    // backward
    bb::FrameBuffer dy_buf(BB_TYPE_FP32, 2, 3);

    dy_buf.SetFP32(0, 0, 1);
    dy_buf.SetFP32(0, 1, 2);
    dy_buf.SetFP32(0, 2, 3);
    dy_buf.SetFP32(1, 0, 4);
    dy_buf.SetFP32(1, 1, 5);
    dy_buf.SetFP32(1, 2, 6);

    auto dx_buf = reduce->Backward(dy_buf);

   EXPECT_FLOAT_EQ(1/3.0f, dx_buf.GetFP32(0, 0));
   EXPECT_FLOAT_EQ(2/3.0f, dx_buf.GetFP32(0, 1));
   EXPECT_FLOAT_EQ(3/3.0f, dx_buf.GetFP32(0, 2));
   EXPECT_FLOAT_EQ(1/3.0f, dx_buf.GetFP32(0, 3));
   EXPECT_FLOAT_EQ(2/3.0f, dx_buf.GetFP32(0, 4));
   EXPECT_FLOAT_EQ(3/3.0f, dx_buf.GetFP32(0, 5));
   EXPECT_FLOAT_EQ(1/3.0f, dx_buf.GetFP32(0, 6));
   EXPECT_FLOAT_EQ(2/3.0f, dx_buf.GetFP32(0, 7));
   EXPECT_FLOAT_EQ(3/3.0f, dx_buf.GetFP32(0, 8));
   EXPECT_FLOAT_EQ(4/3.0f, dx_buf.GetFP32(1, 0));
   EXPECT_FLOAT_EQ(5/3.0f, dx_buf.GetFP32(1, 1));
   EXPECT_FLOAT_EQ(6/3.0f, dx_buf.GetFP32(1, 2));
   EXPECT_FLOAT_EQ(4/3.0f, dx_buf.GetFP32(1, 3));
   EXPECT_FLOAT_EQ(5/3.0f, dx_buf.GetFP32(1, 4));
   EXPECT_FLOAT_EQ(6/3.0f, dx_buf.GetFP32(1, 5));
   EXPECT_FLOAT_EQ(4/3.0f, dx_buf.GetFP32(1, 6));
   EXPECT_FLOAT_EQ(5/3.0f, dx_buf.GetFP32(1, 7));
   EXPECT_FLOAT_EQ(6/3.0f, dx_buf.GetFP32(1, 8));
}


#if 0 // #ifdef BB_WITH_CUDA

template<typename T = float>
void testReLU_cmp(int node_size, int frame_size, int loop_num = 3)
{
    auto act_cpu = bb::ReLU<T>::Create();
    auto act_gpu = bb::ReLU<T>::Create();

    act_cpu->SendCommand("host_only true");

    bb::FrameBuffer x_cpu(BB_TYPE_FP32, frame_size, node_size, true);
    bb::FrameBuffer x_gpu(BB_TYPE_FP32, frame_size, node_size);
    
    act_cpu->SetInputShape(x_cpu.GetShape());
    act_gpu->SetInputShape(x_gpu.GetShape());

    auto valgen = bb::NormalDistributionGenerator<float>::Create(1.2f, 3.3f, 1);
    for ( int loop = 0; loop < loop_num; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                x_cpu.SetFP32(frame, node, valgen->GetValue());
                x_gpu.SetFP32(frame, node, x_cpu.GetFP32(frame, node));
            }
        }

        auto y_cpu = act_cpu->Forward(x_cpu);
        auto y_gpu = act_gpu->Forward(x_gpu);

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = x_cpu.GetFP32(frame, node);
                auto val_gpu = x_gpu.GetFP32(frame, node);
                EXPECT_FLOAT_EQ(val_cpu, val_gpu);
            }
        }

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = y_cpu.GetFP32(frame, node);
                auto val_gpu = y_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.00001f);
            }
        }


        // backward
        bb::FrameBuffer dy_cpu(BB_TYPE_FP32, frame_size, node_size, true);
        bb::FrameBuffer dy_gpu(BB_TYPE_FP32, frame_size, node_size);
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                dy_cpu.SetFP32(frame, node, valgen->GetValue());
                dy_gpu.SetFP32(frame, node, dy_cpu.GetFP32(frame, node));
            }
        }

        auto dx_cpu = act_cpu->Backward(dy_cpu);
        auto dx_gpu = act_gpu->Backward(dy_gpu);

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = dx_cpu.GetFP32(frame, node);
                auto val_gpu = dx_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
            }
        }
    }
}


TEST(ReLUTest, testReLU_cmp0) { testReLU_cmp<float>(1, 1); }
TEST(ReLUTest, testReLU_cmp1) { testReLU_cmp<float>(7, 7); }
TEST(ReLUTest, testReLU_cmp2) { testReLU_cmp<float>(8, 8); }
TEST(ReLUTest, testReLU_cmp3) { testReLU_cmp<float>(9, 9); }
TEST(ReLUTest, testReLU_cmp4) { testReLU_cmp<float>(1024, 1); }
TEST(ReLUTest, testReLU_cmp5) { testReLU_cmp<float>(1, 1024); }
TEST(ReLUTest, testReLU_cmp6) { testReLU_cmp<float>(1023, 2); }
TEST(ReLUTest, testReLU_cmp7) { testReLU_cmp<float>(2, 1025); }
TEST(ReLUTest, testReLU_cmp8) { testReLU_cmp<float>(1025, 3); }

#endif


