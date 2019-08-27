#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Sigmoid.h"
#include "bb/NormalDistributionGenerator.h"


TEST(SigmoidTest, testSigmoid)
{
    auto sigmoid = bb::Sigmoid<>::Create();
    bb::FrameBuffer x_buf(1, {2}, BB_TYPE_FP32);
    sigmoid->SetInputShape(x_buf.GetShape());

    x_buf.SetFP32(0, 0, 1);
    x_buf.SetFP32(0, 1, 2);

    auto y_buf = sigmoid->Forward(x_buf);

    EXPECT_FLOAT_EQ(1.0f / (1.0f + exp(-1.0f)), y_buf.GetFP32(0, 0));
    EXPECT_FLOAT_EQ(1.0f / (1.0f + exp(-2.0f)), y_buf.GetFP32(0, 1));

    bb::FrameBuffer dy_buf(1, {2}, BB_TYPE_FP32);

    dy_buf.SetFP32(0, 0, 2);
    dy_buf.SetFP32(0, 1, 3);
    
    auto dx_buf = sigmoid->Backward(dy_buf);
    
    EXPECT_FLOAT_EQ(dy_buf.GetFP32(0, 0) * (1.0f - y_buf.GetFP32(0, 0)) * y_buf.GetFP32(0, 0), dx_buf.GetFP32(0, 0));
    EXPECT_FLOAT_EQ(dy_buf.GetFP32(0, 1) * (1.0f - y_buf.GetFP32(0, 1)) * y_buf.GetFP32(0, 1), dx_buf.GetFP32(0, 1));
}


TEST(SigmoidTest, testSigmoidBatch)
{
    auto sigmoid = bb::Sigmoid<>::Create();
    
    // forward
    bb::FrameBuffer x_buf(2, {2}, BB_TYPE_FP32);
    sigmoid->SetInputShape(x_buf.GetShape());

    x_buf.SetFP32(0, 0, 1);
    x_buf.SetFP32(1, 0, 2);
    x_buf.SetFP32(0, 1, 3);
    x_buf.SetFP32(1, 1, 4);

    auto y_buf = sigmoid->Forward(x_buf);

    EXPECT_FLOAT_EQ(1.0f / (1.0f + exp(-1.0f)), y_buf.GetFP32(0, 0));
    EXPECT_FLOAT_EQ(1.0f / (1.0f + exp(-2.0f)), y_buf.GetFP32(1, 0));
    EXPECT_FLOAT_EQ(1.0f / (1.0f + exp(-3.0f)), y_buf.GetFP32(0, 1));
    EXPECT_FLOAT_EQ(1.0f / (1.0f + exp(-4.0f)), y_buf.GetFP32(1, 1));



    // backward
    bb::FrameBuffer dy_buf(2, {2}, BB_TYPE_FP32);

    dy_buf.SetFP32(0, 0, 2);
    dy_buf.SetFP32(1, 0, 3);
    dy_buf.SetFP32(0, 1, 4);
    dy_buf.SetFP32(1, 1, -5);

    auto dx_buf = sigmoid->Backward(dy_buf);

    EXPECT_FLOAT_EQ(dy_buf.GetFP32(0, 0) * (1.0f - y_buf.GetFP32(0, 0)) * y_buf.GetFP32(0, 0), dx_buf.GetFP32(0, 0));
    EXPECT_FLOAT_EQ(dy_buf.GetFP32(1, 0) * (1.0f - y_buf.GetFP32(1, 0)) * y_buf.GetFP32(1, 0), dx_buf.GetFP32(1, 0));
    EXPECT_FLOAT_EQ(dy_buf.GetFP32(0, 1) * (1.0f - y_buf.GetFP32(0, 1)) * y_buf.GetFP32(0, 1), dx_buf.GetFP32(0, 1));
    EXPECT_FLOAT_EQ(dy_buf.GetFP32(1, 1) * (1.0f - y_buf.GetFP32(1, 1)) * y_buf.GetFP32(1, 1), dx_buf.GetFP32(1, 1));
}




#ifdef BB_WITH_CUDA

template<typename T = float>
void testSigmoid_cmp(int node_size, int frame_size, int loop_num = 3)
{
    auto act_cpu = bb::Sigmoid<T>::Create();
    auto act_gpu = bb::Sigmoid<T>::Create();

    act_cpu->SendCommand("host_only true");

    bb::FrameBuffer x_cpu(frame_size, {node_size}, BB_TYPE_FP32, true);
    bb::FrameBuffer x_gpu(frame_size, {node_size}, BB_TYPE_FP32);
    
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
        bb::FrameBuffer dy_cpu(frame_size, {node_size}, BB_TYPE_FP32, true);
        bb::FrameBuffer dy_gpu(frame_size, {node_size}, BB_TYPE_FP32);
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


TEST(SigmoidTest, testSigmoid_test_cmp0) { testSigmoid_cmp<float>(1, 1); }
TEST(SigmoidTest, testSigmoid_test_cmp1) { testSigmoid_cmp<float>(7, 7); }
TEST(SigmoidTest, testSigmoid_test_cmp2) { testSigmoid_cmp<float>(8, 8); }
TEST(SigmoidTest, testSigmoid_test_cmp3) { testSigmoid_cmp<float>(9, 9); }
TEST(SigmoidTest, testSigmoid_test_cmp4)
{ 
    testSigmoid_cmp<float>(1024, 1);
}
TEST(SigmoidTest, testSigmoid_test_cmp5) { testSigmoid_cmp<float>(1, 1024); }
TEST(SigmoidTest, testSigmoid_test_cmp6) { testSigmoid_cmp<float>(1023, 2); }
TEST(SigmoidTest, testSigmoid_test_cmp7) { testSigmoid_cmp<float>(2, 1025); }
TEST(SigmoidTest, testSigmoid_test_cmp8) { testSigmoid_cmp<float>(1025, 3); }


#endif

