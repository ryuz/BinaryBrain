#include <stdio.h>
#include <iostream>
#include <random>
#include "gtest/gtest.h"

#include "bb/MicroMlp.h"
#include "bb/OptimizerAdam.h"
#include "bb/UniformDistributionGenerator.h"


#ifdef BB_WITH_CUDA



template<typename T=float>
void MicroMlp_cmp_bit(int const input_node_size, int const output_node_size, int const frame_size, int loop_num)
{
    auto mlp0 = bb::MicroMlp<6, 16, float>::Create(output_node_size);
    auto mlp1 = bb::MicroMlp<6, 16, bb::Bit>::Create(output_node_size);

    auto opt0 = bb::OptimizerAdam<float>::Create();
    auto opt1 = bb::OptimizerAdam<float>::Create();

    bb::FrameBuffer x_buf0(BB_TYPE_FP32, frame_size, input_node_size);
    bb::FrameBuffer x_buf1(BB_TYPE_BIT,  frame_size, input_node_size);
    
    mlp0->SetInputShape(x_buf0.GetShape());
    mlp1->SetInputShape(x_buf1.GetShape());

    // 接続を同一化
    for (int node = 0; node < output_node_size; ++node) {
        for (int i = 0; i < 6; ++i) {
            mlp1->SetNodeInput(node, i, mlp0->GetNodeInput(node, i));
        }
    }

    // 係数を同一化
    {
        auto W0_ptr0 = mlp0->lock_W0_const();
        auto W0_ptr1 = mlp1->lock_W0();
        auto b0_ptr0 = mlp0->lock_b0_const();
        auto b0_ptr1 = mlp1->lock_b0();
        auto W1_ptr0 = mlp0->lock_W1_const();
        auto W1_ptr1 = mlp1->lock_W1();
        auto b1_ptr0 = mlp0->lock_b1_const();
        auto b1_ptr1 = mlp1->lock_b1();
        for (int node = 0; node < output_node_size; ++node) {
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 6; ++j) {
                    W0_ptr1(node, i, j) = W0_ptr0(node, i, j);
                }
                b0_ptr1(node, i) = b0_ptr0(node, i);
                W1_ptr1(node, i) = W1_ptr0(node, i);
            }
            b1_ptr1(node) = b1_ptr0(node);
        }
    }
    
    opt0->SetVariables(mlp0->GetParameters(), mlp0->GetGradients());
    opt1->SetVariables(mlp1->GetParameters(), mlp1->GetGradients());

    std::mt19937_64 mt(1);
    std::uniform_int_distribution<int> dist(0, 1);

    for ( int loop = 0; loop < loop_num; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                bb::Bit val = dist(mt);
                x_buf0.SetBit(frame, node, val);
                x_buf1.SetBit(frame, node, val);
            }
        }

        auto y_buf0 = mlp0->Forward(x_buf0);
        auto y_buf1 = mlp1->Forward(x_buf1);

        EXPECT_EQ(output_node_size, y_buf0.GetNodeSize());
        EXPECT_EQ(output_node_size, y_buf1.GetNodeSize());
        EXPECT_EQ(frame_size, y_buf0.GetFrameSize());
        EXPECT_EQ(frame_size, y_buf1.GetFrameSize());

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                bb::Bit val0 = (bb::Bit)x_buf0.GetFP32(frame, node);
                bb::Bit val1 = x_buf1.GetBit(frame, node);
                EXPECT_EQ(val0, val1);
            }
        }

        /*
        {
            auto W_ptr0 = mlp0->lock_W_const();
            auto W_ptr1 = mlp1->lock_W_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < 64; ++i) {
                    auto val_cpu = W_cpu(node, i);
                    auto val_gpu = W_gpu(node, i);
                    EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
                    if (std::abs(val_cpu - val_gpu) >= 0.0001f) {
                        std::cout <<  node << std::endl;
                    }
                }
            }
        }
        */

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                bb::Bit val0 = (bb::Bit)y_buf0.GetFP32(frame, node);
                bb::Bit val1 = y_buf1.GetBit(frame, node);
                EXPECT_EQ(val0, val1);
                if ( val0 != val1 ) {
                    std::cout << "node : " << node << "  frame : " << frame << std::endl;
                }
            }
        }

#if 0
        // backward
        bb::FrameBuffer dy_cpu(BB_TYPE_FP32, frame_size, output_node_size, true);
        bb::FrameBuffer dy_gpu(BB_TYPE_FP32, frame_size, output_node_size);
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                dy_cpu.SetFP32(frame, node, valgen->GetValue());
                dy_gpu.SetFP32(frame, node, dy_cpu.GetFP32(frame, node));
            }
        }

        auto dx_cpu = lut_cpu->Backward(dy_cpu);
        auto dx_gpu = lut_gpu->Backward(dy_gpu);

        EXPECT_EQ(input_node_size, dx_cpu.GetNodeSize());
        EXPECT_EQ(input_node_size, dx_gpu.GetNodeSize());
        EXPECT_EQ(frame_size, dx_cpu.GetFrameSize());
        EXPECT_EQ(frame_size, dx_gpu.GetFrameSize());

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                auto val_cpu = dx_cpu.GetFP32(frame, node);
                auto val_gpu = dx_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
                if (abs(val_cpu - val_gpu) >= 0.0001f) {
                    std::cout << frame << " " << node << std::endl;
                }
            }
        }

        {
            auto W_cpu = lut_cpu->lock_W_const();
            auto W_gpu = lut_gpu->lock_W_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < 64; ++i) {
                    auto val_cpu = W_cpu(node, i);
                    auto val_gpu = W_gpu(node, i);
                    EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
                    if (std::abs(val_cpu - val_gpu) >= 0.0001f) {
                        std::cout <<  node << std::endl;
                    }
                }
            }
        }

        {
            auto dW_cpu = lut_cpu->lock_dW_const();
            auto dW_gpu = lut_gpu->lock_dW_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < 64; ++i) {
                    auto val_cpu = dW_cpu(node, i);
                    auto val_gpu = dW_gpu(node, i);
                    EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
                    if ( !(abs(val_cpu - val_gpu) < 0.0001f) ) {
                        std::cout << node << std::endl;
                        getchar();
                    }
                }
            }
        }

        opt_cpu->Update();
        opt_gpu->Update();

        {
            auto W_cpu = lut_cpu->lock_W_const();
            auto W_gpu = lut_gpu->lock_W_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < 64; ++i) {
                    auto val_cpu = W_cpu(node, i);
                    auto val_gpu = W_gpu(node, i);
                    EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
                    if ( !(std::abs(val_cpu - val_gpu) < 0.0001f) ) {
                        auto dW_cpu = lut_cpu->lock_dW_const();
                        auto dW_gpu = lut_gpu->lock_dW_const();                      
                        std::cout <<  node << "cpu_dW: " << dW_cpu(node, i) << "  gpu_dW : " << dW_gpu(node, i) << std::endl;
                        getchar();
                    }
                }
            }
        }
#endif
    }
}



TEST(MicroMlpTest, testMicroMlp_cmp_bit)
{
//    MicroMlp_cmp_bit<float>(6, 1, 1, 4);
    MicroMlp_cmp_bit<float>(6, 1, 32, 1);

//  MicroMlp_cmp_bit<float>(14, 1024, 8, 4);
//  MicroMlp_cmp_bit<float>(14, 1024, 3, 4);
//  MicroMlp_cmp_bit<float>(6, 1, 1, 4);
//  MicroMlp_cmp_bit<float>(14, 21, 1024, 4);
//  MicroMlp_cmp_bit<float>(13, 17, 1024 + 512 - 7, 4);
//  MicroMlp_cmp_bit<float>(17, 256, 28*28*16, 4);
}


#endif


