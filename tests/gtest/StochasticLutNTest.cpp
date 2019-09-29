#include <stdio.h>
#include <iostream>
#include <random>
#include "gtest/gtest.h"

#include "bb/StochasticLutN.h"
#include "bb/OptimizerAdam.h"
#include "bb/UniformDistributionGenerator.h"



#if 1 // #ifdef BB_WITH_CUDA


template<typename T=float>
void StochasticLutN_cmp(int const input_node_size, int const output_node_size, int const frame_size, int loop_num)
{
    auto lut_cpu = bb::StochasticLutN<6, float>::Create(output_node_size);
    auto lut_gpu = bb::StochasticLutN<6, float>::Create(output_node_size);

    auto opt_cpu = bb::OptimizerAdam<float>::Create();
    auto opt_gpu = bb::OptimizerAdam<float>::Create();

    lut_cpu->SendCommand("host_only true");

    lut_cpu->SendCommand("lut_binarize true");
    lut_gpu->SendCommand("lut_binarize true");


    bb::FrameBuffer x_cpu(frame_size, {input_node_size}, BB_TYPE_FP32, true);
    bb::FrameBuffer x_gpu(frame_size, {input_node_size}, BB_TYPE_FP32);
    
    lut_cpu->SetInputShape(x_cpu.GetShape());
    lut_gpu->SetInputShape(x_gpu.GetShape());

    // 接続を同一化
    for (int node = 0; node < output_node_size; ++node) {
        for (int i = 0; i < 6; ++i) {
            lut_gpu->SetNodeConnectionIndex(node, i, lut_cpu->GetNodeConnectionIndex(node, i));
        }
    }

    // 係数を同一化
    {
        auto W_cpu = lut_cpu->lock_W_const();
        auto W_gpu = lut_gpu->lock_W();
        for (int node = 0; node < output_node_size; ++node) {
            for (int i = 0; i < 64; ++i) {
                auto W = W_cpu(node, i);
                W_gpu(node, i) = W;
            }
        }
    }
    
    opt_cpu->SetVariables(lut_cpu->GetParameters(), lut_cpu->GetGradients());
    opt_gpu->SetVariables(lut_gpu->GetParameters(), lut_gpu->GetGradients());

    auto valgen = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1);

    for ( int loop = 0; loop < loop_num; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                x_cpu.SetFP32(frame, node, valgen->GetValue());
                x_gpu.SetFP32(frame, node, x_cpu.GetFP32(frame, node));
            }
        }

        auto y_cpu = lut_cpu->Forward(x_cpu);
        auto y_gpu = lut_gpu->Forward(x_gpu);

        EXPECT_EQ(output_node_size, y_cpu.GetNodeSize());
        EXPECT_EQ(output_node_size, y_gpu.GetNodeSize());
        EXPECT_EQ(frame_size, y_cpu.GetFrameSize());
        EXPECT_EQ(frame_size, y_gpu.GetFrameSize());

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                auto val_cpu = x_cpu.GetFP32(frame, node);
                auto val_gpu = x_gpu.GetFP32(frame, node);
                EXPECT_FLOAT_EQ(val_cpu, val_gpu);
                if (std::abs(val_cpu - val_gpu) >= 0.0001f) {
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

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                auto val_cpu = y_cpu.GetFP32(frame, node);
                auto val_gpu = y_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
                if (std::abs(val_cpu - val_gpu) >= 0.0001f) {
                    std::cout << frame << " " << node << std::endl;
                }
            }
        }


        // backward
        bb::FrameBuffer dy_cpu(frame_size, {output_node_size}, BB_TYPE_FP32, true);
        bb::FrameBuffer dy_gpu(frame_size, {output_node_size}, BB_TYPE_FP32);
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



    }


    lut_cpu->SendCommand("lut_binarize false");
    lut_gpu->SendCommand("lut_binarize false");

    for ( int loop = 0; loop < loop_num; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                x_cpu.SetFP32(frame, node, valgen->GetValue());
                x_gpu.SetFP32(frame, node, x_cpu.GetFP32(frame, node));
            }
        }

        auto y_cpu = lut_cpu->Forward(x_cpu, false);
        auto y_gpu = lut_gpu->Forward(x_gpu, false);

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                auto val_cpu = x_cpu.GetFP32(frame, node);
                auto val_gpu = x_gpu.GetFP32(frame, node);
                EXPECT_FLOAT_EQ(val_cpu, val_gpu);
            }
        }

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                auto val_cpu = y_cpu.GetFP32(frame, node);
                auto val_gpu = y_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
            }
        }
    }
}


TEST(StochasticLutNTest, testStochasticLutN_cmp)
{
    StochasticLutN_cmp<float>(6, 1, 1, 1);
    StochasticLutN_cmp<float>(14, 16, 8, 2);

    StochasticLutN_cmp<float>(14, 1024, 8, 4);
    StochasticLutN_cmp<float>(14, 1024, 3, 4);
    StochasticLutN_cmp<float>(6, 1, 1, 4);
//  StochasticLutN_cmp<float>(14, 21, 1024, 4);
    StochasticLutN_cmp<float>(13, 17, 1024 + 512 - 7, 4);
//  StochasticLutN_cmp<float>(17, 256, 28*28*16, 4);
}


/////////////////



template<typename T=float>
void StochasticLutN_cmp_bit(int const input_node_size, int const output_node_size, int const frame_size, int loop_num)
{
    auto lut0 = bb::StochasticLutN<6, float>::Create(output_node_size);
    auto lut1 = bb::StochasticLutN<6, bb::Bit>::Create(output_node_size);

    auto opt0 = bb::OptimizerAdam<float>::Create();
    auto opt1 = bb::OptimizerAdam<float>::Create();

    lut0->SendCommand("binary true");
    lut1->SendCommand("binary true");

    lut0->SendCommand("lut_binarize true");
    lut1->SendCommand("lut_binarize true");


    bb::FrameBuffer x_buf0(frame_size, {input_node_size}, BB_TYPE_FP32);
    bb::FrameBuffer x_buf1(frame_size, {input_node_size}, BB_TYPE_BIT);
    
    lut0->SetInputShape(x_buf0.GetShape());
    lut1->SetInputShape(x_buf1.GetShape());

    // 接続を同一化
    for (int node = 0; node < output_node_size; ++node) {
        for (int i = 0; i < 6; ++i) {
            lut1->SetNodeConnectionIndex(node, i, lut1->GetNodeConnectionIndex(node, i));
        }
    }

    // 係数を同一化
    {
        auto W_ptr0 = lut0->lock_W_const();
        auto W_ptr1 = lut1->lock_W();
        for (int node = 0; node < output_node_size; ++node) {
            for (int i = 0; i < 64; ++i) {
                auto W = W_ptr0(node, i);
                W_ptr1(node, i) = W;
            }
        }
    }
    
    opt0->SetVariables(lut0->GetParameters(), lut0->GetGradients());
    opt1->SetVariables(lut1->GetParameters(), lut1->GetGradients());

    auto valgen = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1);

    for ( int loop = 0; loop < loop_num; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                bool val = (valgen->GetValue() > 0.5);
                x_buf0.SetFP32(frame, node, val ? 1.0f : 0.0f);
                x_buf1.SetBit(frame, node, val);
            }
        }

        auto y_buf0 = lut0->Forward(x_buf0);
        auto y_buf1 = lut1->Forward(x_buf1);

        EXPECT_EQ(output_node_size, y_buf0.GetNodeSize());
        EXPECT_EQ(output_node_size, y_buf1.GetNodeSize());
        EXPECT_EQ(frame_size, y_buf0.GetFrameSize());
        EXPECT_EQ(frame_size, y_buf1.GetFrameSize());

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                bb::Bit val0 = x_buf0.GetFP32(frame, node);
                bb::Bit val1 = x_buf1.GetBit(frame, node);
                EXPECT_EQ(val0, val1);
            }
        }

        {
            auto W_ptr0 = lut0->lock_W_const();
            auto W_ptr1 = lut1->lock_W_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < 64; ++i) {
                    auto val0 = W_ptr0(node, i);
                    auto val1 = W_ptr1(node, i);
                    EXPECT_NEAR(val0, val1, 0.0001f);
                    if (std::abs(val0 - val1) >= 0.0001f) {
                        std::cout <<  node << std::endl;
                    }
                }
            }
        }

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                auto val0 = y_buf0.GetFP32(frame, node);
                auto val1 = y_buf1.GetFP32(frame, node);
                EXPECT_NEAR(val0, val1, 0.0001f);
                if (std::abs(val0 - val1) >= 0.0001f) {
                    std::cout << frame << " " << node << std::endl;
                }
            }
        }


        // backward
        bb::FrameBuffer dy_buf0(frame_size, {output_node_size}, BB_TYPE_FP32, true);
        bb::FrameBuffer dy_buf1(frame_size, {output_node_size}, BB_TYPE_FP32);
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                dy_buf0.SetFP32(frame, node, valgen->GetValue());
                dy_buf1.SetFP32(frame, node, dy_buf0.GetFP32(frame, node));
            }
        }

        auto dx_buf0 = lut0->Backward(dy_buf0);
        auto dx_buf1 = lut1->Backward(dy_buf1);

        EXPECT_EQ(input_node_size, dx_buf0.GetNodeSize());
        EXPECT_EQ(input_node_size, dx_buf1.GetNodeSize());
        EXPECT_EQ(frame_size, dx_buf0.GetFrameSize());
        EXPECT_EQ(frame_size, dx_buf1.GetFrameSize());

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                auto val0 = dx_buf0.GetFP32(frame, node);
                auto val1 = dx_buf1.GetFP32(frame, node);
                EXPECT_NEAR(val0, val1, 0.0001f);
                if (abs(val0 - val1) >= 0.0001f) {
                    std::cout << frame << " " << node << std::endl;
                }
            }
        }

        {
            auto W_ptr0 = lut0->lock_W_const();
            auto W_ptr1 = lut1->lock_W_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < 64; ++i) {
                    auto val0 = W_ptr0(node, i);
                    auto val1 = W_ptr1(node, i);
                    EXPECT_NEAR(val0, val1, 0.0001f);
                    if (std::abs(val0 - val1) >= 0.0001f) {
                        std::cout <<  node << std::endl;
                    }
                }
            }
        }

        {
            auto dW_ptr0 = lut0->lock_dW_const();
            auto dW_ptr1 = lut1->lock_dW_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < 64; ++i) {
                    auto val0 = dW_ptr0(node, i);
                    auto val1 = dW_ptr1(node, i);
                    EXPECT_NEAR(val0, val1, 0.0001f);
                    if ( !(abs(val0 - val1) < 0.0001f) ) {
                        std::cout << node << std::endl;
                        getchar();
                    }
                }
            }
        }

        opt0->Update();
        opt1->Update();

        {
            auto W_ptr0 = lut0->lock_W_const();
            auto W_ptr1 = lut1->lock_W_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < 64; ++i) {
                    auto val0 = W_ptr0(node, i);
                    auto val1 = W_ptr1(node, i);
                    EXPECT_NEAR(val0, val1, 0.0001f);
                    if ( !(std::abs(val0 - val1) < 0.0001f) ) {
                        auto dW_val0 = lut0->lock_dW_const();
                        auto dW_val1 = lut1->lock_dW_const();                      
                        std::cout <<  node << "W0: " << W_ptr0(node, i) << "  W1 : " << W_ptr1(node, i) << std::endl;
                        getchar();
                    }
                }
            }
        }
    }
}


TEST(StochasticLutNTest, testStochasticLutN_cmp_bit)
{
//  StochasticLutN_cmp_bit<float>(14, 16, 4096, 2);

    StochasticLutN_cmp_bit<float>(14, 16, 8, 2);
    StochasticLutN_cmp_bit<float>(14, 1024, 8, 4);
    StochasticLutN_cmp_bit<float>(14, 16, 1024, 2);
    StochasticLutN_cmp_bit<float>(14, 1024, 1024, 1);
    StochasticLutN_cmp_bit<float>(6, 1, 1, 4);
////  StochasticLutN_cmp_bit<float>(14, 21, 1024, 4);
    StochasticLutN_cmp_bit<float>(13, 17, 1024 + 512 - 7, 4);
////  StochasticLutN_cmp_bit<float>(17, 256, 28*28*16, 4);
}


#endif



#if 0

TEST(StochasticLutNTest, testStochasticLutN_connection_depthwise)
{
    bb::indices_t node_shape({2, 3, 4});
    bb::indices_t input_shape({3, 4, 4});


    auto sl = bb::StochasticLutN<6>::Create(node_shape, "depthwise");
    sl->SetInputShape(input_shape);

    for (int c = 0; c < node_shape[2]; ++c) {
        for (int y = 0; y < node_shape[1]; ++y) {
            for (int x = 0; x < node_shape[0]; ++x) {
                bb::indices_t output_indices({x, y, c});
                for (int i = 0; i < 6; ++i) {
                    auto output_index = bb::GetShapeIndex(output_indices, node_shape);
                    auto input_index = sl->GetNodeInput(output_index, i);
                    bb::indices_t input_indices = bb::GetShapeIndices(input_index, input_shape);
                    std::cout << output_index << " ";
                    std::cout << i  << " ";
                    std::cout << input_index << " : out (";
                    std::cout << output_indices[2] << ", ";
                    std::cout << output_indices[1] << ", ";
                    std::cout << output_indices[0] << ") - (";
                    std::cout << input_indices[2] << ", ";
                    std::cout << input_indices[1] << ", ";
                    std::cout << input_indices[0] << ")" << std::endl;
                }
            }
        }
    }
}


TEST(StochasticLutNTest, testStochasticLutN_connection_pointwise)
{
    bb::indices_t node_shape({1, 1, 4});
    bb::indices_t input_shape({1, 1, 4});


    auto sl = bb::StochasticLutN<6>::Create(node_shape, "pointwise");
    sl->SetInputShape(input_shape);

    for (int c = 0; c < node_shape[2]; ++c) {
        for (int y = 0; y < node_shape[1]; ++y) {
            for (int x = 0; x < node_shape[0]; ++x) {
                bb::indices_t output_indices({x, y, c});
                for (int i = 0; i < 6; ++i) {
                    auto output_index = bb::GetShapeIndex(output_indices, node_shape);
                    auto input_index = sl->GetNodeInput(output_index, i);
                    bb::indices_t input_indices = bb::GetShapeIndices(input_index, input_shape);
                    std::cout << output_index << " ";
                    std::cout << i  << " ";
                    std::cout << input_index << " : out (";
                    std::cout << output_indices[2] << ", ";
                    std::cout << output_indices[1] << ", ";
                    std::cout << output_indices[0] << ") - (";
                    std::cout << input_indices[2] << ", ";
                    std::cout << input_indices[1] << ", ";
                    std::cout << input_indices[0] << ")" << std::endl;
                }
            }
        }
    }

}


#endif


