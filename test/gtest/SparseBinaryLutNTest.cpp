#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <random>
#include "gtest/gtest.h"

#include "bb/SparseBinaryLutN.h"
#include "bb/SparseLutN.h"
#include "bb/OptimizerAdam.h"
#include "bb/UniformDistributionGenerator.h"



#ifdef BB_WITH_CUDA


void SparseBinaryLutNTest_cmp(int const input_node_size, int const output_node_size, int const frame_size, int loop_num)
{
    auto lut0 = bb::SparseBinaryLutN<6, bb::Bit>::Create(output_node_size);
    auto lut1 = bb::SparseLutN<6, bb::Bit>::Create(output_node_size);

    auto opt0 = bb::OptimizerAdam<float>::Create();
    auto opt1 = bb::OptimizerAdam<float>::Create();

    lut0->SendCommand("lut_binarize true");
    lut1->SendCommand("lut_binarize true");

    bb::FrameBuffer x_buf0(BB_TYPE_BIT, frame_size, input_node_size);
    bb::FrameBuffer x_buf1(BB_TYPE_BIT, frame_size, input_node_size);
    
    lut0->SetInputShape(x_buf0.GetShape());
    lut1->SetInputShape(x_buf1.GetShape());

    // 接続を同一化
    for (int node = 0; node < output_node_size; ++node) {
        for (int i = 0; i < 6; ++i) {
            lut1->SetNodeInput(node, i, lut1->GetNodeInput(node, i));
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
        {
            auto x_ptr0 = x_buf0.Lock<bb::Bit>();
            auto x_ptr1 = x_buf1.Lock<bb::Bit>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < input_node_size; ++node ) {
                    bool val = (valgen->GetValue() > 0.5);
                    x_ptr0.Set(frame, node, val);
                    x_ptr1.Set(frame, node, val);
                }
            }
        }

        auto y_buf0 = lut0->Forward(x_buf0);
        auto y_buf1 = lut1->Forward(x_buf1);

        EXPECT_EQ(output_node_size, y_buf0.GetNodeSize());
        EXPECT_EQ(output_node_size, y_buf1.GetNodeSize());
        EXPECT_EQ(frame_size, y_buf0.GetFrameSize());
        EXPECT_EQ(frame_size, y_buf1.GetFrameSize());

        {
            auto x_ptr0 = x_buf0.LockConst<bb::Bit>();
            auto x_ptr1 = x_buf1.LockConst<bb::Bit>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < input_node_size; ++node ) {
                    bb::Bit val0 = x_ptr0.Get(frame, node);
                    bb::Bit val1 = x_ptr1.Get(frame, node);
                    EXPECT_EQ(val0, val1);
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
            auto mean_ptr0 = lut0->lock_mean_const();
            auto mean_ptr1 = lut1->lock_mean_const();
            for (int node = 0; node < output_node_size; ++node) {
                auto val0 = mean_ptr0(node);
                auto val1 = mean_ptr1(node);
                EXPECT_NEAR(val0, val1, 0.001f);
            }
        }

        {
            auto var_ptr0 = lut0->lock_var_const();
            auto var_ptr1 = lut1->lock_var_const();
            for (int node = 0; node < output_node_size; ++node) {
                auto val0 = var_ptr0(node);
                auto val1 = var_ptr1(node);
                EXPECT_NEAR(val0, val1, 0.001f);
            }
        }

        {
            auto mean_ptr0 = lut0->lock_tmp_mean_const();
            auto mean_ptr1 = lut1->lock_tmp_mean_const();
            for (int node = 0; node < output_node_size; ++node) {
                auto val0 = mean_ptr0(node);
                auto val1 = mean_ptr1(node);
                EXPECT_NEAR(val0, val1, 0.001f);
            }
        }

        {
            auto rstd_ptr0 = lut0->lock_tmp_rstd_const();
            auto rstd_ptr1 = lut1->lock_tmp_rstd_const();
            for (int node = 0; node < output_node_size; ++node) {
                auto val0 = rstd_ptr0(node);
                auto val1 = rstd_ptr1(node);
                EXPECT_NEAR(val0, val1, 0.001f);
            }
        }


        {
            auto y_ptr0 = y_buf0.LockConst<bb::Bit>();
            auto y_ptr1 = y_buf1.LockConst<bb::Bit>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < output_node_size; ++node ) {
                    bb::Bit val0 = y_ptr0.Get(frame, node);
                    bb::Bit val1 = y_ptr1.Get(frame, node);
                    EXPECT_EQ(val0, val1);
                }
            }
        }


#if 1
        // backward
        bb::FrameBuffer dy_buf0(BB_TYPE_FP32, frame_size, output_node_size);
        bb::FrameBuffer dy_buf1(BB_TYPE_FP32, frame_size, output_node_size);
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                float val = valgen->GetValue();
                dy_buf0.SetFP32(frame, node, val);
                dy_buf1.SetFP32(frame, node, val);
            }
        }

        auto dx_buf0 = lut0->Backward(dy_buf0);
        auto dx_buf1 = lut1->Backward(dy_buf1);

        EXPECT_EQ(input_node_size, dx_buf0.GetNodeSize());
        EXPECT_EQ(input_node_size, dx_buf1.GetNodeSize());
        EXPECT_EQ(frame_size, dx_buf0.GetFrameSize());
        EXPECT_EQ(frame_size, dx_buf1.GetFrameSize());

        // dy一応確認
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                auto val0 = dy_buf0.GetFP32(frame, node);
                auto val1 = dy_buf1.GetFP32(frame, node);
                EXPECT_FLOAT_EQ(val0, val1);
//              std::cout << "frame : " << frame << "  node : " << node << " dy : " << val0 << " " << val1 << std::endl;
            }
        }

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < input_node_size; ++node ) {
                auto val0 = dx_buf0.GetFP32(frame, node);
                auto val1 = dx_buf1.GetFP32(frame, node);
                EXPECT_NEAR(val0, val1, 0.0001f);
                if (abs(val0 - val1) >= 0.0001f) {
                    std::cout << frame << " " << node << std::endl;
                    getchar();
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
                        getchar();
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
#endif
    }
}


TEST(SparseBinaryLutNTest, testSparseBinaryLutN_cmp)
{
    SparseBinaryLutNTest_cmp(6,    1,       8, 2);
    SparseBinaryLutNTest_cmp(6,    1,    1024, 2);
    SparseBinaryLutNTest_cmp(6,    1024,    1, 2);
    SparseBinaryLutNTest_cmp(6,    2,      32, 2);
    SparseBinaryLutNTest_cmp(6,    1024, 1024, 2);
}

#endif



#if 0

void SparseBinaryLutNTest_cmp_split(int input_node_size, int output_node_size, int frame_size, int bw_max_frame_size, int loop_num)
{
    auto lut0 = bb::SparseBinaryLutN<6, bb::Bit>::Create(output_node_size);
    auto lut1 = bb::SparseBinaryLutN<6, bb::Bit>::Create(output_node_size);

    auto opt0 = bb::OptimizerAdam<float>::Create();
    auto opt1 = bb::OptimizerAdam<float>::Create();

    lut0->SendCommand("lut_binarize true");
    lut1->SendCommand("lut_binarize true");

    bb::FrameBuffer x_buf0(BB_TYPE_BIT, frame_size, input_node_size);
    bb::FrameBuffer x_buf1(BB_TYPE_BIT, frame_size, input_node_size);
    
    lut0->SetInputShape(x_buf0.GetShape());
    lut1->SetInputShape(x_buf1.GetShape());

    // 接続を同一化
    for (int node = 0; node < output_node_size; ++node) {
        for (int i = 0; i < 6; ++i) {
            lut1->SetNodeInput(node, i, lut1->GetNodeInput(node, i));
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
        {
            auto x_ptr0 = x_buf0.Lock<bb::Bit>();
            auto x_ptr1 = x_buf1.Lock<bb::Bit>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < input_node_size; ++node ) {
                    bool val = (valgen->GetValue() > 0.5);
                    x_ptr0.Set(frame, node, val);
                    x_ptr1.Set(frame, node, val);
                }
            }
        }

        auto y_buf0 = lut0->Forward(x_buf0);
        auto y_buf1 = lut1->Forward(x_buf1);

        EXPECT_EQ(output_node_size, y_buf0.GetNodeSize());
        EXPECT_EQ(output_node_size, y_buf1.GetNodeSize());
        EXPECT_EQ(frame_size, y_buf0.GetFrameSize());
        EXPECT_EQ(frame_size, y_buf1.GetFrameSize());

        {
            auto x_ptr0 = x_buf0.LockConst<bb::Bit>();
            auto x_ptr1 = x_buf1.LockConst<bb::Bit>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < input_node_size; ++node ) {
                    bb::Bit val0 = x_ptr0.Get(frame, node);
                    bb::Bit val1 = x_ptr1.Get(frame, node);
                    EXPECT_EQ(val0, val1);
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
            auto mean_ptr0 = lut0->lock_mean_const();
            auto mean_ptr1 = lut1->lock_mean_const();
            for (int node = 0; node < output_node_size; ++node) {
                auto val0 = mean_ptr0(node);
                auto val1 = mean_ptr1(node);
                EXPECT_NEAR(val0, val1, 0.001f);
            }
        }

        {
            auto var_ptr0 = lut0->lock_var_const();
            auto var_ptr1 = lut1->lock_var_const();
            for (int node = 0; node < output_node_size; ++node) {
                auto val0 = var_ptr0(node);
                auto val1 = var_ptr1(node);
                EXPECT_NEAR(val0, val1, 0.001f);
            }
        }

        {
            auto mean_ptr0 = lut0->lock_tmp_mean_const();
            auto mean_ptr1 = lut1->lock_tmp_mean_const();
            for (int node = 0; node < output_node_size; ++node) {
                auto val0 = mean_ptr0(node);
                auto val1 = mean_ptr1(node);
                EXPECT_NEAR(val0, val1, 0.001f);
            }
        }

        {
            auto rstd_ptr0 = lut0->lock_tmp_rstd_const();
            auto rstd_ptr1 = lut1->lock_tmp_rstd_const();
            for (int node = 0; node < output_node_size; ++node) {
                auto val0 = rstd_ptr0(node);
                auto val1 = rstd_ptr1(node);
                EXPECT_NEAR(val0, val1, 0.001f);
            }
        }


        {
            auto y_ptr0 = y_buf0.LockConst<bb::Bit>();
            auto y_ptr1 = y_buf1.LockConst<bb::Bit>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < output_node_size; ++node ) {
                    bb::Bit val0 = y_ptr0.Get(frame, node);
                    bb::Bit val1 = y_ptr1.Get(frame, node);
                    EXPECT_EQ(val0, val1);
                }
            }
        }

        // backward
        bb::FrameBuffer dy_buf0(BB_TYPE_FP32, frame_size, output_node_size);
        bb::FrameBuffer dy_buf1(BB_TYPE_FP32, frame_size, output_node_size);
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < output_node_size; ++node ) {
                float val = valgen->GetValue();
                dy_buf0.SetFP32(frame, node, val);
                dy_buf1.SetFP32(frame, node, val);
            }
        }

        auto dx_buf0 = lut0->Backward(dy_buf0);
        
        bb::FrameBuffer dx_buf1(BB_TYPE_FP32, frame_size, input_node_size);
        int x_frame_offset = 0;
        do {
            int bw_frame_size = (int)(frame_size - x_frame_offset);
            bw_frame_size = std::min(bw_frame_size, bw_max_frame_size);
            bb::FrameBuffer dy_tmp(BB_TYPE_FP32, bw_frame_size, output_node_size);
            dy_buf1.CopyTo(dy_tmp, bw_frame_size, x_frame_offset);
            auto dx_tmp = lut1->Backward(dy_tmp, x_frame_offset);
            dx_tmp.CopyTo(dx_buf1, bw_frame_size, 0, x_frame_offset);
            x_frame_offset += bw_frame_size;
        } while ( x_frame_offset < frame_size );

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



TEST(SparseBinaryLutNTest, testSparseBinaryLutN_cmp_split)
{
    SparseBinaryLutNTest_cmp_split(6, 1, 64, 32, 2);
}

#endif



