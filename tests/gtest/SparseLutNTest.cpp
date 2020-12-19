#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <random>
#include "gtest/gtest.h"

#include "bb/DifferentiableLutN.h"
#include "bb/DifferentiableLutDiscreteN.h"
#include "bb/OptimizerAdam.h"
#include "bb/UniformDistributionGenerator.h"


#define MY_EXPECT_NEAR(a, b, th, rate)  EXPECT_NEAR((a), (b), (std::max((th), std::max(std::abs(a)*(rate), std::abs(b)*(rate)))))


#ifdef BB_WITH_CUDA


template<int N, typename BinType, typename Model0=bb::DifferentiableLutN<N, BinType>, typename Model1=bb::DifferentiableLutDiscreteN<N, BinType> >
void DifferentiableLutNTest_cmp(int const input_node_size, int const output_node_size, int const frame_size, int loop_num, bool lut_binarize= true, bool binary_mode=true, bool host_only=false)
{
    auto lut0 = Model0::Create(output_node_size);
    auto lut1 = Model1::Create(output_node_size);

    if ( host_only ) {
        lut1->SendCommand("host_only true");
    }

    auto opt0 = bb::OptimizerAdam<float>::Create();
    auto opt1 = bb::OptimizerAdam<float>::Create();

    if ( binary_mode ) {
        lut0->SendCommand("binary true");
        lut1->SendCommand("binary true");
    }
    else {
        lut0->SendCommand("binary false");
        lut1->SendCommand("binary false");
    }

    if ( lut_binarize ) {
        lut0->SendCommand("lut_binarize true");
        lut1->SendCommand("lut_binarize true");
    }
    else {
        lut0->SendCommand("lut_binarize false");
        lut1->SendCommand("lut_binarize false");
    }

    bb::FrameBuffer x_buf0(frame_size, {input_node_size}, bb::DataType<BinType>::type);
    bb::FrameBuffer x_buf1(frame_size, {input_node_size}, bb::DataType<BinType>::type);
    
    lut0->SetInputShape(x_buf0.GetShape());
    lut1->SetInputShape(x_buf1.GetShape());

    // 接続を同一化
    for (int node = 0; node < output_node_size; ++node) {
        for (int i = 0; i < N; ++i) {
            lut1->SetNodeConnectionIndex(node, i, lut1->GetNodeConnectionIndex(node, i));
        }
    }

    // 係数を同一化
    {
        auto W_ptr0 = lut0->lock_W_const();
        auto W_ptr1 = lut1->lock_W();
        for (int node = 0; node < output_node_size; ++node) {
            for (int i = 0; i < (1 << N); ++i) {
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
            auto x_ptr0 = x_buf0.Lock<BinType>();
            auto x_ptr1 = x_buf1.Lock<BinType>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < input_node_size; ++node ) {
                    if ( bb::DataType<BinType>::type == BB_TYPE_BIT ) {
                        bool val = (valgen->GetValue() > 0.5);
                        x_ptr0.Set(frame, node, val);
                        x_ptr1.Set(frame, node, val);
                    }
                    else {
                        BinType val = (BinType)valgen->GetValue();
                        x_ptr0.Set(frame, node, val);
                        x_ptr1.Set(frame, node, val);
                    }
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
            auto x_ptr0 = x_buf0.LockConst<BinType>();
            auto x_ptr1 = x_buf1.LockConst<BinType>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < input_node_size; ++node ) {
                    BinType val0 = x_ptr0.Get(frame, node);
                    BinType val1 = x_ptr1.Get(frame, node);
                    EXPECT_EQ(val0, val1);
                }
            }
        }

        {
            auto W_ptr0 = lut0->lock_W_const();
            auto W_ptr1 = lut1->lock_W_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < (1 << N); ++i) {
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
                MY_EXPECT_NEAR(val0, val1, 0.005f, 0.005f);
            }
        }


        {
            auto y_ptr0 = y_buf0.LockConst<BinType>();
            auto y_ptr1 = y_buf1.LockConst<BinType>();
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < output_node_size; ++node ) {
                    BinType val0 = y_ptr0.Get(frame, node);
                    BinType val1 = y_ptr1.Get(frame, node);
                    if ( bb::DataType<BinType>::type == BB_TYPE_BIT ) {
                        EXPECT_EQ(val0, val1);
                    }
                    else {
                        EXPECT_NEAR(val0, val1, 0.001f);
                    }
                }
            }
        }

        // backward
        bb::FrameBuffer dy_buf0(frame_size, {output_node_size}, BB_TYPE_FP32);
        bb::FrameBuffer dy_buf1(frame_size, {output_node_size}, BB_TYPE_FP32);
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
                EXPECT_NEAR(val0, val1, 0.1f);
                if (abs(val0 - val1) >= 0.1f) {
                    std::cout << frame << " " << node << std::endl;
 //                 getchar();
                }
            }
        }

        {
            auto W_ptr0 = lut0->lock_W_const();
            auto W_ptr1 = lut1->lock_W_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < (1 << N); ++i) {
                    auto val0 = W_ptr0(node, i);
                    auto val1 = W_ptr1(node, i);
                    EXPECT_NEAR(val0, val1, 0.001f);
                    if (std::abs(val0 - val1) >= 0.001f) {
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
                for (int i = 0; i < (1 << N); ++i) {
                    auto val0 = dW_ptr0(node, i);
                    auto val1 = dW_ptr1(node, i);
                    MY_EXPECT_NEAR(val0, val1, 0.005f, 0.005f);
//                  if ( !(abs(val0 - val1) < 0.01f) ) {
//                      std::cout << node << std::endl;
//                      getchar();
//                  }
                }
            }
        }

        opt0->Update();
        opt1->Update();

        {
            auto W_ptr0 = lut0->lock_W_const();
            auto W_ptr1 = lut1->lock_W_const();
            for (int node = 0; node < output_node_size; ++node) {
                for (int i = 0; i < (1 << N); ++i) {
                    auto val0 = W_ptr0(node, i);
                    auto val1 = W_ptr1(node, i);
                    EXPECT_NEAR(val0, val1, 0.001f);
                    if ( !(std::abs(val0 - val1) < 0.001f) ) {
                        auto dW_val0 = lut0->lock_dW_const();
                        auto dW_val1 = lut1->lock_dW_const();                      
                        std::cout <<  node << "W0: " << W_ptr0(node, i) << "  W1 : " << W_ptr1(node, i) << std::endl;
                        getchar();
                    }
                }
            }
        }

        for ( int node = 0; node < output_node_size; ++node )
        {
            std::vector<double> x_vec(N);
            for ( int i = 0; i < N; ++i ) {
                x_vec[i] = (double)valgen->GetValue();
            }
            auto y0_vec = lut0->ForwardNode(node, x_vec);
            auto y1_vec = lut1->ForwardNode(node, x_vec);
            EXPECT_NEAR(y0_vec[0], y1_vec[0], 0.0001);
        }
    }
}


TEST(DifferentiableLutNTest, testDifferentiableLutN_cmp_float)
{
    DifferentiableLutNTest_cmp<6, float>(6,    1,       1, 2, true,  true);
    DifferentiableLutNTest_cmp<6, float>(6,    1,    32+7, 2, true,  true);
    DifferentiableLutNTest_cmp<6, float>(6,    1,      64, 2, true,  true);
    DifferentiableLutNTest_cmp<6, float>(6,    1,    1024, 2, true,  true);
    DifferentiableLutNTest_cmp<6, float>(6,    1024,    1, 2, true,  true);
    DifferentiableLutNTest_cmp<6, float>(6,    2,      32, 2, true,  true);

    DifferentiableLutNTest_cmp<6, float>(6,    1,       1, 2, false, true);
    DifferentiableLutNTest_cmp<6, float>(6,    1,    32+7, 2, false, true);
    DifferentiableLutNTest_cmp<6, float>(6,    1,      64, 2, false, true);
    DifferentiableLutNTest_cmp<6, float>(6,    1,    1024, 2, false, true);
    DifferentiableLutNTest_cmp<6, float>(6,    1024,    1, 2, false, true);
    DifferentiableLutNTest_cmp<6, float>(6,    2,      32, 2, false, true);

    DifferentiableLutNTest_cmp<6, float>(6,    1,       1, 2, true,  false);
    DifferentiableLutNTest_cmp<6, float>(6,    1,    32+7, 2, true,  false);
    DifferentiableLutNTest_cmp<6, float>(6,    1,      64, 2, true,  false);
    DifferentiableLutNTest_cmp<6, float>(6,    1,    1024, 2, true,  false);
    DifferentiableLutNTest_cmp<6, float>(6,    1024,    1, 2, true,  false);
    DifferentiableLutNTest_cmp<6, float>(6,    2,      32, 2, true,  false);

    DifferentiableLutNTest_cmp<6, float>(6,    1,       1, 2, false, false);
    DifferentiableLutNTest_cmp<6, float>(6,    1,    32+7, 2, false, false);
    DifferentiableLutNTest_cmp<6, float>(6,    1,      64, 2, false, false);
    DifferentiableLutNTest_cmp<6, float>(6,    1,    1024, 2, false, false);
    DifferentiableLutNTest_cmp<6, float>(6,    1024,    1, 2, false, false);
    DifferentiableLutNTest_cmp<6, float>(6,    2,      32, 2, false, false);
}

TEST(DifferentiableLutNTest, testDifferentiableLutN_cmp_bit_1)
{
    DifferentiableLutNTest_cmp<6, bb::Bit>(6,  1,   32+7, 2, true);
}


TEST(DifferentiableLutNTest, testDifferentiableLutN_cmp_bit)
{
    DifferentiableLutNTest_cmp<6, bb::Bit>(6,  1,   32+7, 2, true);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6, 16,  128+3, 2, true);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6, 1,      64, 2, true);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6, 1,    1024, 2, true);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6, 1024,    1, 2, true);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6, 2,      32, 2, true);
    
    DifferentiableLutNTest_cmp<6, bb::Bit>(6,    1,   32+7, 2, false);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6,   16,    128, 2, false);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6,    1,     64, 2, false);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6,    1,   1024, 2, false);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6, 1024,      1, 2, false);
    DifferentiableLutNTest_cmp<6, bb::Bit>(6,    2,     32, 2, false);
}


TEST(DifferentiableLutNTest, testDifferentiableLutN_cmp_gpu)
{
//  DifferentiableLutNTest_cmp<6, float, bb::DifferentiableLutN<6, float>, bb::DifferentiableLutN<6, float>>(6,  1,  1, 2, true, true, true);
    
    DifferentiableLutNTest_cmp<6, float, bb::DifferentiableLutN<6, float>, bb::DifferentiableLutN<6, float>>(6, 16, 32, 2, true,  true,  true);
    DifferentiableLutNTest_cmp<6, float, bb::DifferentiableLutN<6, float>, bb::DifferentiableLutN<6, float>>(6, 16, 32, 2, false, true,  true);
    DifferentiableLutNTest_cmp<6, float, bb::DifferentiableLutN<6, float>, bb::DifferentiableLutN<6, float>>(6, 16, 32, 2, true,  false, true);
//  DifferentiableLutNTest_cmp<6, float, bb::DifferentiableLutN<6, float>, bb::DifferentiableLutN<6, float>>(6, 16, 32, 2, false, false, true);

//  DifferentiableLutNTest_cmp<6, bb::Bit, bb::DifferentiableLutN<6, bb::Bit>, bb::DifferentiableLutN<6, bb::Bit>>(6,  1,   1, 2, true,  true, true);

    DifferentiableLutNTest_cmp<6, bb::Bit, bb::DifferentiableLutN<6, bb::Bit>, bb::DifferentiableLutN<6, bb::Bit>>(6,  16, 32, 2, true,  true, true);
    DifferentiableLutNTest_cmp<6, bb::Bit, bb::DifferentiableLutN<6, bb::Bit>, bb::DifferentiableLutN<6, bb::Bit>>(6,  16, 32, 2, false, true, true);
}





TEST(DifferentiableLutNTest, testDifferentiableLut4_cmp_float)
{
    DifferentiableLutNTest_cmp<4, float>(4,    1,       1, 2, true,  true);
    DifferentiableLutNTest_cmp<4, float>(4,    1,    32+7, 2, true,  true);
    DifferentiableLutNTest_cmp<4, float>(4,    1,      64, 2, true,  true);
    DifferentiableLutNTest_cmp<4, float>(4,    1,    1024, 2, true,  true);
    DifferentiableLutNTest_cmp<4, float>(4,    1024,    1, 2, true,  true);
    DifferentiableLutNTest_cmp<4, float>(4,    2,      32, 2, true,  true);
    DifferentiableLutNTest_cmp<4, float>(4,    1,       1, 2, false, true);
    DifferentiableLutNTest_cmp<4, float>(4,    1,    32+7, 2, false, true);
    DifferentiableLutNTest_cmp<4, float>(4,    1,      64, 2, false, true);

//    DifferentiableLutNTest_cmp<4, float>(4,    1,    1024, 2, false, true);
//    DifferentiableLutNTest_cmp<4, float>(4,    1024,    1, 2, false, true);
//    DifferentiableLutNTest_cmp<4, float>(4,    2,      32, 2, false, true);
//    DifferentiableLutNTest_cmp<4, float>(4,    1,       1, 2, true,  false);
//    DifferentiableLutNTest_cmp<4, float>(4,    1,    32+7, 2, true,  false);
//    DifferentiableLutNTest_cmp<4, float>(4,    1,      64, 2, true,  false);
//    DifferentiableLutNTest_cmp<4, float>(4,    1,    1024, 2, true,  false);
//    DifferentiableLutNTest_cmp<4, float>(4,    1024,    1, 2, true,  false);
//    DifferentiableLutNTest_cmp<4, float>(4,    2,      32, 2, true,  false);
//    DifferentiableLutNTest_cmp<4, float>(4,    1,       1, 2, false, false);
//    DifferentiableLutNTest_cmp<4, float>(4,    1,    32+7, 2, false, false);
//    DifferentiableLutNTest_cmp<4, float>(4,    1,      64, 2, false, false);
//    DifferentiableLutNTest_cmp<4, float>(4,    1,    1024, 2, false, false);
//    DifferentiableLutNTest_cmp<4, float>(4,    1024,    1, 2, false, false);
//    DifferentiableLutNTest_cmp<4, float>(4,    2,      32, 2, false, false);
}


#endif


