#include <string>
#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "bb/BinaryLutN.h"
#include "bb/UniformDistributionGenerator.h"
#include "bb/NormalDistributionGenerator.h"


#if 0

TEST(BinaryLutTest, testBinaryLut)
{
    auto lut = bb::BinaryLutN<6>::Create(2);
    
    bb::FrameBuffer x_buf(1, 16, BB_TYPE_BIT);
    
    lut->SetInputShape(x_buf.GetShape());

    x_buf.SetBit(0, 0, false);
    x_buf.SetBit(0, 1, true);
    x_buf.SetBit(0, 2, true);
    x_buf.SetBit(0, 3, false);
    x_buf.SetBit(0, 4, false);
    x_buf.SetBit(0, 5, true);
    x_buf.SetBit(0, 6, true);
    x_buf.SetBit(0, 7, true);

    // 0x1d
    lut->SetNodeInput(0, 0, 6); // 1
    lut->SetNodeInput(0, 1, 4); // 0
    lut->SetNodeInput(0, 2, 1); // 1
    lut->SetNodeInput(0, 3, 7); // 1
    lut->SetNodeInput(0, 4, 2); // 1
    lut->SetNodeInput(0, 5, 3); // 0

    // 0x1c
    lut->SetNodeInput(1, 0, 0); // 0
    lut->SetNodeInput(1, 1, 4); // 0
    lut->SetNodeInput(1, 2, 1); // 1
    lut->SetNodeInput(1, 3, 5); // 1
    lut->SetNodeInput(1, 4, 6); // 1
    lut->SetNodeInput(1, 5, 3); // 0

    for (int i = 0; i < 64; i++) {
        lut->SetLutTable(0, i, i == 0x1d);
        lut->SetLutTable(1, i, i != 0x1c);
    }
    
    auto y_buf = lut->Forward(x_buf);

    EXPECT_EQ(true,  y_buf.GetBit(0, 0));
    EXPECT_EQ(false, y_buf.GetBit(0, 1));
}



TEST(NeuralNetBinaryLut6, testNeuralNetBinaryLut6Batch)
{
    auto lut = bb::BinaryLutN<6>::Create(2);
    
    bb::FrameBuffer x_buf(2, 16, BB_TYPE_BIT);
    
    lut->SetInputShape(x_buf.GetShape());

    x_buf.SetBit(0, 0, false);
    x_buf.SetBit(0, 1, true);
    x_buf.SetBit(0, 2, true);
    x_buf.SetBit(0, 3, false);
    x_buf.SetBit(0, 4, false);
    x_buf.SetBit(0, 5, true);
    x_buf.SetBit(0, 6, true);
    x_buf.SetBit(0, 7, true);

    x_buf.SetBit(1, 0, true);
    x_buf.SetBit(1, 1, false);
    x_buf.SetBit(1, 2, false);
    x_buf.SetBit(1, 3, true);
    x_buf.SetBit(1, 4, true);
    x_buf.SetBit(1, 5, false);
    x_buf.SetBit(1, 6, false);
    x_buf.SetBit(1, 7, false);


    // 0x1d
    lut->SetNodeInput(0, 0, 6); // 1
    lut->SetNodeInput(0, 1, 4); // 0
    lut->SetNodeInput(0, 2, 1); // 1
    lut->SetNodeInput(0, 3, 7); // 1
    lut->SetNodeInput(0, 4, 2); // 1
    lut->SetNodeInput(0, 5, 3); // 0

    // 0x1c
    lut->SetNodeInput(1, 0, 0); // 0
    lut->SetNodeInput(1, 1, 4); // 0
    lut->SetNodeInput(1, 2, 1); // 1
    lut->SetNodeInput(1, 3, 5); // 1
    lut->SetNodeInput(1, 4, 6); // 1
    lut->SetNodeInput(1, 5, 3); // 0

    for (int i = 0; i < 64; i++) {
        lut->SetLutTable(0, i, i == 0x1d || i == 0x22 );
        lut->SetLutTable(1, i, i != 0x1c && i != 0x23 );
    }

    auto y_buf = lut->Forward(x_buf);

    EXPECT_EQ(true,  y_buf.GetBit(0, 0));
    EXPECT_EQ(false, y_buf.GetBit(0, 1));
    EXPECT_EQ(true,  y_buf.GetBit(1, 0));
    EXPECT_EQ(false, y_buf.GetBit(1, 1));
}

#endif


template <int N = 6, typename FT = bb::Bit, typename BT = float>
void testBinaryLut6_cmpare(
        int loop_num,
        int input_node_size,
        int output_node_size,
        int frame_size)
{
    auto layer_cpu = bb::BinaryLutN<N, FT, BT>::Create(output_node_size);
    auto layer_gpu = bb::BinaryLutN<N, FT, BT>::Create(output_node_size);

    bb::FrameBuffer x_cpu(frame_size, {input_node_size}, bb::DataType<FT>::type, true);
    bb::FrameBuffer x_gpu(frame_size, {input_node_size}, bb::DataType<FT>::type);
    
    layer_cpu->SetInputShape(x_cpu.GetShape());
    layer_gpu->SetInputShape(x_gpu.GetShape());

    // パラメータ統一
    for ( int node = 0; node < output_node_size; ++node) {
        for ( int i = 0; i < N; ++i) {
            layer_gpu->SetNodeConnectionIndex(node, i, layer_cpu->GetNodeConnectionIndex(node, i));
        }

        for ( int i = 0; i < (1 << N); ++i) {
            layer_gpu->SetLutTable(node, i, layer_cpu->GetLutTable(node, i));
        }
    }

    layer_cpu->SendCommand("host_only true");

    auto valgen = bb::UniformDistributionGenerator<FT>::Create(0, 1, 1);

    for ( int loop = 0; loop < loop_num; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < x_cpu.GetNodeSize(); ++node ) {
                x_cpu.SetFP32(frame, node, valgen->GetValue());
                x_gpu.SetFP32(frame, node, x_cpu.GetFP32(frame, node));
            }
        }

        auto y_cpu = layer_cpu->Forward(x_cpu);
        auto y_gpu = layer_gpu->Forward(x_gpu);

        // 壊れていないことを一応確認
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < x_cpu.GetNodeSize(); ++node ) {
                auto val_cpu = x_cpu.GetFP32(frame, node);
                auto val_gpu = x_gpu.GetFP32(frame, node);
                EXPECT_EQ(val_cpu, val_gpu);
            }
        }

        // 結果比較
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < y_cpu.GetNodeSize(); ++node ) {
                auto val_cpu = y_cpu.GetFP32(frame, node);
                auto val_gpu = y_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
                if (val_cpu != val_gpu) {
                    std::cout << frame << " " << node << std::endl;
                }
            }
        }
    }
}



TEST(NeuralNetBinaryLut6, testLoweringConvolution_cmp_bit)
{
    testBinaryLut6_cmpare<6, bb::Bit, float>(2, 16, 16, 32);
}

