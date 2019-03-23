#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/BinaryToReal.h"
#include "bb/NormalDistributionGenerator.h"

#if 0

TEST(BinaryToRealTest, testBinaryToReal_Bit)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int frame_size = 1;

    auto bin2real = bb::BinaryToReal<bb::Bit, float, float>::Create(bb::indices_t({node_size}), mux_size);
    
//	bb::NeuralNetBinaryToReal<> bin2real(node_size, node_size);
//	testSetupLayerBuffer(bin2real);

//	bin2real.SetMuxSize(mux_size);
//	bin2real.SetBatchSize(1);

    bb::FrameBuffer x_buf(BB_TYPE_BIT, frame_size*mux_size, node_size);

    bin2real->SetInputShape(x_buf.GetShape());

   	x_buf.SetBit(0, 0, true);
	x_buf.SetBit(1, 0, true);
	x_buf.SetBit(0, 1, false);
	x_buf.SetBit(1, 1, false);
	x_buf.SetBit(0, 2, false);
	x_buf.SetBit(1, 2, true);

    auto y_buf = bin2real->Forward(x_buf);

	EXPECT_EQ(2, x_buf.GetFrameSize());
	EXPECT_EQ(1, y_buf.GetFrameSize());

	EXPECT_EQ(1.0, y_buf.GetFP32(0, 0));
	EXPECT_EQ(0.0, y_buf.GetFP32(0, 1));
	EXPECT_EQ(0.5, y_buf.GetFP32(0, 2));
	

    // backward
//	auto out_err = bin2real.GetOutputErrorBuffer();
//	auto in_err = bin2real.GetInputErrorBuffer();

    bb::FrameBuffer dy_buf(BB_TYPE_FP32, frame_size, node_size);

	dy_buf.SetFP32(0, 0, 0.0f);
	dy_buf.SetFP32(0, 1, 1.0f);
	dy_buf.SetFP32(0, 2, 0.5f);

	auto dx_buf = bin2real->Backward(dy_buf);

	EXPECT_EQ(0.0f, dx_buf.GetFP32(0, 0));
	EXPECT_EQ(0.0f, dx_buf.GetFP32(1, 0));
	EXPECT_EQ(1.0f, dx_buf.GetFP32(0, 1));
	EXPECT_EQ(1.0f, dx_buf.GetFP32(1, 1));
	EXPECT_EQ(0.5f, dx_buf.GetFP32(0, 2));
	EXPECT_EQ(0.5f, dx_buf.GetFP32(1, 2));
}


TEST(BinaryToRealTest, testBinaryToReal_fp32)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int frame_size = 1;

    auto bin2real = bb::BinaryToReal<float, float, float>::Create(bb::indices_t({node_size}), mux_size);

    bb::FrameBuffer x_buf(BB_TYPE_FP32, frame_size*mux_size, node_size);

    bin2real->SetInputShape(x_buf.GetShape());

   	x_buf.SetFP32(0, 0, 1.0f);
	x_buf.SetFP32(1, 0, 1.0f);
	x_buf.SetFP32(0, 1, 0.0f);
	x_buf.SetFP32(1, 1, 0.0f);
	x_buf.SetFP32(0, 2, 0.0f);
	x_buf.SetFP32(1, 2, 1.0f);

    auto y_buf = bin2real->Forward(x_buf);

	EXPECT_EQ(2, x_buf.GetFrameSize());
	EXPECT_EQ(1, y_buf.GetFrameSize());

	EXPECT_EQ(1.0, y_buf.GetFP32(0, 0));
	EXPECT_EQ(0.0, y_buf.GetFP32(0, 1));
	EXPECT_EQ(0.5, y_buf.GetFP32(0, 2));
	

    // backward
//	auto out_err = bin2real.GetOutputErrorBuffer();
//	auto in_err = bin2real.GetInputErrorBuffer();

    bb::FrameBuffer dy_buf(BB_TYPE_FP32, frame_size, node_size);

	dy_buf.SetFP32(0, 0, 0.0f);
	dy_buf.SetFP32(0, 1, 1.0f);
	dy_buf.SetFP32(0, 2, 0.5f);

	auto dx_buf = bin2real->Backward(dy_buf);

	EXPECT_EQ(0.0f, dx_buf.GetFP32(0, 0));
	EXPECT_EQ(0.0f, dx_buf.GetFP32(1, 0));
	EXPECT_EQ(1.0f, dx_buf.GetFP32(0, 1));
	EXPECT_EQ(1.0f, dx_buf.GetFP32(1, 1));
	EXPECT_EQ(0.5f, dx_buf.GetFP32(0, 2));
	EXPECT_EQ(0.5f, dx_buf.GetFP32(1, 2));

 
}

#endif


#ifdef BB_WITH_CUDA

TEST(BinaryToRealTest, testBinaryToRealTest_cmp)
{
	const int node_mux_size  = 3;
	const int frame_mux_size = 7;
 	const int y_node_size  = 7;
	const int y_frame_size = 15;
 	const int x_node_size  = y_node_size  * node_mux_size;
	const int x_frame_size = y_frame_size * frame_mux_size;

    auto bin2real_cpu = bb::BinaryToReal<float, float, float>::Create(bb::indices_t({y_node_size}), frame_mux_size);
    auto bin2real_gpu = bb::BinaryToReal<float, float, float>::Create(bb::indices_t({y_node_size}), frame_mux_size);

    bb::FrameBuffer x_cpu(BB_TYPE_FP32, x_frame_size, x_node_size, true);
    bb::FrameBuffer x_gpu(BB_TYPE_FP32, x_frame_size, x_node_size);
    
    bin2real_cpu->SetInputShape(x_cpu.GetShape());
    bin2real_gpu->SetInputShape(x_gpu.GetShape());

    auto valgen = bb::NormalDistributionGenerator<float>::Create(1.2f, 3.3f, 1);

    for ( int loop = 0; loop < 1; ++ loop ) 
    {
        for ( int frame = 0; frame < x_frame_size; ++frame) {
            for ( int node = 0; node < x_node_size; ++node ) {
                x_cpu.SetFP32(frame, node, valgen->GetValue());
                x_gpu.SetFP32(frame, node, x_cpu.GetFP32(frame, node));
            }
        }

        auto y_cpu = bin2real_cpu->Forward(x_cpu);
        auto y_gpu = bin2real_gpu->Forward(x_gpu);

        // 入力側のチェック(おまけ)
        for ( int frame = 0; frame < x_frame_size; ++frame) {
            for ( int node = 0; node < x_node_size; ++node ) {
                auto val_cpu = x_cpu.GetFP32(frame, node);
                auto val_gpu = x_gpu.GetFP32(frame, node);
                EXPECT_FLOAT_EQ(val_cpu, val_gpu);
            }
        }

        // 出力比較
        for ( int frame = 0; frame < y_frame_size; ++frame) {
            for ( int node = 0; node < y_node_size; ++node ) {
                auto val_cpu = y_cpu.GetFP32(frame, node);
                auto val_gpu = y_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
            }
        }


        // backward
        bb::FrameBuffer dy_cpu(BB_TYPE_FP32, y_frame_size, y_node_size, true);
        bb::FrameBuffer dy_gpu(BB_TYPE_FP32, y_frame_size, y_node_size);
        for ( int frame = 0; frame < y_frame_size; ++frame) {
            for ( int node = 0; node < y_node_size; ++node ) {
                dy_cpu.SetFP32(frame, node, valgen->GetValue());
                dy_gpu.SetFP32(frame, node, dy_cpu.GetFP32(frame, node));
            }
        }

        auto dx_cpu = bin2real_cpu->Backward(dy_cpu);
        auto dx_gpu = bin2real_gpu->Backward(dy_gpu);

        // 結果比較
        for ( int frame = 0; frame < x_frame_size; ++frame) {
            for ( int node = 0; node < x_node_size; ++node ) {
                auto val_cpu = dx_cpu.GetFP32(frame, node);
                auto val_gpu = dx_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
            }
        }
    }
}

#endif

