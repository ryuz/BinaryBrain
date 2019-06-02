#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/MaxPooling.h"
#include "bb/NormalDistributionGenerator.h"


TEST(MaxPoolingTest, testMaxPoolingTest)
{
    bb::index_t const frame_size = 2;
    bb::index_t const c_size = 3;
    bb::index_t const input_h_size  = 4;
    bb::index_t const input_w_size  = 6;
    bb::index_t const filter_h_size = 2;
    bb::index_t const filter_w_size = 3;
    bb::index_t const output_h_size = input_h_size / filter_h_size;
    bb::index_t const output_w_size = input_w_size / filter_w_size;

    auto maxpol = bb::MaxPooling<>::Create(filter_h_size, filter_w_size);

    bb::FrameBuffer    x_buf(BB_TYPE_FP32, frame_size, {input_w_size, input_h_size, c_size});

    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < input_h_size; ++y) {
                for (bb::index_t x = 0; x < input_w_size; ++x) {
                    x_buf.SetFP32(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
                }
            }
        }
    }
    x_buf.SetFP32(0, { 1, 0, 0 }, 99);

    auto y_buf = maxpol->Forward(x_buf);
    
    EXPECT_EQ(bb::indices_t({2, 2, 3}), y_buf.GetShape());

    EXPECT_EQ(99, y_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(15, y_buf.GetFP32(0, { 1, 0, 0 }));
    EXPECT_EQ(32, y_buf.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(35, y_buf.GetFP32(0, { 1, 1, 0 }));

    EXPECT_EQ(112, y_buf.GetFP32(0, { 0, 0, 1 }));
    EXPECT_EQ(115, y_buf.GetFP32(0, { 1, 0, 1 }));
    EXPECT_EQ(132, y_buf.GetFP32(0, { 0, 1, 1 }));
    EXPECT_EQ(135, y_buf.GetFP32(0, { 1, 1, 1 }));

    EXPECT_EQ(212, y_buf.GetFP32(0, { 0, 0, 2 }));
    EXPECT_EQ(215, y_buf.GetFP32(0, { 1, 0, 2 }));
    EXPECT_EQ(232, y_buf.GetFP32(0, { 0, 1, 2 }));
    EXPECT_EQ(235, y_buf.GetFP32(0, { 1, 1, 2 }));

    EXPECT_EQ(1012, y_buf.GetFP32(1, { 0, 0, 0 }));
    EXPECT_EQ(1015, y_buf.GetFP32(1, { 1, 0, 0 }));
    EXPECT_EQ(1032, y_buf.GetFP32(1, { 0, 1, 0 }));
    EXPECT_EQ(1035, y_buf.GetFP32(1, { 1, 1, 0 }));

    EXPECT_EQ(1112, y_buf.GetFP32(1, { 0, 0, 1 }));
    EXPECT_EQ(1115, y_buf.GetFP32(1, { 1, 0, 1 }));
    EXPECT_EQ(1132, y_buf.GetFP32(1, { 0, 1, 1 }));
    EXPECT_EQ(1135, y_buf.GetFP32(1, { 1, 1, 1 }));

    EXPECT_EQ(1212, y_buf.GetFP32(1, { 0, 0, 2 }));
    EXPECT_EQ(1215, y_buf.GetFP32(1, { 1, 0, 2 }));
    EXPECT_EQ(1232, y_buf.GetFP32(1, { 0, 1, 2 }));
    EXPECT_EQ(1235, y_buf.GetFP32(1, { 1, 1, 2 }));

    // backward

    bb::FrameBuffer dy_buf(BB_TYPE_FP32, frame_size, {output_w_size, output_h_size, c_size});

    for (bb::index_t f = 0; f < 2; ++f) {
        for (bb::index_t c = 0; c < 3; ++c) {
            for (bb::index_t y = 0; y < 2; ++y) {
                for (bb::index_t x = 0; x < 2; ++x) {
                    dy_buf.SetFP32(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x + 1));
                }
            }
        }
    }

    auto dx_buf = maxpol->Backward(dy_buf);

    EXPECT_EQ(bb::indices_t({input_w_size, input_h_size, c_size}), dx_buf.GetShape());

    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(1,  dx_buf.GetFP32(0, { 1, 0, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 2, 0, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 1, 1, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 2, 1, 0 }));
                  
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 3, 0, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 4, 0, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 5, 0, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 3, 1, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 4, 1, 0 }));
    EXPECT_EQ(2,  dx_buf.GetFP32(0, { 5, 1, 0 }));

    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 0, 2, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 1, 2, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 2, 2, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 0, 3, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 1, 3, 0 }));
    EXPECT_EQ(11, dx_buf.GetFP32(0, { 2, 3, 0 }));

    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 3, 2, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 4, 2, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 5, 2, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 3, 3, 0 }));
    EXPECT_EQ(0,  dx_buf.GetFP32(0, { 4, 3, 0 }));
    EXPECT_EQ(12, dx_buf.GetFP32(0, { 5, 3, 0 }));

    //
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 0, 0, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 1, 0, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 2, 0, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 0, 1, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 1, 1, 1 }));
    EXPECT_EQ(101, dx_buf.GetFP32(0, { 2, 1, 1 }));

    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 3, 0, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 4, 0, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 5, 0, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 3, 1, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 4, 1, 1 }));
    EXPECT_EQ(102, dx_buf.GetFP32(0, { 5, 1, 1 }));

    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 0, 2, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 1, 2, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 2, 2, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 0, 3, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 1, 3, 1 }));
    EXPECT_EQ(111, dx_buf.GetFP32(0, { 2, 3, 1 }));

    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 3, 2, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 4, 2, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 5, 2, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 3, 3, 1 }));
    EXPECT_EQ(0,   dx_buf.GetFP32(0, { 4, 3, 1 }));
    EXPECT_EQ(112, dx_buf.GetFP32(0, { 5, 3, 1 }));


    //
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 0, 0, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 1, 0, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 0, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 0, 1, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 1, 1, 2 }));
    EXPECT_EQ(1201, dx_buf.GetFP32(1, { 2, 1, 2 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 3, 0, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 4, 0, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 5, 0, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 3, 1, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 4, 1, 2 }));
    EXPECT_EQ(1202, dx_buf.GetFP32(1, { 5, 1, 2 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 0, 2, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 1, 2, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 2, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 0, 3, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 1, 3, 2 }));
    EXPECT_EQ(1211, dx_buf.GetFP32(1, { 2, 3, 2 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 3, 2, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 4, 2, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 5, 2, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 3, 3, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 4, 3, 2 }));
    EXPECT_EQ(1212, dx_buf.GetFP32(1, { 5, 3, 2 }));

}


#ifdef BB_WITH_CUDA

// CPU版とGPU版で結果比較
template<typename FT=float, typename BT=float>
void MaxPoolingTest_Compare(
        bb::index_t const frame_size = 1221,
        bb::index_t const c_size = 3,
        bb::index_t const input_h_size  = 12,
        bb::index_t const input_w_size  = 8,
        bb::index_t const filter_h_size = 3,
        bb::index_t const filter_w_size = 2
    )
{
    bb::index_t const output_h_size = (input_h_size + filter_h_size - 1) / filter_h_size;
    bb::index_t const output_w_size = (input_w_size + filter_w_size - 1) / filter_w_size;

    auto maxpol_cpu = bb::MaxPooling<FT, BT>::Create(filter_h_size, filter_w_size);
    auto maxpol_gpu = bb::MaxPooling<FT, BT>::Create(filter_h_size, filter_w_size);

    bb::FrameBuffer x_cpu(bb::DataType<FT>::type, frame_size, {input_w_size, input_h_size, c_size}, true);
    bb::FrameBuffer x_gpu(bb::DataType<FT>::type, frame_size, {input_w_size, input_h_size, c_size});

    maxpol_cpu->SetInputShape(x_cpu.GetShape());
    maxpol_gpu->SetInputShape(x_gpu.GetShape());

    auto val_gen = bb::NormalDistributionGenerator<double>::Create(0.0, 1.0);

    // forward
    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < input_h_size; ++y) {
                for (bb::index_t x = 0; x < input_w_size; ++x) {
                    FT val = (FT)val_gen->GetValue();
                    x_cpu.SetValue<FT>(f, { x, y, c }, val);
                    x_gpu.SetValue<FT>(f, { x, y, c }, val);
                }
            }
        }
    }

    EXPECT_EQ(bb::indices_t({output_w_size, output_h_size, c_size}), maxpol_cpu->GetOutputShape());
    EXPECT_EQ(bb::indices_t({output_w_size, output_h_size, c_size}), maxpol_gpu->GetOutputShape());

    auto y_cpu = maxpol_cpu->Forward(x_cpu);
    auto y_gpu = maxpol_gpu->Forward(x_gpu);

    EXPECT_EQ(bb::indices_t({output_w_size, output_h_size, c_size}), y_cpu.GetShape());
    EXPECT_EQ(bb::indices_t({output_w_size, output_h_size, c_size}), y_gpu.GetShape());

    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < output_h_size; ++y) {
                for (bb::index_t x = 0; x < output_w_size; ++x) {
                    auto val_cpu = y_cpu.GetValue<FT>(f, { x, y, c });
                    auto val_gpu = y_gpu.GetValue<FT>(f, { x, y, c });
                    EXPECT_EQ(val_cpu, val_gpu);
                }
            }
        }
    }



    // backward
    bb::FrameBuffer dy_cpu(bb::DataType<BT>::type, frame_size, {output_w_size, output_h_size, c_size}, true);
    bb::FrameBuffer dy_gpu(bb::DataType<BT>::type, frame_size, {output_w_size, output_h_size, c_size});
    
    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < output_h_size; ++y) {
                for (bb::index_t x = 0; x < output_w_size; ++x) {
                    FT val = (FT)val_gen->GetValue();
                    dy_cpu.SetValue<FT>(f, { x, y, c }, val);
                    dy_gpu.SetValue<FT>(f, { x, y, c }, val);
                }
            }
        }
    }
    
    auto dx_cpu = maxpol_cpu->Backward(dy_cpu);
    auto dx_gpu = maxpol_gpu->Backward(dy_gpu);

    EXPECT_EQ(bb::indices_t({input_w_size, input_h_size, c_size}), dx_cpu.GetShape());
    EXPECT_EQ(bb::indices_t({input_w_size, input_h_size, c_size}), dx_gpu.GetShape());

    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < input_h_size; ++y) {
                for (bb::index_t x = 0; x < input_w_size; ++x) {
//                  std::cout << " f:" << f  << " c:" << c  << " y:" << y  << " x:" << x << std::endl;
                    auto val_cpu = dx_cpu.GetValue<FT>(f, { x, y, c });
                    auto val_gpu = dx_gpu.GetValue<FT>(f, { x, y, c });
                    EXPECT_EQ(val_cpu, val_gpu);
                }
            }
        }
    }

}


TEST(MaxPoolingTest, testMaxPooling_cmp_fp32_fp32)
{
    MaxPoolingTest_Compare<float, float>();
    MaxPoolingTest_Compare<bb::Bit, float>();
}


#endif

