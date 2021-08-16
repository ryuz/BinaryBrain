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

    bb::FrameBuffer    x_buf(frame_size, {c_size, input_h_size, input_w_size }, BB_TYPE_FP32);

    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < input_h_size; ++y) {
                for (bb::index_t x = 0; x < input_w_size; ++x) {
                    x_buf.SetFP32(f, { c, y, x }, (float)(1000 * f + 100 * c + 10 * y + x));
                }
            }
        }
    }
    x_buf.SetFP32(0, { 0, 0, 1 }, 99);

    auto y_buf = maxpol->Forward(x_buf);
    
    EXPECT_EQ(bb::indices_t({3, 2, 2}), y_buf.GetShape());

    EXPECT_EQ(99,   y_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(15,   y_buf.GetFP32(0, { 0, 0, 1 }));
    EXPECT_EQ(32,   y_buf.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(35,   y_buf.GetFP32(0, { 0, 1, 1 }));

    EXPECT_EQ(112,  y_buf.GetFP32(0, { 1, 0, 0 }));
    EXPECT_EQ(115,  y_buf.GetFP32(0, { 1, 0, 1 }));
    EXPECT_EQ(132,  y_buf.GetFP32(0, { 1, 1, 0 }));
    EXPECT_EQ(135,  y_buf.GetFP32(0, { 1, 1, 1 }));

    EXPECT_EQ(212,  y_buf.GetFP32(0, { 2, 0, 0 }));
    EXPECT_EQ(215,  y_buf.GetFP32(0, { 2, 0, 1 }));
    EXPECT_EQ(232,  y_buf.GetFP32(0, { 2, 1, 0 }));
    EXPECT_EQ(235,  y_buf.GetFP32(0, { 2, 1, 1 }));

    EXPECT_EQ(1012, y_buf.GetFP32(1, { 0, 0, 0 }));
    EXPECT_EQ(1015, y_buf.GetFP32(1, { 0, 0, 1 }));
    EXPECT_EQ(1032, y_buf.GetFP32(1, { 0, 1, 0 }));
    EXPECT_EQ(1035, y_buf.GetFP32(1, { 0, 1, 1 }));

    EXPECT_EQ(1112, y_buf.GetFP32(1, { 1, 0, 0 }));
    EXPECT_EQ(1115, y_buf.GetFP32(1, { 1, 0, 1 }));
    EXPECT_EQ(1132, y_buf.GetFP32(1, { 1, 1, 0 }));
    EXPECT_EQ(1135, y_buf.GetFP32(1, { 1, 1, 1 }));

    EXPECT_EQ(1212, y_buf.GetFP32(1, { 2, 0, 0 }));
    EXPECT_EQ(1215, y_buf.GetFP32(1, { 2, 0, 1 }));
    EXPECT_EQ(1232, y_buf.GetFP32(1, { 2, 1, 0 }));
    EXPECT_EQ(1235, y_buf.GetFP32(1, { 2, 1, 1 }));

    // backward

    bb::FrameBuffer dy_buf(frame_size, {c_size, output_h_size, output_w_size}, BB_TYPE_FP32);

    for (bb::index_t f = 0; f < 2; ++f) {
        for (bb::index_t c = 0; c < 3; ++c) {
            for (bb::index_t y = 0; y < 2; ++y) {
                for (bb::index_t x = 0; x < 2; ++x) {
                    dy_buf.SetFP32(f, { c, y, x }, (float)(1000 * f + 100 * c + 10 * y + x + 1));
                }
            }
        }
    }

    auto dx_buf = maxpol->Backward(dy_buf);

    EXPECT_EQ(bb::indices_t({c_size, input_h_size, input_w_size}), dx_buf.GetShape());

    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(1,    dx_buf.GetFP32(0, { 0, 0, 1 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 0, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 1, 1 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 1, 2 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 0, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 0, 4 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 0, 5 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 1, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 1, 4 }));
    EXPECT_EQ(2,    dx_buf.GetFP32(0, { 0, 1, 5 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 2, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 2, 1 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 2, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 3, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 3, 1 }));
    EXPECT_EQ(11,   dx_buf.GetFP32(0, { 0, 3, 2 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 2, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 2, 4 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 2, 5 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 3, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 0, 3, 4 }));
    EXPECT_EQ(12,   dx_buf.GetFP32(0, { 0, 3, 5 }));


    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 0, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 0, 1 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 0, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 1, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 1, 1 }));
    EXPECT_EQ(101,  dx_buf.GetFP32(0, { 1, 1, 2 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 0, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 0, 4 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 0, 5 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 1, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 1, 4 }));
    EXPECT_EQ(102,  dx_buf.GetFP32(0, { 1, 1, 5 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 2, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 2, 1 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 2, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 3, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 3, 1 }));
    EXPECT_EQ(111,  dx_buf.GetFP32(0, { 1, 3, 2 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 2, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 2, 4 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 2, 5 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 3, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(0, { 1, 3, 4 }));
    EXPECT_EQ(112,  dx_buf.GetFP32(0, { 1, 3, 5 }));


    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 0, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 0, 1 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 0, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 1, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 1, 1 }));
    EXPECT_EQ(1201, dx_buf.GetFP32(1, { 2, 1, 2 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 0, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 0, 4 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 0, 5 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 1, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 1, 4 }));
    EXPECT_EQ(1202, dx_buf.GetFP32(1, { 2, 1, 5 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 2, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 2, 1 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 2, 2 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 3, 0 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 3, 1 }));
    EXPECT_EQ(1211, dx_buf.GetFP32(1, { 2, 3, 2 }));

    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 2, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 2, 4 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 2, 5 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 3, 3 }));
    EXPECT_EQ(0,    dx_buf.GetFP32(1, { 2, 3, 4 }));
    EXPECT_EQ(1212, dx_buf.GetFP32(1, { 2, 3, 5 }));

}

template<typename T>
void print2d(bb::FrameBuffer& buf)
{
    auto shape = buf.GetShape();
    for ( int y = 0; y < shape[1]; ++y) {
        for ( int x = 0; x < shape[2]; ++x) {
            auto v = buf.GetValue<T>(0, { 0, y, x });
            printf("%+2.2f ", v);
        }
        printf("\n");
    }
    printf("\n");
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

    bb::FrameBuffer x_cpu(frame_size, {c_size, input_h_size, input_w_size}, bb::DataType<FT>::type, true);
    bb::FrameBuffer x_gpu(frame_size, {c_size, input_h_size, input_w_size}, bb::DataType<FT>::type);

    maxpol_cpu->SetInputShape(x_cpu.GetShape());
    maxpol_gpu->SetInputShape(x_gpu.GetShape());

    auto val_gen = bb::NormalDistributionGenerator<double>::Create(0.0, 1.0);

    // forward
    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < input_h_size; ++y) {
                for (bb::index_t x = 0; x < input_w_size; ++x) {
                    FT val = (FT)val_gen->GetValue();
                    x_cpu.SetValue<FT>(f, { c, y, x }, val);
                    x_gpu.SetValue<FT>(f, { c, y, x }, val);
                }
            }
        }
    }

    EXPECT_EQ(bb::indices_t({c_size, output_h_size, output_w_size}), maxpol_cpu->GetOutputShape());
    EXPECT_EQ(bb::indices_t({c_size, output_h_size, output_w_size}), maxpol_gpu->GetOutputShape());

    auto y_cpu = maxpol_cpu->Forward(x_cpu);
    auto y_gpu = maxpol_gpu->Forward(x_gpu);

    EXPECT_EQ(bb::indices_t({c_size, output_h_size, output_w_size}), y_cpu.GetShape());
    EXPECT_EQ(bb::indices_t({c_size, output_h_size, output_w_size}), y_gpu.GetShape());

    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < output_h_size; ++y) {
                for (bb::index_t x = 0; x < output_w_size; ++x) {
                    auto val_cpu = y_cpu.GetValue<FT>(f, { c, y, x });
                    auto val_gpu = y_gpu.GetValue<FT>(f, { c, y, x });
                    EXPECT_EQ(val_cpu, val_gpu);
                }
            }
        }
    }


    // backward
    bb::FrameBuffer dy_cpu(frame_size, {c_size, output_h_size, output_w_size}, bb::DataType<BT>::type, true);
    bb::FrameBuffer dy_gpu(frame_size, {c_size, output_h_size, output_w_size}, bb::DataType<BT>::type);
    
    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < output_h_size; ++y) {
                for (bb::index_t x = 0; x < output_w_size; ++x) {
                    FT val = (FT)val_gen->GetValue();
                    dy_cpu.SetValue<FT>(f, { c, y, x }, val);
                    dy_gpu.SetValue<FT>(f, { c, y, x }, val);
                }
            }
        }
    }
    
    auto dx_cpu = maxpol_cpu->Backward(dy_cpu);
    auto dx_gpu = maxpol_gpu->Backward(dy_gpu);

    /*
    printf("<y_cpu>\n");
    print2d<float>(y_cpu);
    printf("<y_gpu>\n");
    print2d<float>(y_gpu);

    printf("<x_cpu>\n");
    print2d<float>(x_cpu);
    printf("<x_gpu>\n");
    print2d<float>(x_gpu);

    printf("<dy_cpu>\n");
    print2d<float>(dy_cpu);
    printf("<dy_gpu>\n");
    print2d<float>(dy_gpu);

    printf("<dx_cpu>\n");
    print2d<float>(dx_cpu);
    printf("<dx_gpu>\n");
    print2d<float>(dx_gpu);
    */

    EXPECT_EQ(bb::indices_t({c_size, input_h_size, input_w_size}), dx_cpu.GetShape());
    EXPECT_EQ(bb::indices_t({c_size, input_h_size, input_w_size}), dx_gpu.GetShape());

    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < input_h_size; ++y) {
                for (bb::index_t x = 0; x < input_w_size; ++x) {

                    auto x_val_cpu = x_cpu.GetValue<FT>(f, { c, y, x });
                    auto x_val_gpu = x_gpu.GetValue<FT>(f, { c, y, x });
                    EXPECT_EQ(x_val_cpu, x_val_gpu);

                    auto val_cpu = dx_cpu.GetValue<FT>(f, { c, y, x });
                    auto val_gpu = dx_gpu.GetValue<FT>(f, { c, y, x });
                    EXPECT_EQ(val_cpu, val_gpu);

//                 std::cout << " f:" << f  << " c:" << c  << " y:" << y  << " x:" << x << std::endl;
//                 printf("[%02d][%02d] (%f %f), (%f, %f)\n", (int)y, (int)x, val_cpu, x_val_cpu, val_gpu, x_val_gpu);
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


TEST(MaxPoolingTest, testMaxPooling_stack)
{
    int frame_size = 17;

    bb::FrameBuffer x0(frame_size, {32, 32*2, 72*2}, BB_TYPE_BIT);
    bb::FrameBuffer x1(frame_size, {32, 16*2, 24*2}, BB_TYPE_BIT);
    bb::FrameBuffer x2(frame_size, {32,  8*2, 12*2}, BB_TYPE_BIT);
    bb::FrameBuffer x3(frame_size, {32, 33*2, 17*2}, BB_TYPE_BIT);

    bb::FrameBuffer dy0(frame_size, {32, 32, 72}, BB_TYPE_FP32);
    bb::FrameBuffer dy1(frame_size, {32, 16, 24}, BB_TYPE_FP32);
    bb::FrameBuffer dy2(frame_size, {32,  8, 12}, BB_TYPE_FP32);
    bb::FrameBuffer dy3(frame_size, {32, 33, 17}, BB_TYPE_FP32);

    auto pool = bb::MaxPooling<bb::Bit>::Create(2, 2);
    pool->SetInputShape(x0.GetShape());
    auto y0 = pool->Forward(x0);
    auto y1 = pool->Forward(x1);
    auto y2 = pool->Forward(x2);
    auto y3 = pool->Forward(x3);

    auto dx3 = pool->Backward(dy3);
    auto dx2 = pool->Backward(dy2);
    auto dx1 = pool->Backward(dy1);
    auto dx0 = pool->Backward(dy0);

    EXPECT_EQ(y0.GetShape(), dy0.GetShape());
    EXPECT_EQ(y1.GetShape(), dy1.GetShape());
    EXPECT_EQ(y2.GetShape(), dy2.GetShape());
    EXPECT_EQ(y3.GetShape(), dy3.GetShape());
    EXPECT_EQ(dx0.GetShape(), x0.GetShape());
    EXPECT_EQ(dx1.GetShape(), x1.GetShape());
    EXPECT_EQ(dx2.GetShape(), x2.GetShape());
    EXPECT_EQ(dx3.GetShape(), x3.GetShape());
}






TEST(MaxPoolingTest, testMaxPooling_minus)
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

    bb::FrameBuffer    x_buf(frame_size, {c_size, input_h_size, input_w_size }, BB_TYPE_FP32);

    for (bb::index_t f = 0; f < frame_size; ++f) {
        for (bb::index_t c = 0; c < c_size; ++c) {
            for (bb::index_t y = 0; y < input_h_size; ++y) {
                for (bb::index_t x = 0; x < input_w_size; ++x) {
                    x_buf.SetFP32(f, { c, y, x }, (float)-10);
                }
            }
        }
    }
    x_buf.SetFP32(0, { 0, 0, 0 }, -3);

    auto y_buf = maxpol->Forward(x_buf);
    
    EXPECT_EQ(bb::indices_t({3, 2, 2}), y_buf.GetShape());

    EXPECT_EQ( -3,   y_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(-10,   y_buf.GetFP32(1, { 0, 0, 0 }));
    EXPECT_EQ(-10,   y_buf.GetFP32(0, { 0, 0, 1 }));


}

#endif

