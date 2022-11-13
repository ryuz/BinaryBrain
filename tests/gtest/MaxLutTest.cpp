#include <string>
#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "bb/MaxLut.h"

#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/Utility.h"




TEST(MaxLutTest, testMaxLut_001)
{
    auto lut = bb::MaxLut<bb::Bit>::Create(6, {2}, "serial");
    bb::FrameBuffer x_buf(3, {2, 6}, BB_TYPE_BIT);
    lut->SetInputShape(x_buf.GetShape());

//  lut->SendCommand("host_only true");

    x_buf.SetBit(0, {0, 0}, false);
    x_buf.SetBit(0, {0, 1}, false);
    x_buf.SetBit(0, {0, 2}, true);
    x_buf.SetBit(0, {0, 3}, false);
    x_buf.SetBit(0, {0, 4}, false);
    x_buf.SetBit(0, {0, 5}, false);
    x_buf.SetBit(0, {1, 0}, false);
    x_buf.SetBit(0, {1, 1}, false);
    x_buf.SetBit(0, {1, 2}, false);
    x_buf.SetBit(0, {1, 3}, false);
    x_buf.SetBit(0, {1, 4}, false);
    x_buf.SetBit(0, {1, 5}, false);

    x_buf.SetBit(1, {0, 0}, false);
    x_buf.SetBit(1, {0, 1}, true);
    x_buf.SetBit(1, {0, 2}, false);
    x_buf.SetBit(1, {0, 3}, false);
    x_buf.SetBit(1, {0, 4}, false);
    x_buf.SetBit(1, {0, 5}, true);
    x_buf.SetBit(1, {1, 0}, false);
    x_buf.SetBit(1, {1, 1}, false);
    x_buf.SetBit(1, {1, 2}, false);
    x_buf.SetBit(1, {1, 3}, false);
    x_buf.SetBit(1, {1, 4}, false);
    x_buf.SetBit(1, {1, 5}, true);

    x_buf.SetBit(2, {0, 0}, false);
    x_buf.SetBit(2, {0, 1}, false);
    x_buf.SetBit(2, {0, 2}, false);
    x_buf.SetBit(2, {0, 3}, false);
    x_buf.SetBit(2, {0, 4}, false);
    x_buf.SetBit(2, {0, 5}, false);
    x_buf.SetBit(2, {1, 0}, true);
    x_buf.SetBit(2, {1, 1}, false);
    x_buf.SetBit(2, {1, 2}, false);
    x_buf.SetBit(2, {1, 3}, false);
    x_buf.SetBit(2, {1, 4}, false);
    x_buf.SetBit(2, {1, 5}, false);

    auto y_buf = lut->Forward(x_buf);
    EXPECT_EQ(true,  y_buf.GetBit(0, 0));
    EXPECT_EQ(true,  y_buf.GetBit(1, 0));
    EXPECT_EQ(false, y_buf.GetBit(2, 0));
    EXPECT_EQ(false, y_buf.GetBit(0, 1));
    EXPECT_EQ(true,  y_buf.GetBit(1, 1));
    EXPECT_EQ(true,  y_buf.GetBit(2, 1));

    bb::FrameBuffer dy_buf(3, {2}, BB_TYPE_FP32);
    dy_buf.SetFP32(0, {0}, 0.1f);
    dy_buf.SetFP32(1, {0}, 0.2f);
    dy_buf.SetFP32(2, {0}, 0.3f);
    dy_buf.SetFP32(0, {1}, 0.4f);
    dy_buf.SetFP32(1, {1}, 0.5f);
    dy_buf.SetFP32(2, {1}, 0.6f);

    auto dx_buf = lut->Backward(dy_buf);
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {0, 0}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {0, 1}));
    EXPECT_EQ(0.1f, dx_buf.GetFP32(0, {0, 2}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {0, 3}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {0, 4}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {0, 5}));

    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {0, 0}));
    EXPECT_EQ(0.2f, dx_buf.GetFP32(1, {0, 1}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {0, 2}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {0, 3}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {0, 4}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {0, 5}));

    EXPECT_EQ(0.3f, dx_buf.GetFP32(2, {0, 0}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {0, 1}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {0, 2}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {0, 3}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {0, 4}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {0, 5}));


    EXPECT_EQ(0.4f, dx_buf.GetFP32(0, {1, 0}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {1, 1}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {1, 2}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {1, 3}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {1, 4}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(0, {1, 5}));

    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {1, 0}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {1, 1}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {1, 2}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {1, 3}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(1, {1, 4}));
    EXPECT_EQ(0.5f, dx_buf.GetFP32(1, {1, 5}));

    EXPECT_EQ(0.6f, dx_buf.GetFP32(2, {1, 0}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {1, 1}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {1, 2}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {1, 3}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {1, 4}));
    EXPECT_EQ(0.0f, dx_buf.GetFP32(2, {1, 5}));
}


#if 0

TEST(MaxLutTest, testMaxLut_002)
{
    const int frame_size       = 1230;
    const int output_node_size = 100;
//    const int frame_size       = 1;
//    const int output_node_size = 2;
    const int input_node_size  = output_node_size*2;

    std::mt19937_64                     mt(1);
    std::uniform_int_distribution<int>  dist(0, 10);

    auto lut_gpu = bb::MaxLut<bb::Bit>::Create(6, {output_node_size}, "serial");
    auto lut_cpu = bb::MaxLut<bb::Bit>::Create(6, {output_node_size}, "serial");
    lut_cpu->SendCommand("host_only true");

    bb::FrameBuffer x_gpu(frame_size, {input_node_size}, BB_TYPE_BIT);
    bb::FrameBuffer x_cpu(frame_size, {input_node_size}, BB_TYPE_BIT, true);
    lut_gpu->SetInputShape(x_gpu.GetShape());
    lut_cpu->SetInputShape(x_cpu.GetShape());
    
    for ( int f = 0; f < frame_size; ++f ) {
        for ( int n = 0; n < input_node_size; ++n ) {
            bool v = (dist(mt) == 0);
            x_gpu.SetBit(f, n, v);
            x_cpu.SetBit(f, n, v);
//            printf("[x] (%d %d) : %d\n", f, n, v);
        }
    }

    auto y_gpu = lut_gpu->Forward(x_gpu);
    auto y_cpu = lut_cpu->Forward(x_cpu);

    for ( int f = 0; f < frame_size; ++f ) {
        for ( int n = 0; n < output_node_size; ++n ) {
            EXPECT_EQ(y_cpu.GetBit(f, n), y_gpu.GetBit(f, n));
//            printf("[y] (%d %d) : %d %d\n", f, n, (int)y_gpu.GetBit(f, n), (int)y_cpu.GetBit(f, n));
        }
    }


    // backward
    bb::FrameBuffer dy_gpu(frame_size, {output_node_size}, BB_TYPE_FP32);
    bb::FrameBuffer dy_cpu(frame_size, {output_node_size}, BB_TYPE_FP32, true);

    for ( int f = 0; f < frame_size; ++f ) {
        for ( int n = 0; n < output_node_size; ++n ) {
            float v = (float)dist(mt);
            dy_gpu.SetFP32(f, n, v);
            dy_cpu.SetFP32(f, n, v);
        }
    }

    auto dx_gpu = lut_gpu->Backward(dy_gpu);
    auto dx_cpu = lut_cpu->Backward(dy_cpu);

    for ( int f = 0; f < frame_size; ++f ) {
        for ( int n = 0; n < input_node_size; ++n ) {
            EXPECT_EQ(dx_cpu.GetFP32(f, n), dx_gpu.GetFP32(f, n));
        }
    }

}

#endif
