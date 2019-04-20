#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <valarray>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gtest/gtest.h"

#if BB_WITH_CUDA

#include "bb/FrameBuffer.h"
#include "bbcu/bbcu.h"


TEST(cudaMatrixColwiseSumTest, test_cudaMatrixColwiseSum)
{
    int const node_size  = 2;
    int const frame_size = 3;

    bb::FrameBuffer x_buf(BB_TYPE_FP32, frame_size, node_size);
    bb::Tensor      y_buf(BB_TYPE_FP32, node_size);

    {
        x_buf.SetFP32(0, 0, 1);
        x_buf.SetFP32(1, 0, 2);
        x_buf.SetFP32(2, 0, 3);

        x_buf.SetFP32(0, 1, 4);
        x_buf.SetFP32(1, 1, 5);
        x_buf.SetFP32(2, 1, 6);
    }

    {
        auto x_ptr = x_buf.LockDeviceMemoryConst();
        auto y_ptr = y_buf.LockDeviceMemory(true);
        bbcu_fp32_MatrixColwiseSum
            (
                (float const *)x_ptr.GetAddr(),
                (float       *)y_ptr.GetAddr(),
                (int          )x_buf.GetNodeSize(),
                (int          )x_buf.GetFrameSize(),
                (int          )(x_buf.GetFrameStride() / sizeof(float))
            );
    }

    {
        auto y_ptr = y_buf.LockConst<float>();

        EXPECT_FLOAT_EQ(1+2+3, y_ptr(0));
        EXPECT_FLOAT_EQ(4+5+6, y_ptr(1));
    }
}


#endif

