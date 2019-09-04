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

TEST(cudacudaMatrixRowwiseSetVectorTest, test_cudaMatrixRowwiseSetVector)
{
    int const node_size  = 513;
    int const frame_size = 1021;

    bb::Tensor      x_buf(BB_TYPE_FP32, node_size);
    bb::FrameBuffer y_buf(BB_TYPE_FP32, frame_size, node_size);

    {
        auto x_ptr = x_buf.Lock<float>();
        for (int node = 0; node < node_size; ++node) {
            x_ptr(node) = node + 1;
        }
    }

    {
        auto x_ptr = x_buf.LockDeviceMemoryConst();
        auto y_ptr = y_buf.LockDeviceMemory(true);
        bbcu_fp32_MatrixRowwiseSetVector
            (
                (float const *)x_ptr.GetAddr(),
                (float       *)y_ptr.GetAddr(),
                (int          )y_buf.GetNodeSize(),
                (int          )y_buf.GetFrameSize(),
                (int          )(y_buf.GetFrameStride() / sizeof(float))
            );
    }

    {
        for (int node = 0; node < node_size; ++node) {
            for (int frame = 0; frame < frame_size; ++frame) {
                EXPECT_FLOAT_EQ((float)(node+1), y_buf.GetFP32(frame, node));
            }
        }
    }
}


#endif

