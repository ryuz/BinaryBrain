#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <valarray>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gtest/gtest.h"

#include "bb/FrameBuffer.h"
#include "bbcu/bbcu.h"


#if BB_WITH_CUDA


static double calc_mean(std::valarray<double> const &varray)
{
    return varray.sum() / varray.size();
}

static double calc_var(std::valarray<double> const &varray)
{
    double mean = calc_mean(varray);
    return ((varray*varray).sum() - mean * mean*varray.size()) / varray.size();
}


TEST(cudaMatrixColwiseMeanVarTest, test_MatrixColwiseMeanVar)
{
    const int n = 1027;

    std::mt19937_64 mt(1);
    std::normal_distribution<double> dist0(1.0, 2.0);
    std::normal_distribution<double> dist1(-1.0, 3.0);
    std::normal_distribution<double> dist2(2.0, 4.0);

    std::valarray<double> arr0(n);
    std::valarray<double> arr1(n);
    std::valarray<double> arr2(n);
    for (int i = 0; i < n; ++i) {
        arr0[i] = dist0(mt);
        arr1[i] = dist1(mt);
        arr2[i] = dist2(mt);
    }

    bb::FrameBuffer x_buf(n, {3}, BB_TYPE_FP32);
    for (int i = 0; i < n; ++i) {
        x_buf.SetFP32(i, 0, (float)arr0[i]);
        x_buf.SetFP32(i, 1, (float)arr1[i]);
        x_buf.SetFP32(i, 2, (float)arr2[i]);
    }

    bb::Tensor m_buf({3}, BB_TYPE_FP32);
    bb::Tensor v_buf({3}, BB_TYPE_FP32);
    {
        auto x_ptr = x_buf.LockDeviceMemoryConst();
        auto m_ptr = m_buf.LockDeviceMemory();
        auto v_ptr = v_buf.LockDeviceMemory();
        bbcu_fp32_MatrixColwiseMeanVar
            (
                (const float *)x_ptr.GetAddr(),
                (float       *)m_ptr.GetAddr(),
                (float       *)v_ptr.GetAddr(),
                (int          )3,
                (int          )n,
                (int          )x_buf.GetFrameStride() / sizeof(float)
            );
    }

    {
        auto m_ptr = m_buf.LockConst<float>();
        auto v_ptr = v_buf.LockConst<float>();

        EXPECT_FLOAT_EQ((float)calc_mean(arr0), m_ptr[0]);
        EXPECT_FLOAT_EQ((float)calc_mean(arr1), m_ptr[1]);
        EXPECT_FLOAT_EQ((float)calc_mean(arr2), m_ptr[2]);
        EXPECT_FLOAT_EQ((float)calc_var(arr0), v_ptr[0]);
        EXPECT_FLOAT_EQ((float)calc_var(arr1), v_ptr[1]);
        EXPECT_FLOAT_EQ((float)calc_var(arr2), v_ptr[2]);
    }
}


#endif

