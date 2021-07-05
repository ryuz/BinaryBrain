#include <stdio.h>
#include <iostream>
#include <random>
#include "gtest/gtest.h"
#include "bb/LossMeanSquaredError.h"


template <typename T, int frame_size, int node_size, bool host_only=false>
void testLossMeanSquaredError()
{
    std::mt19937_64             mt(1);
    std::normal_distribution<T> dist(0.0, 1.0);

    auto lossFunc = bb::LossMeanSquaredError<float>::Create();

    for ( int i = 0; i < 3; ++i ) {
        lossFunc->Clear();

        static T       y[frame_size][node_size];
        static T       t[frame_size][node_size];
        static T       dy[frame_size][node_size];
        double  loss = 0;
        for (int f = 0; f < frame_size; ++f) {
            for (int n = 0; n < node_size; ++n) {
                y[f][n] = dist(mt);
                t[f][n] = dist(mt);
                auto grad = y[f][n] - t[f][n]; 
                dy[f][n] = grad / (T)frame_size;
                loss += grad*grad;
            }
        }
        loss /= (node_size * frame_size);


        bb::FrameBuffer y_buf(frame_size, {node_size}, bb::DataType<T>::type, host_only);
        bb::FrameBuffer t_buf(frame_size, {node_size}, bb::DataType<T>::type, host_only);
        for (int f = 0; f < frame_size; ++f) {
            for (int n = 0; n < node_size; ++n) {
                y_buf.SetFP32(f, n, y[f][n]);
                t_buf.SetFP32(f, n, t[f][n]);
            }
        }

        auto   dy_buf      = lossFunc->CalculateLoss(y_buf, t_buf, y_buf.GetFrameSize());
        auto   result_loss = lossFunc->GetLoss();
        for (int f = 0; f < frame_size; ++f) {
            for (int n = 0; n < node_size; ++n) {
                auto dy_value = dy_buf.GetFP32(f, n);
                ASSERT_NEAR(dy_value, dy[f][n], 1.0e-7);
            }
        }

        ASSERT_NEAR(result_loss, loss, 1.0e-7);
    }
}



TEST(LossMeanSquaredErrorTest, test0_gpu) { testLossMeanSquaredError<float, 1, 1>(); }
TEST(LossMeanSquaredErrorTest, test0_cpu) { testLossMeanSquaredError<float, 1, 1, true>(); }

TEST(LossMeanSquaredErrorTest, test1_gpu) { testLossMeanSquaredError<float, 13, 17>(); }
TEST(LossMeanSquaredErrorTest, test1_cpu) { testLossMeanSquaredError<float, 13, 17, true>(); }

TEST(LossMeanSquaredErrorTest, test2_gpu) { testLossMeanSquaredError<float, 32, 1>(); }
TEST(LossMeanSquaredErrorTest, test2_cpu) { testLossMeanSquaredError<float, 32, 1, true>(); }

TEST(LossMeanSquaredErrorTest, test3_gpu) { testLossMeanSquaredError<float, 32, 32>(); }
TEST(LossMeanSquaredErrorTest, test3_cpu) { testLossMeanSquaredError<float, 32, 32, true>(); }

TEST(LossMeanSquaredErrorTest, test4_gpu) { testLossMeanSquaredError<float, 1123, 977>(); }
TEST(LossMeanSquaredErrorTest, test4_cpu) { testLossMeanSquaredError<float, 1123, 977, true>(); }
