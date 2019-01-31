#include <stdio.h>
#include <random>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Memory.h"
#include "bb/Tensor.h"


TEST(TensorTest, testTensor_Transpose)
{
    const int L = 4;
    const int M = 3;
    const int N = 2;

    float   data[L][M][N];

    bb::Tensor_<float> t({N, M, L});

    t.Lock();
    for ( int i = 0; i < L; ++i ) {
        for ( int j = 0; j < M; ++j ) {
            for ( int k = 0; k < N; ++k ) {
                data[i][j][k] = (i+1)*10000 + (j+1) * 100 + (k+1); 
                t({k, j, i}) = data[i][j][k];
//              std::cout << "[" << i << "][" << j<< "][" << k << "] : " << t({k, j, i}) << std::endl;
            }
        }
    }
    t.Unlock();

    t.Lock();
    for ( int i = 0; i < L; ++i ) {
        for ( int j = 0; j < M; ++j ) {
            for ( int k = 0; k < N; ++k ) {
                EXPECT_EQ(t({k, j, i}), data[i][j][k]);
//              std::cout << "[" << i << "][" << j<< "][" << k << "] : " << t({k, j, i}) << std::endl;
            }
        }
    }
    t.Unlock();

    t.Transpose({1, 2, 0});

    t.Lock();
    for ( int i = 0; i < L; ++i ) {
        for ( int j = 0; j < M; ++j ) {
            for ( int k = 0; k < N; ++k ) {
                EXPECT_EQ(t({j, i, k}), data[i][j][k]);
//              std::cout << "[" << i << "][" << j<< "][" << k << "] : " << t({k, j, i}) << std::endl;
            }
        }
    }
    t.Unlock();

    t.Reshape({2, -1});
}


TEST(TensorTest, testTensor_Reshape)
{
    const int L = 3;
    const int M = 1;
    const int N = 2;
    bb::Tensor_<float> t({N, M, L});

    t.Lock();
    float index = 1; 
    for ( int i = 0; i < L; ++i ) {
        for ( int j = 0; j < M; ++j ) {
            for ( int k = 0; k < N; ++k ) {
                t({k, j, i}) = index++; 
            }
        }
    }
    t.Unlock();

    t.Lock();
    EXPECT_EQ(t[0], 1);
    EXPECT_EQ(t[1], 2);
    EXPECT_EQ(t[2], 3);
    EXPECT_EQ(t[3], 4);
    EXPECT_EQ(t[4], 5);
    EXPECT_EQ(t[5], 6);
    t.Unlock();

    t.Lock();
    EXPECT_EQ(t({0, 0, 0}), 1);
    EXPECT_EQ(t({1, 0, 0}), 2);
    EXPECT_EQ(t({0, 0, 1}), 3);
    EXPECT_EQ(t({1, 0, 1}), 4);
    EXPECT_EQ(t({0, 0, 2}), 5);
    EXPECT_EQ(t({1, 0, 2}), 6);
    t.Unlock();

    t.Reshape({3, -1});

    t.Lock();
    EXPECT_EQ(t({0, 0}), 1);
    EXPECT_EQ(t({1, 0}), 2);
    EXPECT_EQ(t({2, 0}), 3);
    EXPECT_EQ(t({0, 1}), 4);
    EXPECT_EQ(t({1, 1}), 5);
    EXPECT_EQ(t({2, 1}), 6);
    t.Unlock();
}




TEST(TensorTest, testTensorOp)
{
    bb::Tensor_<float> t0(16);
    bb::Tensor_<float> t1(16);
    bb::Tensor_<float> t2(16);

    t0.Lock();
    t0[0] = 1;
    t0.Unlock();


    bb::Tensor tt(16, BB_TYPE_FP32);

    t0 += t1;
    t0 += 1.0f;
    t0 -= t1;
    t0 -= 1.0f;
    t0 *= t1;
    t0 *= 1.0f;
    t0 /= t1;
    t0 /= 1.0f;

//    t0 = t1 + 1.0f;
}
