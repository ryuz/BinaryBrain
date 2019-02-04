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
                data[i][j][k] = (float)((i+1)*10000 + (j+1) * 100 + (k+1)); 
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


    // clone
    auto t1 = t.Clone();

    t.Lock();
    t[0] = 11;
    t[1] = 21;
    t[2] = 31;
    t[3] = 41;
    t[4] = 51;
    t[5] = 61;
    t.Unlock();

    t1.Lock();
    EXPECT_EQ(t1({0, 0}), 1);
    EXPECT_EQ(t1({1, 0}), 2);
    EXPECT_EQ(t1({2, 0}), 3);
    EXPECT_EQ(t1({0, 1}), 4);
    EXPECT_EQ(t1({1, 1}), 5);
    EXPECT_EQ(t1({2, 1}), 6);
    t1.Unlock();
}




TEST(TensorTest, testTensorOp)
{
    const int N = 16;
    std::mt19937_64 mt(1);

    float d0[N];
    float d1[N];
    for (int i = 0; i < N; ++i) {
        d0[i] = (float)(mt() % 10000);
        d1[i] = (float)(mt() % 10000);
//       d0[i] = (float)(i);
//       d1[i] = (float)(i * 100);
    }
    
    bb::Tensor_<float> t0(16);
    bb::Tensor_<float> t1(16);
    bb::Tensor_<float> t2(16);

    // ‰ÁŽZ1-1
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t0 += t1;
    
    t0.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t0[i], d0[i] + d1[i]);
    }
    t0.Unlock();


    // ‰ÁŽZ1-2
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t0 += d1[0];
    
    t0.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t0[i], d0[i] + d1[0]);
    }
    t0.Unlock();



    // ‰ÁŽZ2-1
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = t0 + t1;
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[i] + d1[i]);
    }
    t2.Unlock();


    // ‰ÁŽZ2-2
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = t0 + d1[0];
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[i] + d1[0]);
    }
    t2.Unlock();
    

    // ‰ÁŽZ2-3
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = d0[0] + t1;
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[0] + d1[i]);
    }
    t2.Unlock();


    //////////////

    // Œ¸ŽZ1-1
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t0 -= t1;
    
    t0.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t0[i], d0[i] - d1[i]);
    }
    t0.Unlock();


    // Œ¸ŽZ1-2
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t0 -= d1[0];
    
    t0.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t0[i], d0[i] - d1[0]);
    }
    t0.Unlock();



    // Œ¸ŽZ2-1
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = t0 - t1;
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[i] - d1[i]);
    }
    t2.Unlock();


    // Œ¸ŽZ2-2
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = t0 - d1[0];
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[i] - d1[0]);
    }
    t2.Unlock();
    

    // Œ¸ŽZ2-3
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = d0[0] - t1;
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[0] - d1[i]);
    }
    t2.Unlock();




    t0 += 1.0f;
    t0 -= t1;
    t0 -= 1.0f;
    t0 *= t1;
    t0 *= 1.0f;
    t0 /= t1;
    t0 /= 1.0f;

//    t0 = t1 + 1.0f;
}



TEST(TensorTest, testTensor_cast)
{
    bb::Tensor_<float> t_fp32({2, 3});
    t_fp32.Lock();
    t_fp32({0, 0}) = 1;
    t_fp32({1, 0}) = 2;
    t_fp32({0, 1}) = 3;
    t_fp32({1, 1}) = 4;
    t_fp32({0, 2}) = 5;
    t_fp32({1, 2}) = 6;
    t_fp32.Unlock();

    bb::Tensor t(t_fp32);

    auto t_int32 = static_cast< bb::Tensor_<int> >(t);

    t_int32.Lock();
    EXPECT_EQ(t_int32({0, 0}), 1);
    EXPECT_EQ(t_int32({1, 0}), 2);
    EXPECT_EQ(t_int32({0, 1}), 3);
    EXPECT_EQ(t_int32({1, 1}), 4);
    EXPECT_EQ(t_int32({0, 2}), 5);
    EXPECT_EQ(t_int32({1, 2}), 6);
    t_int32.Unlock();

}
