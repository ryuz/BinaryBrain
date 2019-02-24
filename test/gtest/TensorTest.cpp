#include <stdio.h>
#include <random>
#include <iostream>
#include <fstream>
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

    {
        auto ptr = t.GetPtr();
        for ( int i = 0; i < L; ++i ) {
            for ( int j = 0; j < M; ++j ) {
                for ( int k = 0; k < N; ++k ) {
                    data[i][j][k] = (float)((i+1)*10000 + (j+1) * 100 + (k+1)); 
                    ptr(i, j, k) = data[i][j][k];
                }
            }
        }
    }

    {
        auto ptr = t.GetConstPtr();
        for ( int i = 0; i < L; ++i ) {
            for ( int j = 0; j < M; ++j ) {
                for ( int k = 0; k < N; ++k ) {
                    EXPECT_EQ(ptr({k, j, i}), data[i][j][k]);
                    EXPECT_EQ(ptr(i, j, k), data[i][j][k]);
                    EXPECT_EQ(ptr[(i*M+j)*N+k], data[i][j][k]);
                }
            }
        }
    }

    t.Transpose({1, 2, 0});

    {
        auto ptr = t.GetConstPtr();
        for ( int i = 0; i < L; ++i ) {
            for ( int j = 0; j < M; ++j ) {
                for ( int k = 0; k < N; ++k ) {
                    EXPECT_EQ(ptr({j, i, k}), data[i][j][k]);
                    EXPECT_EQ(ptr(k, i, j), data[i][j][k]);
                    EXPECT_EQ(ptr[(i*M+j)*N+k], data[i][j][k]);
                }
            }
        }
    }


    t.Reshape({2, -1});

    t = t.sqrt();
}



TEST(TensorTest, testTensor_SaveLoad)
{
    const int L = 4;
    const int M = 3;
    const int N = 2;

    float   data[L][M][N];

    bb::Tensor_<float> t({N, M, L});

    {
        auto ptr = t.GetPtr();
        for ( int i = 0; i < L; ++i ) {
            for ( int j = 0; j < M; ++j ) {
                for ( int k = 0; k < N; ++k ) {
                    data[i][j][k] = (float)((i+1)*10000 + (j+1) * 100 + (k+1)); 
                    ptr(i, j, k) = data[i][j][k];
                }
            }
        }
    }

    {
        auto ptr = t.GetConstPtr();
        for ( int i = 0; i < L; ++i ) {
            for ( int j = 0; j < M; ++j ) {
                for ( int k = 0; k < N; ++k ) {
                    EXPECT_EQ(ptr({k, j, i}), data[i][j][k]);
                    EXPECT_EQ(ptr(i, j, k), data[i][j][k]);
                    EXPECT_EQ(ptr[(i*M+j)*N+k], data[i][j][k]);
                }
            }
        }
    }

    t.Transpose({1, 2, 0});

    {
        auto ptr = t.GetConstPtr();
        for ( int i = 0; i < L; ++i ) {
            for ( int j = 0; j < M; ++j ) {
                for ( int k = 0; k < N; ++k ) {
                    EXPECT_EQ(ptr({j, i, k}), data[i][j][k]);
                    EXPECT_EQ(ptr(k, i, j), data[i][j][k]);
                    EXPECT_EQ(ptr[(i*M+j)*N+k], data[i][j][k]);
                }
            }
        }
    }

    {
        bb::Tensor t1(t);
        std::ofstream ofs("TensorTest.bin", std::ios::binary);
        t1.Save(ofs);
    }

    {
        bb::Tensor t2;
        std::ifstream ifs("TensorTest.bin", std::ios::binary);
        t2.Load(ifs);

        auto ptr = t2.GetConstPtr<float>();
        for ( int i = 0; i < L; ++i ) {
            for ( int j = 0; j < M; ++j ) {
                for ( int k = 0; k < N; ++k ) {
                    EXPECT_EQ(ptr({j, i, k}), data[i][j][k]);
                    EXPECT_EQ(ptr(k, i, j), data[i][j][k]);
                    EXPECT_EQ(ptr[(i*M+j)*N+k], data[i][j][k]);
                }
            }
        }
    }

}




TEST(TensorTest, testTensor_Reshape)
{
    const int L = 3;
    const int M = 1;
    const int N = 2;
    bb::Tensor_<float> t({N, M, L});


    {
        auto ptr = t.GetPtr();
        float index = 1; 
        for ( int i = 0; i < L; ++i ) {
            for ( int j = 0; j < M; ++j ) {
                for ( int k = 0; k < N; ++k ) {
                    ptr({k, j, i}) = index++; 
                }
            }
        }
    }

    {
        auto ptr = t.GetConstPtr();
        EXPECT_EQ(ptr[0], 1);
        EXPECT_EQ(ptr[1], 2);
        EXPECT_EQ(ptr[2], 3);
        EXPECT_EQ(ptr[3], 4);
        EXPECT_EQ(ptr[4], 5);
        EXPECT_EQ(ptr[5], 6);
    }

    {
        auto ptr = t.GetConstPtr();
        EXPECT_EQ(ptr({0, 0, 0}), 1);
        EXPECT_EQ(ptr({1, 0, 0}), 2);
        EXPECT_EQ(ptr({0, 0, 1}), 3);
        EXPECT_EQ(ptr({1, 0, 1}), 4);
        EXPECT_EQ(ptr({0, 0, 2}), 5);
        EXPECT_EQ(ptr({1, 0, 2}), 6);
    }

    t.Reshape({3, -1});

   {
        auto ptr = t.GetConstPtr();
        EXPECT_EQ(ptr({0, 0}), 1);
        EXPECT_EQ(ptr({1, 0}), 2);
        EXPECT_EQ(ptr({2, 0}), 3);
        EXPECT_EQ(ptr({0, 1}), 4);
        EXPECT_EQ(ptr({1, 1}), 5);
        EXPECT_EQ(ptr({2, 1}), 6);
    }


    // clone
    auto t1 = t.Clone();

    {
        auto ptr = t.GetPtr();
        ptr[0] = 11;
        ptr[1] = 21;
        ptr[2] = 31;
        ptr[3] = 41;
        ptr[4] = 51;
        ptr[5] = 61;
    }

    {
        auto ptr = t1.GetPtr();
        EXPECT_EQ(ptr({0, 0}), 1);
        EXPECT_EQ(ptr({1, 0}), 2);
        EXPECT_EQ(ptr({2, 0}), 3);
        EXPECT_EQ(ptr({0, 1}), 4);
        EXPECT_EQ(ptr({1, 1}), 5);
        EXPECT_EQ(ptr({2, 1}), 6);
    }
}


TEST(TensorTest, testTensorCpuGpu)
{
    bb::Tensor  t0(true);
    bb::Tensor  t1(true);
    bb::Tensor  t2(false);
    bb::Tensor  t3(false);
    bb::Tensor  t4(true);
    bb::Tensor  t5(false);
    bb::Tensor  t6(true);

    const int size = 256;
    t0.Resize(BB_TYPE_FP32, size);
    t1.Resize(BB_TYPE_FP32, size);
    t2.Resize(BB_TYPE_FP32, size);
    t3.Resize(BB_TYPE_FP32, size);
    t4.Resize(BB_TYPE_FP32, size);
    t5.Resize(BB_TYPE_FP32, size);
    t6.Resize(BB_TYPE_FP32, size);

    std::mt19937_64 mt(1);
    float data0[size];
    float data1[size];
    float data2[size];
    float data3[size];
    float data4[size];
    float data5[size];
    float data6[size];
    for (int i = 0; i < size; ++i) {
        data0[i] = (float)(mt() % 100000);
    }

    {
        auto ptr0 = t0.GetPtr<float>();
        for (int i = 0; i < size; ++i) {
            ptr0[i] = data0[i];
        }
    }

    t1 = t0 + 1.0f;    for (int i = 0; i < size; ++i) { data1[i] = data0[i] + 1.0f; }
    t2 = t1 * 0.5f;    for (int i = 0; i < size; ++i) { data2[i] = data1[i] * 0.5f; }
    t3 = t2 - 5.0f;    for (int i = 0; i < size; ++i) { data3[i] = data2[i] - 5.0f; }
    t4 = t3 + 1.5f;    for (int i = 0; i < size; ++i) { data4[i] = data3[i] + 1.5f; }
    t5 = t4 * 2.5f;    for (int i = 0; i < size; ++i) { data5[i] = data4[i] * 2.5f; }
    t6 = t5 - 1.2f;    for (int i = 0; i < size; ++i) { data6[i] = data5[i] - 1.2f; }

    {
        auto ptr0 = t0.GetConstPtr<float>();
        auto ptr1 = t1.GetConstPtr<float>();
        auto ptr2 = t2.GetConstPtr<float>();
        auto ptr3 = t3.GetConstPtr<float>();
        auto ptr4 = t4.GetConstPtr<float>();
        auto ptr5 = t5.GetConstPtr<float>();
        auto ptr6 = t6.GetConstPtr<float>();
        for (int i = 0; i < size; ++i) {
            EXPECT_FLOAT_EQ(data0[i], ptr0[i]);
            EXPECT_FLOAT_EQ(data1[i], ptr1[i]);
            EXPECT_FLOAT_EQ(data2[i], ptr2[i]);
            EXPECT_FLOAT_EQ(data3[i], ptr3[i]);
            EXPECT_FLOAT_EQ(data4[i], ptr4[i]);
            EXPECT_FLOAT_EQ(data5[i], ptr5[i]);
            EXPECT_FLOAT_EQ(data6[i], ptr6[i]);
        }
    }

    t5 += t6 + 1.0f;    for (int i = 0; i < size; ++i) { data5[i] += data6[i] + 1.0f; }
    t4 *= t5 * 0.5f;    for (int i = 0; i < size; ++i) { data4[i] *= data5[i] * 0.5f; }
    t3 -= t4 - 5.0f;    for (int i = 0; i < size; ++i) { data3[i] -= data4[i] - 5.0f; }
    t2 += t3 + 1.5f;    for (int i = 0; i < size; ++i) { data2[i] += data3[i] + 1.5f; }
    t1 *= t2 * 2.5f;    for (int i = 0; i < size; ++i) { data1[i] *= data2[i] * 2.5f; }
    t0 -= t1 - 1.2f;    for (int i = 0; i < size; ++i) { data0[i] -= data1[i] - 1.2f; }

    t1 += t0 + 1.1f;    for (int i = 0; i < size; ++i) { data1[i] += data0[i] + 1.1f; }
    t2 *= t1 * 0.2f;    for (int i = 0; i < size; ++i) { data2[i] *= data1[i] * 0.2f; }
    t3 -= t2 - 5.3f;    for (int i = 0; i < size; ++i) { data3[i] -= data2[i] - 5.3f; }
    t4 += t3 + 1.4f;    for (int i = 0; i < size; ++i) { data4[i] += data3[i] + 1.4f; }
    t5 *= t4 * 2.5f;    for (int i = 0; i < size; ++i) { data5[i] *= data4[i] * 2.5f; }
    t6 -= t5 - 1.6f;    for (int i = 0; i < size; ++i) { data6[i] -= data5[i] - 1.6f; }

   {
        auto ptr0 = t0.GetConstPtr<float>();
        auto ptr1 = t1.GetConstPtr<float>();
        auto ptr2 = t2.GetConstPtr<float>();
        auto ptr3 = t3.GetConstPtr<float>();
        auto ptr4 = t4.GetConstPtr<float>();
        auto ptr5 = t5.GetConstPtr<float>();
        auto ptr6 = t6.GetConstPtr<float>();
        for (int i = 0; i < size; ++i) {
            EXPECT_FLOAT_EQ(data0[i], ptr0[i]);
            EXPECT_FLOAT_EQ(data1[i], ptr1[i]);
            EXPECT_FLOAT_EQ(data2[i], ptr2[i]);
            EXPECT_FLOAT_EQ(data3[i], ptr3[i]);
            EXPECT_FLOAT_EQ(data4[i], ptr4[i]);
            EXPECT_FLOAT_EQ(data5[i], ptr5[i]);
            EXPECT_FLOAT_EQ(data6[i], ptr6[i]);
        }
    }
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
    
    bb::Tensor_<float> t0((bb::index_t)16);
    bb::Tensor_<float> t1((bb::index_t)16);
    bb::Tensor_<float> t2((bb::index_t)16);

#if 0
    // â¡éZ1-1
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t0 += t1;
    
    t0.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t0[i], d0[i] + d1[i]);
    }
    t0.Unlock();


    // â¡éZ1-2
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t0 += d1[0];
    
    t0.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t0[i], d0[i] + d1[0]);
    }
    t0.Unlock();



    // â¡éZ2-1
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = t0 + t1;
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[i] + d1[i]);
    }
    t2.Unlock();


    // â¡éZ2-2
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = t0 + d1[0];
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[i] + d1[0]);
    }
    t2.Unlock();
    

    // â¡éZ2-3
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

    // å∏éZ1-1
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t0 -= t1;
    
    t0.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t0[i], d0[i] - d1[i]);
    }
    t0.Unlock();


    // å∏éZ1-2
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t0 -= d1[0];
    
    t0.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t0[i], d0[i] - d1[0]);
    }
    t0.Unlock();



    // å∏éZ2-1
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = t0 - t1;
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[i] - d1[i]);
    }
    t2.Unlock();


    // å∏éZ2-2
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = t0 - d1[0];
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[i] - d1[0]);
    }
    t2.Unlock();
    

    // å∏éZ2-3
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = d0[0] - t1;
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(t2[i], d0[0] - d1[i]);
    }
    t2.Unlock();
#endif

    // â¡éZ1-1
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t0 += t1;
    
    {
        auto ptr0 = t0.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr0[i], d0[i] + d1[i]);
        }
    }


    // â¡éZ1-2
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t0 += d1[0];
    
    {
        auto ptr0 = t0.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr0[i], d0[i] + d1[0]);
        }
    }


    // â¡éZ2-1
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t2 = t0 + t1;
    
    {
        auto ptr2 = t2.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr2[i], d0[i] + d1[i]);
        }
    }


    // â¡éZ2-2
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t2 = t0 + d1[0];
    
    {
        auto ptr2 = t2.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr2[i], d0[i] + d1[0]);
        }
    }


    // â¡éZ2-3
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t2 = d0[0] + t1;
    
    {
        auto ptr2 = t2.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr2[i], d0[0] + d1[i]);
        }
    }


    //////////////

    // å∏éZ1-1
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t0 -= t1;
    
    {
        auto ptr0 = t0.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr0[i], d0[i] - d1[i]);
        }
    }


    // å∏éZ1-2
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t0 -= d1[0];
    
    {
        auto ptr0 = t0.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr0[i], d0[i] - d1[0]);
        }
    }



    // å∏éZ2-1
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t2 = t0 - t1;
    
    {
        auto ptr2 = t2.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr2[i], d0[i] - d1[i]);
        }
    }


    // å∏éZ2-2
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t2 = t0 - d1[0];
    
    {
        auto ptr2 = t2.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr2[i], d0[i] - d1[0]);
        }
    }
    

    // å∏éZ2-3
    {
        auto ptr0 = t0.GetPtr();
        auto ptr1 = t1.GetPtr();
        for (int i = 0; i < N; ++i) { ptr0[i] = d0[i]; ptr1[i] = d1[i]; }
    }

    t2 = d0[0] - t1;
    
    {
        auto ptr2 = t2.GetConstPtr();
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(ptr2[i], d0[0] - d1[i]);
        }
    }


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
    bb::Tensor_<float> t_fp32(std::vector<bb::index_t>{2, 3});

#if 0
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
#endif

    {
        auto ptr = t_fp32.GetPtr();
        ptr({0, 0}) = 1;
        ptr({1, 0}) = 2;
        ptr({0, 1}) = 3;
        ptr({1, 1}) = 4;
        ptr({0, 2}) = 5;
        ptr({1, 2}) = 6;
    }

    bb::Tensor t(t_fp32);

    auto t_int32 = static_cast< bb::Tensor_<int> >(t);

    {
        auto ptr = t_fp32.GetConstPtr();
        EXPECT_EQ(ptr({0, 0}), 1);
        EXPECT_EQ(ptr({1, 0}), 2);
        EXPECT_EQ(ptr({0, 1}), 3);
        EXPECT_EQ(ptr({1, 1}), 4);
        EXPECT_EQ(ptr({0, 2}), 5);
        EXPECT_EQ(ptr({1, 2}), 6);
    }

    
    {
        auto ptr = t.GetConstPtr<float>();
        EXPECT_EQ(ptr({0, 0}), 1);
        EXPECT_EQ(ptr({1, 0}), 2);
        EXPECT_EQ(ptr({0, 1}), 3);
        EXPECT_EQ(ptr({1, 1}), 4);
        EXPECT_EQ(ptr({0, 2}), 5);
        EXPECT_EQ(ptr({1, 2}), 6);
    }
}




template <typename T>
void test_Operator(bb::indices_t shape)
{
    auto node_size = bb::GetShapeSize(shape);

    bb::Tensor_<T>  base_dst(shape);
    bb::Tensor_<T>  base_src0(shape);
    bb::Tensor_<T>  base_src1(shape);
    base_dst.InitNormalDistribution(0, 1000, 1);
    base_src0.InitNormalDistribution(0, 1000, 2);
    base_src1.InitNormalDistribution(0, 1000, 3);
    T scalar = (T)123.4;

    {
        // É[ÉçèúéZâÒî
        auto b_s1 = base_src1.GetPtr();
        for (bb::index_t i = 0; i < node_size; ++i) {
            if ( b_s1[i] == 0 ) {
                b_s1[i]++;
            }
        }
    }
 

    // +
    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst += src0;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]+b_s0[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst += scalar;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]+scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 + src1;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]+b_s1[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 + scalar;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]+scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = scalar + src1;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(scalar+b_s1[i]));
        }
    }


    // -
    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst -= src0;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]-b_s0[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst -= scalar;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]-scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 - src1;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]-b_s1[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 - scalar;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]-scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = scalar - src1;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(scalar-b_s1[i]));
        }
    }


    // *
    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst *= src0;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]*b_s0[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst *= scalar;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]*scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 * src1;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]*b_s1[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 * scalar;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]*scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = scalar * src1;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(scalar*b_s1[i]));
        }
    }


    // /
    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst /= src0;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]/b_s0[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst /= scalar;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]/scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 / src1;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]/b_s1[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 / scalar;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]/scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = scalar / src1;
        
        auto b_d  = base_dst.GetConstPtr();
        auto b_s0 = base_src0.GetConstPtr();
        auto b_s1 = base_src1.GetConstPtr();
        auto d  = dst.GetConstPtr();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(scalar/b_s1[i]));
        }
    }
}


TEST(TensorTest, testTensor_Operator)
{
    test_Operator<float>({2, 3, 4});
    test_Operator<double>({7, 2, 5});
    test_Operator<std::int8_t>({7, 2, 5});
    test_Operator<std::int16_t>({7, 2, 5});
    test_Operator<std::int32_t>({7, 2, 5});
    test_Operator<std::int64_t>({7, 2, 5});
    test_Operator<std::uint8_t>({7, 2, 5});
    test_Operator<std::uint16_t>({7, 2, 5});
    test_Operator<std::uint32_t>({7, 2, 5});
    test_Operator<std::uint64_t>({7, 2, 5});
}



template <typename T>
void test_OperatorX(bb::indices_t shape)
{
    auto node_size = bb::GetShapeSize(shape);

    bb::Tensor  base_dst(bb::DataType<T>::type, shape);
    bb::Tensor  base_src0(bb::DataType<T>::type, shape);
    bb::Tensor  base_src1(bb::DataType<T>::type, shape);
    base_dst.InitNormalDistribution(0, 1000, 1);
    base_src0.InitNormalDistribution(0, 1000, 2);
    base_src1.InitNormalDistribution(0, 1000, 3);
    T scalar = (T)123.4;
    
    {
        // É[ÉçèúéZâÒî
        auto b_s1 = base_src1.GetPtr<T>();
        for (bb::index_t i = 0; i < node_size; ++i) {
            if ( b_s1[i] == 0 ) {
                b_s1[i]++;
            }
        }
    }
        

    // +
    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst += src0;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]+b_s0[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst += scalar;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]+scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 + src1;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]+b_s1[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 + scalar;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]+scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = scalar + src1;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(scalar+b_s1[i]));
        }
    }


    // -
    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst -= src0;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]-b_s0[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst -= scalar;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]-scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 - src1;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]-b_s1[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 - scalar;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]-scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = scalar - src1;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(scalar-b_s1[i]));
        }
    }


    // *
    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst *= src0;
        
         auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]*b_s0[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst *= scalar;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]*scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 * src1;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]*b_s1[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 * scalar;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]*scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = scalar * src1;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(scalar*b_s1[i]));
        }
    }


    // /
    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst /= src1;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
        
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]/b_s1[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst /= scalar;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_d[i]/scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 / src1;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]/b_s1[i]));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = src0 / scalar;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(b_s0[i]/scalar));
        }
    }

    {
        auto dst  = base_dst.Clone();
        auto src0 = base_src0.Clone();
        auto src1 = base_src1.Clone();

        dst = scalar / src1;
        
        auto b_d  = base_dst.GetConstPtr<T>();
        auto b_s0 = base_src0.GetConstPtr<T>();
        auto b_s1 = base_src1.GetConstPtr<T>();
        auto d  = dst.GetConstPtr<T>();
       
        for (bb::index_t i = 0; i < node_size; ++i) {
            EXPECT_EQ(d[i], (T)(scalar/b_s1[i]));
        }
    }
}



TEST(TensorTest, testTensorX_Operator)
{
    test_OperatorX<float        >({1, 2, 3});
    test_OperatorX<double       >({2, 2, 3});
    test_OperatorX<std::int8_t  >({3, 2, 3});
    test_OperatorX<std::int16_t >({2, 2, 3});
    test_OperatorX<std::int32_t >({4, 2, 3});
//    test_OperatorX<std::int64_t >({1, 2, 3});
    test_OperatorX<std::uint8_t >({1, 2, 3});
    test_OperatorX<std::uint16_t>({5, 2});
    test_OperatorX<std::uint32_t>({1, 2, 3, 7});
//    test_OperatorX<std::uint64_t>({1, 2, 3});
}
