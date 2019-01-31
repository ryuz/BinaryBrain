// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <omp.h>

#include "bb/Tensor.h"



// メイン関数
int main()
{
	std::cout << "Hello" << std::endl;
	
    const int N = 16;
    std::mt19937_64 mt(1);

    float d0[N];
    float d1[N];
    for (int i = 0; i < N; ++i) {
        d0[i] = (float)(mt() % 10000);
        d1[i] = (float)(mt() % 10000);
    }
    
    bb::Tensor_<float> t0(16);
    bb::Tensor_<float> t1(16);
    bb::Tensor_<float> t2(16);

    // 加算1
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t0 += t1;
    
    cudaDeviceSynchronize();
    t0.Lock();
    for (int i = 0; i < N; ++i) {
//        EXPECT_EQ(t0[i], d0[i] + d1[i]);
    }
    t0.Unlock();


    // 加算2
    t0.Lock(); t1.Lock();
    for (int i = 0; i < N; ++i) { t0[i] = d0[i]; t1[i] = d1[i]; }
    t0.Unlock(); t1.Unlock();

    t2 = t0 + t1;
    
    t2.Lock();
    for (int i = 0; i < N; ++i) {
//        EXPECT_EQ(t2[i], d0[i] + d1[i]);
    }
    t2.Unlock();

	
	return 0;
}






