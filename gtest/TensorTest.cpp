#include <stdio.h>
#include <random>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Memory.h"
#include "bb/Tensor.h"



TEST(TensorTest, testTensor)
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
