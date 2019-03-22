#include <stdio.h>
#include <random>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Variables.h"


TEST(VariablesTest, VariablesTest_Test)
{
    auto t0 = std::make_shared<bb::Tensor>(BB_TYPE_FP32,  bb::indices_t({2, 3, 4}));
    auto t1 = std::make_shared<bb::Tensor>(BB_TYPE_FP64,  bb::indices_t({3, 2, 6}));
    auto t2 = std::make_shared<bb::Tensor>(BB_TYPE_INT32, bb::indices_t({4, 1, 7}));

    bb::Variables   var1;
    var1.PushBack(t0);
    var1.PushBack(t1);
    var1.PushBack(t2);

    bb::Variables   var2(var1.GetTypes(), var1.GetShapes());
    bb::Variables   var3(var1.GetTypes(), var1.GetShapes());

    var1 = 1;
    var2 = 2;
    var3 = 0;

    var2 += var1;

    {
        auto ptr2_0 = var2[0].Lock<float>();
        auto ptr2_1 = var2[1].Lock<double>();
        auto ptr2_2 = var2[2].Lock<std::int32_t>();
        EXPECT_EQ(3.0f, ptr2_0[0]);
        EXPECT_EQ(3.0f, ptr2_0[1]);
        EXPECT_EQ(3.0f, ptr2_0[2]);
        EXPECT_EQ(3.0, ptr2_1[0]);
        EXPECT_EQ(3.0, ptr2_1[1]);
        EXPECT_EQ(3.0, ptr2_1[2]);
        EXPECT_EQ(3, ptr2_2[0]);
        EXPECT_EQ(3, ptr2_2[1]);
        EXPECT_EQ(3, ptr2_2[2]);
    }

    var2 += 11;

    var3 = var1 + var2;
    var3 = var1 + 1;
    var3 = 2 + var1 + 1;

    var3 -= var1;
    var3 -= 5;
    var3 = var1 - var2;
    var3 = var1 - 1;
    var3 = 2 - var1;

    var3 *= var1;
    var3 *= 5;
    var3 = var1 * var2;
    var3 = var1 * 1;
    var3 = 2 * var1;

    var3 /= var1;
    var3 /= 5;
    var3 = var1 / var2;
    var3 = var1 / 1;
    var3 = 2 / var1;
}

