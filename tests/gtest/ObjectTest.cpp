#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"


#include "bb/Object.h"


TEST(ObjectTest, testObject_test0)
{
}


#include "bb/ObjectReconstructor.h"


TEST(ObjectTest, Resonstruct_Tensor)
{
    // test
    bb::Tensor  src_t({2, 3}, BB_TYPE_FP32);
    
    {
        auto ptr = src_t.Lock<float>();
        ptr(0, 0) = 1;
        ptr(0, 1) = 2;
        ptr(0, 2) = 3;
        ptr(1, 0) = 4;
        ptr(1, 1) = 5;
        ptr(1, 2) = 6;
    }

    {
        std::ofstream ofs("test_obj.bin", std::ios::binary);
        src_t.DumpObject(ofs);
    }

    {
        std::ifstream ifs("test_obj.bin", std::ios::binary);
        auto test_obj = std::dynamic_pointer_cast<bb::Tensor>(bb::Object_Reconstruct(ifs));
        EXPECT_TRUE(test_obj);

        auto dst_t = *test_obj;
        auto ptr = dst_t.LockConst<float>();
        EXPECT_EQ(ptr(0, 0), 1);
        EXPECT_EQ(ptr(0, 1), 2);
        EXPECT_EQ(ptr(0, 2), 3);
        EXPECT_EQ(ptr(1, 0), 4);
        EXPECT_EQ(ptr(1, 1), 5);
        EXPECT_EQ(ptr(1, 2), 6);
    }
}

TEST(ObjectTest, Resonstruct_Tensor_int32)
{
    // test
    bb::Tensor_<std::int32_t>  src_t(bb::indices_t({2, 3}));
    
    {
        auto ptr = src_t.Lock();
        ptr(0, 0) = 1;
        ptr(0, 1) = 2;
        ptr(0, 2) = 3;
        ptr(1, 0) = 4;
        ptr(1, 1) = 5;
        ptr(1, 2) = 6;
    }

    {
        std::ofstream ofs("test_obj.bin", std::ios::binary);
        src_t.DumpObject(ofs);
    }

    {
        std::ifstream ifs("test_obj.bin", std::ios::binary);
        auto test_obj = std::dynamic_pointer_cast< bb::Tensor_<std::int32_t> >(bb::Object_Reconstruct(ifs));
        EXPECT_TRUE(test_obj);

        auto dst_t = *test_obj;
        auto ptr = dst_t.LockConst();
        EXPECT_EQ(ptr(0, 0), 1);
        EXPECT_EQ(ptr(0, 1), 2);
        EXPECT_EQ(ptr(0, 2), 3);
        EXPECT_EQ(ptr(1, 0), 4);
        EXPECT_EQ(ptr(1, 1), 5);
        EXPECT_EQ(ptr(1, 2), 6);
    }
}


TEST(ObjectTest, Resonstruct_BatchNormalization)
{
    // test
    auto bn = bb::BatchNormalization<>::Create();
    bn->SetInputShape({32});

    auto obj_name = bn->GetObjectName();
//  std::cout << obj_name << std::endl;

    {
        std::ofstream ofs("test_obj.bin", std::ios::binary);
        bn->DumpObject(ofs);
    }

    {
        std::ifstream ifs("test_obj.bin", std::ios::binary);
        auto test_obj = bb::Object_Reconstruct(ifs);
//      std::cout << test_obj->GetObjectName() << std::endl;
        EXPECT_EQ(test_obj->GetObjectName(), bn->GetObjectName());
    }
}


