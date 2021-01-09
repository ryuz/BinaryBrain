#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"


#include "bb/Object.h"


TEST(ObjectTest, testObject_test0)
{
}


#include "bb/ObjectReconstructor.h"


TEST(ObjectTest, SerializeModel)
{
    auto src_net = bb::Sequential::Create();

    src_net->Add(bb::RealToBinary<float, float>::Create());
    src_net->Add(bb::RealToBinary<bb::Bit, float>::Create());
    src_net->Add(bb::BinaryToReal<float, float>::Create());
    src_net->Add(bb::BinaryToReal<bb::Bit, float>::Create());
    src_net->Add(bb::BitEncode<float, float>::Create());
    src_net->Add(bb::BitEncode<bb::Bit, float>::Create());
    src_net->Add(bb::DifferentiableLutN<6, float, float>::Create());
    src_net->Add(bb::DifferentiableLutN<6, bb::Bit, float>::Create());
    src_net->Add(bb::DifferentiableLutN<5, float, float>::Create());
    src_net->Add(bb::DifferentiableLutN<5, bb::Bit, float>::Create());
    src_net->Add(bb::DifferentiableLutN<4, float, float>::Create());
    src_net->Add(bb::DifferentiableLutN<4, bb::Bit, float>::Create());
    src_net->Add(bb::DifferentiableLutN<3, float, float>::Create());
    src_net->Add(bb::DifferentiableLutN<3, bb::Bit, float>::Create());
    src_net->Add(bb::DifferentiableLutN<2, float, float>::Create());
    src_net->Add(bb::DifferentiableLutN<2, bb::Bit, float>::Create());
    src_net->Add(bb::BatchNormalization<float>::Create());

    {
        std::ofstream ofs("test_obj.bin", std::ios::binary);
        src_net->DumpObject(ofs);
    }

    {
        std::ifstream ifs("test_obj.bin", std::ios::binary);
        auto dst_net = std::dynamic_pointer_cast<bb::Sequential>(bb::Object_Reconstruct(ifs));
        EXPECT_TRUE(dst_net);
        
        EXPECT_EQ(dst_net->Get(0)->GetModelName(), "RealToBinary");
        EXPECT_EQ(dst_net->Get(1)->GetModelName(), "RealToBinary");
        EXPECT_EQ(dst_net->Get(2)->GetModelName(), "BinaryToReal");
        EXPECT_EQ(dst_net->Get(3)->GetModelName(), "BinaryToReal");

        EXPECT_EQ(dst_net->Get(4)->GetModelName(), "BitEncode");
        EXPECT_EQ(dst_net->Get(5)->GetModelName(), "BitEncode");
        EXPECT_EQ(dst_net->Get(5)->GetObjectName(), "BitEncode_bit_fp32");

        EXPECT_EQ(dst_net->Get(6)->GetModelName(), "DifferentiableLut6");
        EXPECT_EQ(dst_net->Get(7)->GetModelName(), "DifferentiableLut6");
        EXPECT_EQ(dst_net->Get(8)->GetModelName(), "DifferentiableLut5");
        EXPECT_EQ(dst_net->Get(9)->GetModelName(), "DifferentiableLut5");
        EXPECT_EQ(dst_net->Get(10)->GetModelName(), "DifferentiableLut4");
        EXPECT_EQ(dst_net->Get(11)->GetModelName(), "DifferentiableLut4");
        EXPECT_EQ(dst_net->Get(12)->GetModelName(), "DifferentiableLut3");
        EXPECT_EQ(dst_net->Get(13)->GetModelName(), "DifferentiableLut3");
        EXPECT_EQ(dst_net->Get(14)->GetModelName(), "DifferentiableLut2");
        EXPECT_EQ(dst_net->Get(15)->GetModelName(), "DifferentiableLut2");

        EXPECT_EQ(dst_net->Get(16)->GetModelName(), "BatchNormalization");
    }
}








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

    {
        bb::Tensor  t;
        
        std::ifstream ifs("test_obj.bin", std::ios::binary);
        t.LoadObject(ifs);

        auto ptr = t.LockConst<float>();
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


TEST(ObjectTest, Resonstruct_DifferentiableLutN)
{
    // test
    auto model = bb::DifferentiableLutN<6, bb::Bit, float>::Create({1,2,3});
    model->SetInputShape({32, 8});

    auto obj_name = model->GetObjectName();
//  std::cout << obj_name << std::endl;

    {
        std::ofstream ofs("test_obj.bin", std::ios::binary);
        model->DumpObject(ofs);
    }

    {
        std::ifstream ifs("test_obj.bin", std::ios::binary);
        auto test_obj = std::dynamic_pointer_cast< bb::DifferentiableLutN<6, bb::Bit, float> >(bb::Object_Reconstruct(ifs));
        EXPECT_TRUE(test_obj);
//      std::cout << test_obj->GetObjectName() << std::endl;
        EXPECT_EQ(test_obj->GetObjectName(), model->GetObjectName());
        EXPECT_EQ(test_obj->GetInputShape(), bb::indices_t({32, 8}));
        EXPECT_EQ(test_obj->GetOutputShape(), bb::indices_t({1, 2, 3}));
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


TEST(ObjectTest, SerializeValue)
{
    bool                        src_bool0      = false;
    bool                        src_bool1      = true;
    std::string                 src_str        = "abcde";
    std::int8_t                 src_int8       = 7;
    int                         src_int        = 123;
    std::int32_t                src_int32      = 111;
    std::int64_t                src_int64      = 222;
    bb::index_t                 src_index      = 9999;;
    bb::indices_t               src_indices    = {1, 2, 3};
    std::vector<float>          src_vec_fp32   = {1.5, 2, 1.25};
    std::vector<std::int64_t>   src_vec_int64  = {7, 6, 5, 4};

    {
        std::ofstream ofs("test_obj.bin", std::ios::binary);
        bb::SaveValue(ofs, src_bool0);
        bb::SaveValue(ofs, src_bool1);
        bb::SaveValue(ofs, src_str);
        bb::SaveValue(ofs, src_int8);
        bb::SaveValue(ofs, src_int);
        bb::SaveValue(ofs, src_int32);
        bb::SaveValue(ofs, src_int64);
        bb::SaveValue(ofs, src_index);
//      bb::SaveValue(ofs, src_indices);
        bb::SaveIndices(ofs, src_indices);  // 同じかどうか確認
        bb::SaveValue(ofs, src_vec_fp32);
        bb::SaveValue(ofs, src_vec_int64);
    }

    {
        bool                        dst_bool0;
        bool                        dst_bool1;
        std::string                 dst_str;
        std::int8_t                 dst_int8;
        int                         dst_int;
        std::int32_t                dst_int32;
        std::int64_t                dst_int64;
        bb::index_t                 dst_index;
        bb::indices_t               dst_indices;
        std::vector<float>          dst_vec_fp32;
        std::vector<std::int64_t>   dst_vec_int64;

        std::ifstream ifs("test_obj.bin", std::ios::binary);
        bb::LoadValue(ifs, dst_bool0);
        bb::LoadValue(ifs, dst_bool1);
        bb::LoadValue(ifs, dst_str);
        bb::LoadValue(ifs, dst_int8);
        bb::LoadValue(ifs, dst_int);
        bb::LoadValue(ifs, dst_int32);
        bb::LoadValue(ifs, dst_int64);
        bb::LoadValue(ifs, dst_index);
        bb::LoadValue(ifs, dst_indices);
        bb::LoadValue(ifs, dst_vec_fp32);
        bb::LoadValue(ifs, dst_vec_int64);

        EXPECT_EQ(dst_bool0,     src_bool0    );
        EXPECT_EQ(dst_bool1,     src_bool1    );
        EXPECT_EQ(dst_str,       src_str      );
        EXPECT_EQ(dst_int8,      src_int8     );
        EXPECT_EQ(dst_int,       src_int      );
        EXPECT_EQ(dst_int32,     src_int32    );
        EXPECT_EQ(dst_int64,     src_int64    );
        EXPECT_EQ(dst_index,     src_index    );
        EXPECT_EQ(dst_indices,   src_indices  );
        EXPECT_EQ(dst_vec_fp32,  src_vec_fp32 );
        EXPECT_EQ(dst_vec_int64, src_vec_int64);
    }
}
