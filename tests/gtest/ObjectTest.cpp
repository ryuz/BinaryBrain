#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"


#include "bb/Object.h"


TEST(ObjectTest, testObject_test0)
{
}


#include "bb/ObjectReconstructor.h"


TEST(ObjectTest, Resonstruct_BatchNormalization)
{
    // test
    auto bn = bb::BatchNormalization<>::Create();
    bn->SetInputShape({32});

    auto obj_name = bn->GetObjectName();
    std::cout << obj_name << std::endl;

    {
        std::ofstream ofs("test_obj.bin", std::ios::binary);
        bn->DumpObject(ofs);
    }

    {
        std::ifstream ifs("test_obj.bin", std::ios::binary);
        auto test_obj = bb::Object_Reconstrutor(ifs);
        std::cout << test_obj->GetObjectName() << std::endl;
    }
}


