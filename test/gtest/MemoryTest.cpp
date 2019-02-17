#include <stdio.h>
#include <random>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/Memory.h"
#include "bb/Tensor.h"


TEST(MemoryTest, testMem)
{
	{
		auto mem = bb::Memory::Create(1024, true);
		mem->GetPtr();
		mem->GetDevPtr();
		mem->IsDeviceAvailable();

        auto wp = mem->GetPtr();
        auto rp = mem->GetConstPtr();

//      bb::Memory::ConstPtr rr(wp.GetConst());
//      bb::Memory::ConstPtr rr = (bb::Memory::ConstPtr)wp;
        rp = wp;
	}

	{
		auto mem = bb::Memory::Create(1024);
		mem->GetPtr();
		mem->GetDevPtr();
		mem->IsDeviceAvailable();
	}

	{
		auto mem = bb::Memory::Create(1024, false);
		mem->GetPtr();
		mem->GetDevPtr();
		mem->IsDeviceAvailable();
	}
    

#if 0
    bb::Tensor a(16, BB_TYPE_FP32);
    bb::Tensor b(16, BB_TYPE_FP32);

    a.Lock();
    a.At<float>(0) = 1;
    a.At<float>(1) = 11;
    a.Unlock();

    b.Lock();
    b.At<float>(0) = 2;
    b.At<float>(1) = 33;
    b.Unlock();

    a += b;

    a.Lock();
    std::cout << "a[0] = " <<  a.At<float>(0) << std::endl;
    std::cout << "a[1] = " <<  a.At<float>(1) << std::endl;
    a.Unlock();
#endif
}
