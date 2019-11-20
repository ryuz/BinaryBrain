#include <stdio.h>
#include <iostream>
#include <chrono>
#include <random>

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"

int Test_MicroMlp_Forward(void);
int Test_MicroMlp_Backward(void);

int Test_StochasticLut6_Forward(void);
int Test_StochasticLut6_Backward(void);

void bbcu_ShufleTest(void);

int main()
{
    bbcu_ShufleTest();
    getchar();
    return 0;

    void* ptr0 = bbcu_LocalHeap_Malloc(2*1024);
    void* ptr1 = bbcu_LocalHeap_Malloc(1*1024);
    void* ptr2 = bbcu_LocalHeap_Malloc(3*1024);

    bbcu_LocalHeap_Free(ptr0);
    bbcu_LocalHeap_Free(ptr2);

    void* ptr00 = bbcu_LocalHeap_Malloc(2*1024);
    void* ptr02 = bbcu_LocalHeap_Malloc(3*1024);

    bbcu_LocalHeap_Free(ptr00);
    bbcu_LocalHeap_Free(ptr1);
    bbcu_LocalHeap_Free(ptr02);

#if 0
    std::cout << "---- Test_MicroMlp_Forward ----" << std::endl;
    Test_MicroMlp_Forward();
    
    std::cout << "---- Test_MicroMlp_Backward ----" << std::endl;
    Test_MicroMlp_Backward();
#endif

#if 1
    std::cout << "---- Test_StochasticLut6_Forward ----" << std::endl;
    Test_StochasticLut6_Forward();
    
//  std::cout << "---- Test_StochasticLut6_Backward ----" << std::endl;
//  Test_StochasticLut6_Backward();
#endif

    return 0;
}


