#include <iostream>
#include <chrono>
#include <random>


int Test_MicroMlp_Forward(void);
int Test_MicroMlp_Backward(void);

int Test_StochasticLut6_Forward(void);
int Test_StochasticLut6_Backward(void);

int main()
{
#if 0
    std::cout << "---- Test_MicroMlp_Forward ----" << std::endl;
    Test_MicroMlp_Forward();
    
    std::cout << "---- Test_MicroMlp_Backward ----" << std::endl;
    Test_MicroMlp_Backward();
#endif

#if 1
    std::cout << "---- Test_StochasticLut6_Forward ----" << std::endl;
    Test_StochasticLut6_Forward();
    
    std::cout << "---- Test_StochasticLut6_Backward ----" << std::endl;
    Test_StochasticLut6_Backward();
#endif

    return 0;
}


