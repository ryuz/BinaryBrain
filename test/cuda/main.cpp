#include <iostream>
#include <chrono>
#include <random>


int Test_MicroMlp_Forward(void);
int Test_MicroMlp_Backward(void);


int main()
{
	std::cout << "---- Test_MicroMlp_Forward ----" << std::endl;
    Test_MicroMlp_Forward();
	
	std::cout << "---- Test_MicroMlp_Backward ----" << std::endl;
    Test_MicroMlp_Backward();

	return 0;
}


