#include <iostream>
#include <chrono>
#include <random>


int MicroMlp_Test(void);
int UnifiedMemory(void);
int cudnn_test();

int main()
{
//	return cudnn_test();

//	return UnifiedMemory();
	return MicroMlp_Test();

	return 0;
}


