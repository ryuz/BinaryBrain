// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <omp.h>


void MnistSimpleMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistSequentialMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);

// メイン関数
int main()
{
	omp_set_num_threads(4);

    MnistSequentialMicroMlp(64, 64, true);
//	MnistSimpleMicroMlp(64, 1024, true);

//	getchar();
	return 0;
}

