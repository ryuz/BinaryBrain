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
void MnistSimpleCnnMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistDenseAffine(int epoch_size, size_t mini_batch_size);

// メイン関数
int main()
{
	omp_set_num_threads(4);

//  MnistDenseAffine(64, 32);

    MnistSimpleCnnMlp(64, 32, true);
//  MnistSequentialMicroMlp(64, 128, true);
//	MnistSimpleMicroMlp(64, 128, true);

//	getchar();
	return 0;
}

