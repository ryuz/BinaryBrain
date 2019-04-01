// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <omp.h>

#include "bb/Manager.h"


void MnistSimpleMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistSequentialMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistSimpleCnnMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistDenseAffine(int epoch_size, size_t mini_batch_size);
void MnistMiniLutCnn(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistDeepMicroMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistRealLut4(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistStochasticLut6(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistStochasticLut6Cnn(int epoch_size, size_t mini_batch_size, bool binary_mode);


// メイン関数
int main()
{
	omp_set_num_threads(4);

//    MnistStochasticLut6Cnn(0, 16, true);
    MnistStochasticLut6(8, 256, true);

//    MnistRealLut4(64, 256, true);

//   MnistDeepMicroMlp(128, 64, true);

//    MnistMiniLutCnn(16, 16, true);

//  bb::Manager::SetHostOnly(true); // GPU版でもCPUのみ利用する場合

//  MnistDenseAffine(64, 32);
//  MnistSimpleCnnMlp(64, 16, true);
//  MnistSequentialMicroMlp(64, 256, true);
//  MnistSimpleMicroMlp(64, 16*1024, true);

//	getchar();
	return 0;
}


// end of file
