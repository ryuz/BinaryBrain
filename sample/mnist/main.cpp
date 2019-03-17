// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <omp.h>
#include <string.h>

void MnistSimpleLutMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistSimpleLutCnn(int epoch_size, size_t mini_batch_size, bool binary_mode);
void MnistDenseAffine(int epoch_size, size_t mini_batch_size);
void MnistSimpleMicroMlpScratch(int epoch_size, size_t mini_batch_size, bool binary_mode);


// メイン関数
int main(int argc, char *argv[])
{
	omp_set_num_threads(4);

	if ( argc < 2 ) {
		return 1;
	}

	if ( strcmp(argv[1], "All") == 0 || strcmp(argv[1], "LutMlp") == 0 ) {
		MnistSimpleLutMlp(16, 64, true);
	}

	if ( strcmp(argv[1], "All") == 0 || strcmp(argv[1], "LutCnn") == 0 ) {
    	MnistSimpleLutCnn(64, 16, true);
	}

	if ( strcmp(argv[1], "All") == 0 || strcmp(argv[1], "DenseAffine") == 0 ) {
		MnistDenseAffine(16, 64);
	}

	if ( strcmp(argv[1], "All") == 0 || strcmp(argv[1], "SimpleMicroMlpScratch") == 0 ) {
		MnistSimpleMicroMlpScratch(16, 64, true);
	}

	return 0;
}

