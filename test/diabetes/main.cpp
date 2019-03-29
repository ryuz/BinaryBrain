// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <omp.h>

#include "bb/Manager.h"

void DiabetesAffineRegression(int epoch_size, size_t mini_batch_size);
void DiabetesRegressionBinaryLut(int epoch_size, size_t mini_batch_size, size_t mux_size);


// メイン関数
int main()
{
	omp_set_num_threads(4);

	DiabetesAffineRegression(32, 16);
	DiabetesRegressionBinaryLut(32, 16, 255);

	return 0;
}


// end of file
