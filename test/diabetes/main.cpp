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
void DiabetesRegressionStochasticLut6(int epoch_size, size_t mini_batch_size);
void DiabetesRegressionMicroMlpLut(int epoch_size, size_t mini_batch_size, size_t mux_size);


// メイン関数
int main()
{
	omp_set_num_threads(4);

//	DiabetesAffineRegression(32, 16);
//	DiabetesRegressionStochasticLut6(8, 16);
    DiabetesRegressionMicroMlpLut(8, 16, 15);

	return 0;
}


// end of file
