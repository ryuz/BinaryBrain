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
void DiabetesRegression(int epoch_size, size_t mini_batch_size);


// メイン関数
int main()
{
	omp_set_num_threads(4);

	DiabetesRegression(64, 16);
	DiabetesAffineRegression(64, 16);
	
	return 0;
}


// end of file
