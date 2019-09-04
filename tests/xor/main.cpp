// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
// --------------------------------------------------------------------------

#include <omp.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>


void XorMicroMlp(int epoch_size, bool binary_mode);
void StochasticLut6(int epoch_size, bool binary_mode);


// メイン関数
int main()
{
    omp_set_num_threads(1);

//  XorMicroMlp(65536, true);
    StochasticLut6(65536, true);

    return 0;
}

