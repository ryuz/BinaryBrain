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
void DiabetesRegressionMicroMlpLut(int epoch_size, size_t mini_batch_size, size_t mux_size);
void DiabetesRegressionStochasticLut6(int epoch_size, size_t mini_batch_size);


// メイン関数
int main()
{
    omp_set_num_threads(4);

    // 普通のDenseAffineでの回帰
    DiabetesAffineRegression(32, 16);

    // μMLPによるバイナリネットでの回帰
    DiabetesRegressionMicroMlpLut(32, 16, 255);

    // 確率的LUTによる回帰と、バイナリネットでの再生
    DiabetesRegressionStochasticLut6(64, 16);

    return 0;
}


// end of file
