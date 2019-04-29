// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <cstdint>
#include <vector>

#include "bb/DataType.h"


namespace bb {


template <typename T = float>
class LoadXor
{
public:
    static TrainData<T> Load(int bit_size, int mul=1)
    {
        TrainData<T>    td;
        
        int data_size = (1 << bit_size);
        td.x_train.resize(data_size);
        td.t_train.resize(data_size);
        for (size_t i = 0; i < data_size; ++i) {
            td.x_train[i].resize(bit_size);
            td.t_train[i].resize(1);

            int y = 0;
            for (int j = 0; j < bit_size; ++j) {
                if ((i >> j) & 1) {
                    y ^= 1;
                    td.x_train[i][j] = (T)1.0;
                }
                else {
                    td.x_train[i][j] = (T)0.0;
                }
            }
            td.t_train[i][0] = (T)y;
        }

        td.x_test = td.x_train;
        td.t_test = td.t_train;

        td.x_train.resize(data_size*mul);
        td.t_train.resize(data_size*mul);
        for (size_t i = data_size; i < data_size*mul; ++i) {
            td.x_train[i] = td.x_train[i%data_size];
            td.t_train[i] = td.t_train[i%data_size];
        }

        td.x_shape = indices_t({bit_size});
        td.t_shape = indices_t({1});

        return td;
    }
};


}

