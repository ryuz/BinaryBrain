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

#include "TrainData.h"


namespace bb {


template <typename T = float>
class LoadXor
{
public:
	static TrainData<T> Load(int bit_size, int mul=1)
	{
		TrainData<T>	td;
		
		int data_size = (1 << bit_size);
		td.x_train.resize(data_size);
		td.y_train.resize(data_size);
		for (size_t i = 0; i < data_size; ++i) {
			td.x_train[i].resize(bit_size);
			td.y_train[i].resize(1);

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
			td.y_train[i][0] = (T)y;
		}

		td.x_test = td.x_train;
		td.y_test = td.y_train;

		td.x_train.resize(data_size*mul);
		td.y_train.resize(data_size*mul);
		for (size_t i = data_size; i < data_size*mul; ++i) {
			td.x_train[i] = td.x_train[i%data_size];
			td.y_train[i] = td.y_train[i%data_size];
		}

		return td;
	}
};


}