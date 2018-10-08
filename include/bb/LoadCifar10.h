// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>

#include "TrainData.h"


namespace bb {


template <typename T = float>
class LoadCifar10
{
public:
	static bool ReadFile(std::istream& is, std::vector< std::vector<T> >& x, std::vector< std::vector<T> >& y)
	{
		std::vector<T> onehot(10);
		std::vector<T> image(32 * 32 * 3);

		while (!is.eof()) {
			std::uint8_t label;
			is.read((char*)&label, 1);
			if (is.eof()) { break; }
			std::fill(onehot.begin(), onehot.end(), (T)0.0);
			onehot[label] = (T)1.0;
			y.push_back(onehot);

			std::array<std::uint8_t, 32 * 32 * 3> image_u8;
			is.read((char*)&image_u8[0], 32 * 32 * 3);

			for (int i = 0; i < 32 * 32 * 3; ++i) {
				image[i] = (T)image_u8[i] / (T)255.0;
			}
			x.push_back(image);
		}

		return true;
	}

	static bool ReadFile(std::string filename, std::vector< std::vector<T> >& x, std::vector< std::vector<T> >& y)
	{
		std::ifstream ifs(filename, std::ios::binary);
		if (!ifs.is_open()) {
			return false;
		}

		return ReadFile(ifs, x, y);
	}

	static bool Load(std::vector< std::vector<T> >& x_train, std::vector< std::vector<T> >& y_train,
		std::vector< std::vector<T> >& x_test, std::vector< std::vector<T> >& y_test)
	{
		if (!ReadFile("data_batch_1.bin", x_train, y_train)) { return false; }
		if (!ReadFile("data_batch_2.bin", x_train, y_train)) { return false; }
		if (!ReadFile("data_batch_3.bin", x_train, y_train)) { return false; }
		if (!ReadFile("data_batch_4.bin", x_train, y_train)) { return false; }
		if (!ReadFile("data_batch_5.bin", x_train, y_train)) { return false; }
		if (!ReadFile("test_batch.bin", x_test, y_test)) { return false; }
	}
	
	static TrainData<T> Load(void)
	{
		TrainData<T>	td;
		if (!Load(td.x_train, td.y_train, td.x_test, td.y_test)) {
			td.clear();
		}
		return td;
	}
};

}

