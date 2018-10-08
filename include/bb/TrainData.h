// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <vector>


namespace bb {

template <typename T = float>
struct TrainData
{
	std::vector< std::vector<T> >	x_train;
	std::vector< std::vector<T> >	y_train;
	std::vector< std::vector<T> >	x_test;
	std::vector< std::vector<T> >	y_test;

	void clear(void) {
		x_train.clear();
		y_train.clear();
		x_test.clear();
		y_test.clear();
	}

	bool empty(void) {
		return x_train.empty() || y_train.empty() || x_test.empty() || y_test.empty();
	}
};

}

