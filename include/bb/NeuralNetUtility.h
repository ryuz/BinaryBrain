// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

namespace bb {

// argmax
template <typename T = float, typename INDEX = int>
INDEX argmax(std::vector<T> vec)
{
	auto maxIt = std::max_element(vec.begin(), vec.end());
	return (INDEX)std::distance(vec.begin(), maxIt);
}


// ラベル値をワンホットデータに変換
template <typename LT, typename T=float>
std::vector< std::vector<T> > LabelToOnehot(const std::vector<LT>& labels, LT label_size, T f_val = (T)0.0, T t_val = (T)1.0)
{
	std::vector < std::vector<T> >	onehot(labels.size());
	for (size_t i = 0; i < labels.size(); ++i) {
		onehot[i].resize(label_size);
		for (size_t j = 0; j < label_size; ++j) {
			onehot[i][j] = (j == labels[i]) ? t_val : f_val;
		}
	}

	return onehot;
}


}