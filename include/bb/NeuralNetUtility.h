// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

namespace bb {

#include <cstdint>

// argmax
template <typename T = float, typename INDEX = int>
INDEX argmax(std::vector<T> vec)
{
	auto maxIt = std::max_element(vec.begin(), vec.end());
	return (INDEX)std::distance(vec.begin(), maxIt);
}


// データ型を変換(uint8_t(0-255) -> float(0.0-1.0)への変換に使う想定)
template <typename ST = std::uint8_t, typename DT = float, typename MT = float>
std::vector<DT> DataTypeConvert(const std::vector<ST>& src, MT mul=(MT)1)
{
	std::vector<DT> dst(src.size());
	for (size_t i = 0; i < src.size(); ++i) {
		dst[i] = (DT)(src[i] * mul);
	}

	return dst;
}

template <typename ST = std::uint8_t, typename DT = float, typename MT = float>
std::vector< std::vector<DT> > DataTypeConvert(const std::vector< std::vector<ST> >& src, MT mul = (MT)1)
{
	std::vector< std::vector<DT> > dst(src.size());
	for (size_t i = 0; i < src.size(); ++i) {
		dst[i] = DataTypeConvert<ST, DT, MT>(src[i], mul);
	}

	return dst;
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