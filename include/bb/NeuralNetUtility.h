// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <vector>
#include <random>

#include "NeuralNetType.h"


namespace bb {


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


// ワンホットデータをラベル値に変換
template <typename LT, typename T = float>
std::vector<LT> OnehotToLabel(const std::vector<std::vector<T>>& onehot)
{
	std::vector<LT>	label(onehot.size());
	for (size_t i = 0; i < onehot.size(); ++i) {
		label[i] = argmax<T, LT>(onehot[i]);
	}

	return label;
}


// トレーニングデータセットのシャッフル
template <typename T0>
void ShuffleDataSet(std::uint64_t seed, std::vector<T0>& data0)
{
	size_t data_size = data0.size();

	std::mt19937_64							mt(seed);
	std::uniform_int_distribution<size_t>	rand_distribution(0, data_size - 1);

	for (size_t i = 0; i < data0.size(); ++i) {
		size_t j = rand_distribution(mt);
		std::swap(data0[i], data0[j]);
	}
}

template <typename T0, typename T1>
void ShuffleDataSet(std::uint64_t seed, std::vector<T0>& data0, std::vector<T1>& data1)
{
	size_t data_size = data0.size();

	BB_ASSERT(data1.size() == data_size);

	std::mt19937_64							mt(seed);
	std::uniform_int_distribution<size_t>	rand_distribution(0, data_size - 1);

	for (size_t i = 0; i < data0.size(); ++i) {
		size_t j = rand_distribution(mt);
		std::swap(data0[i], data0[j]);
		std::swap(data1[i], data1[j]);
	}
}

template <typename T0, typename T1, typename T2>
void ShuffleDataSet(std::uint64_t seed, std::vector<T0>& data0, std::vector<T1>& data1, std::vector<T2>& data2)
{
	size_t data_size = data0.size();

	BB_ASSERT(data1.size() == data_size);
	BB_ASSERT(data2.size() == data_size);

	std::mt19937_64							mt(seed);
	std::uniform_int_distribution<size_t>	rand_distribution(0, data_size - 1);

	for (size_t i = 0; i < data0.size(); ++i) {
		size_t j = rand_distribution(mt);
		std::swap(data0[i], data0[j]);
		std::swap(data1[i], data1[j]);
		std::swap(data2[i], data2[j]);
	}
}



// ostream 用 tee
template<class _Elem, class _Traits = std::char_traits<_Elem> >
class basic_streambuf_tee : public std::basic_streambuf<_Elem, _Traits>
{
	typedef std::basic_streambuf<_Elem, _Traits>	stmbuf;
	typedef typename _Traits::int_type				int_type;

	std::vector<stmbuf*>	m_streambufs;
	
	virtual int_type overflow(int_type ch = _Traits::eof())
	{
		for (auto s : m_streambufs) {
			s->sputc(ch);
		}
		return ch;
	}

	int sync(void)
	{
		for (auto s : m_streambufs) {
			s->pubsync();
		}

		return 0;
	}

public:
	basic_streambuf_tee()
	{
		this->setp(0, 0);
	}

	void add(stmbuf* stm)
	{
		m_streambufs.push_back(stm);
	}
};

template<class _Elem, class _Traits = std::char_traits<_Elem> >
class basic_ostream_tee : public std::basic_ostream<_Elem, _Traits>
{
	typedef std::basic_ostream<_Elem, _Traits>	ostm;
	typedef basic_streambuf_tee<_Elem, _Traits> stmbuf;

public:
	basic_ostream_tee() : ostm(new basic_streambuf_tee<_Elem, _Traits>())
	{
	}

	~basic_ostream_tee()
	{
		delete this->rdbuf();
	}

	void add(ostm& stm)
	{
		dynamic_cast<stmbuf *>(this->rdbuf())->add(stm.rdbuf());
	}
};


typedef basic_ostream_tee<char>		ostream_tee;
typedef basic_ostream_tee<wchar_t>	ostream_wtee;


}