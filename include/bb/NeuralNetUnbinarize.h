// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <random>


#include "NeuralNetRealBinaryConverter.h"


namespace bb {


// NeuralNetの抽象クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNetUnbinarize : public NeuralNetRealBinaryConverter<T, INDEX>
{
public:
	NeuralNetUnbinarize() {}

	NeuralNetUnbinarize(INDEX input_node_size, INDEX output_node_size, INDEX mux_size, INDEX batch_size = 1, std::uint64_t seed = 1)
	{
		Setup(output_node_size, input_node_size, mux_size, batch_size, seed);
	}

	~NeuralNetUnbinarize() {}		// デストラクタ

	INDEX GetInputFrameSize(void) const { return GetBinaryFrameSize(); }
	INDEX GetInputNodeSize(void) const { return GetBinaryNodeSize(); }
	INDEX GetOutputFrameSize(void) const { return GetRealFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return GetRealNodeSize(); }

	int   GetInputValueDataType(void) const { return NN_TYPE_BINARY; }
	int   GetInputErrorDataType(void) const { return NN_TYPE_BINARY; }
	int   GetOutputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	void Forward(void)
	{
		BinaryToReal(GetInputValueBuffer(), GetOutputValueBuffer());
	}

	void Backward(void)
	{
		RealToBinary(GetOutputErrorBuffer(), GetInputErrorBuffer());
	}

	void Update(double learning_rate)
	{
	}

};

}
