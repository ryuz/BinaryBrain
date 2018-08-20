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
class NeuralNetBinarize : public NeuralNetRealBinaryConverter<T, INDEX>
{
public:
	NeuralNetBinarize() {}

	NeuralNetBinarize(INDEX input_node_size, INDEX output_node_size, INDEX mux_size, INDEX batch_size = 1, std::uint64_t seed = 1)
	{
		Setup(input_node_size, input_node_size, mux_size, batch_size, seed);
	}

	~NeuralNetBinarize() {}		// デストラクタ

	INDEX GetInputFrameSize(void) const { return GetRealFrameSize(); }
	INDEX GetInputNodeSize(void) const { return GetRealNodeSize(); }
	INDEX GetOutputFrameSize(void) const { return GetBinaryFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return GetBinaryNodeSize(); }

	int   GetInputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputValueDataType(void) const { return NN_TYPE_BINARY; }
	int   GetOutputErrorDataType(void) const { return NN_TYPE_BINARY; }

	void Forward(void)
	{
		RealToBinary(GetInputValueBuffer(), GetOutputValueBuffer());
	}

	void Backward(void)
	{
		BinaryToReal(GetOutputErrorBuffer(), GetInputErrorBuffer());
	}

	void Update(double learning_rate)
	{
	}

};

}