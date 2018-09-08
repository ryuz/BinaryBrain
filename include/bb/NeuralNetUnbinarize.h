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

	NeuralNetUnbinarize(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1)
	{
		Resize(input_node_size, output_node_size);
		InitializeCoeff(seed);
	}

	~NeuralNetUnbinarize() {}		// デストラクタ

	void Resize(INDEX input_node_size, INDEX output_node_size)
	{
		NeuralNetRealBinaryConverter<T, INDEX>::Resize(output_node_size, input_node_size);
	}

	INDEX GetInputFrameSize(void) const { return GetBinaryFrameSize(); }
	INDEX GetInputNodeSize(void) const { return GetBinaryNodeSize(); }
	INDEX GetOutputFrameSize(void) const { return GetRealFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return GetRealNodeSize(); }

	int   GetInputValueDataType(void) const { return BB_TYPE_BINARY; }
	int   GetOutputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	void Forward(bool train = true)
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
