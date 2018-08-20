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

	NeuralNetUnbinarize(INDEX node_size, INDEX mux_size, INDEX batch_size = 1, std::uint64_t seed = 1)
	{
		Setup(node_size, mux_size, batch_size, seed);
	}

	~NeuralNetUnbinarize() {}		// デストラクタ

//	void SetInputValuePtr(const void* inputValue) { m_inputValue = inputValue; }
//	void SetOutputValuePtr(void* outputValue) { m_outputValue = outputValue; }
//	void SetOutputErrorPtr(const void* outputError) { m_outputError = outputError; }
//	void SetInputErrorPtr(void* inputError) { m_inputError = inputError; }

	INDEX GetInputFrameSize(void) const { return GetBinaryFrameSize(); }
	INDEX GetInputNodeSize(void) const { return GetNodeSize(); }
	INDEX GetOutputFrameSize(void) const { return GetRealFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return GetNodeSize(); }

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
