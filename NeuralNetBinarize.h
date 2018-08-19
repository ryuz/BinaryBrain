


#pragma once

#include <random>


#include "NeuralNetRealBinaryConverter.h"


// NeuralNetの抽象クラス
template <typename T=float, typename INDEX=size_t>
class NeuralNetBinarize : public NeuralNetRealBinaryConverter<T, INDEX>
{
protected:
//	const void*	m_inputValue;
//	void*		m_outputValue;
//	void*		m_inputError;
//	const void*	m_outputError;

public:
	NeuralNetBinarize() {}
	
	NeuralNetBinarize(INDEX node_size, INDEX mux_size, INDEX batch_size=1, std::uint64_t seed=1)
	{
		Setup(node_size, mux_size, batch_size, seed);
	}
	
	~NeuralNetBinarize() {}		// デストラクタ

//	void SetInputValuePtr(const void* inputValue) { m_inputValue = inputValue; }
//	void SetOutputValuePtr(void* outputValue) { m_outputValue = outputValue; }
//	void SetOutputErrorPtr(const void* outputError) { m_outputError = outputError; }
//	void SetInputErrorPtr(void* inputError) { m_inputError = inputError; }

	INDEX GetInputFrameSize(void) const { return GetRealFrameSize(); }
	INDEX GetInputNodeSize(void) const { return GetNodeSize(); }
	INDEX GetOutputFrameSize(void) const { return GetBinaryFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return GetNodeSize(); }

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

