


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

	int   GetInputValueBitSize(void) const { return sizeof(T) * 8; }
	int   GetInputErrorBitSize(void) const { return sizeof(T) * 8; }
	int   GetOutputValueBitSize(void) const { return 1; }
	int   GetOutputErrorBitSize(void) const { return 1; }

	void Forward(void)
	{
		RealToBinary(GetInputValueBuffer().GetBuffer(), GetOutputValueBuffer().GetBuffer());
	}

	void Backward(void)
	{
		BinaryToReal(GetOutputErrorBuffer().GetBuffer(), GetInputErrorBuffer().GetBuffer());
	}

	void Update(double learning_rate)
	{
	}

};

