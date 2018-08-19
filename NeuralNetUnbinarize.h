


#pragma once

#include <random>


#include "NeuralNetRealBinaryConverter.h"


// NeuralNetの抽象クラス
template <typename T=float, typename INDEX=size_t>
class NeuralNetUnbinarize : public NeuralNetRealBinaryConverter<T, INDEX>
{
protected:
//	const void*	m_inputValue;
//	void*		m_outputValue;
//	void*		m_inputError;
//	const void*	m_outputError;

public:
	NeuralNetUnbinarize() {}
	
	NeuralNetUnbinarize(INDEX node_size, INDEX mux_size, INDEX batch_size=1, std::uint64_t seed=1)
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
	
	int   GetInputValueBitSize(void) const { return 1; }
	int   GetInputErrorBitSize(void) const { return 1; }
	int   GetOutputValueBitSize(void) const { return sizeof(T) * 8; }
	int   GetOutputErrorBitSize(void) const { return sizeof(T) * 8; }
	
	void Forward(void)
	{
		BinaryToReal(GetInputValueBuffer().GetBuffer(), GetOutputValueBuffer().GetBuffer());
	}
	
	void Backward(void)
	{
		RealToBinary(GetOutputErrorBuffer().GetBuffer(), GetInputErrorBuffer().GetBuffer());
	}

	void Update(double learning_rate)
	{
	}

};

