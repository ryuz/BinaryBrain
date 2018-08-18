


#pragma once


#include <Eigen/Core>


// NeuralNetの抽象クラス
template <typename T=float, typename INDEX=size_t>
class NeuralNetSoftmax : public NeuralNetLayer<T, INDEX>
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;
	typedef Eigen::Matrix<T, -1, 1>						Vector;
	
	INDEX		m_frame_size;
	INDEX		m_node_size;
	
	const T*	m_inputValue;
	T*			m_outputValue;
	T*			m_inputError;
	const T*	m_outputError;

public:
	NeuralNetSoftmax() {}

	NeuralNetSoftmax(INDEX node_size, INDEX batch_size=1)
	{
		Setup(node_size, batch_size);
	}

	~NeuralNetSoftmax() {}		// デストラクタ

	void Setup(INDEX node_size, INDEX batch_size =1)
	{
		m_frame_size = batch_size;
		m_node_size = node_size;
	}

	void SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }
	void SetInputValuePtr(const void* inputValue) { m_inputValue = (const T*)inputValue; }
	void SetOutputValuePtr(void* outputValue) { m_outputValue = (T*)outputValue; }
	void SetOutputErrorPtr(const void* outputError) { m_outputError = (const T*)outputError; }
	void SetInputErrorPtr(void* inputError) { m_inputError = (T*)inputError; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputValueBitSize(void) const { return sizeof(T) * 8; }
	int   GetInputErrorBitSize(void) const { return sizeof(T) * 8; }
	int   GetOutputValueBitSize(void) const { return sizeof(T) * 8; }
	int   GetOutputErrorBitSize(void) const { return sizeof(T) * 8; }

	void Forward(void)
	{
		Eigen::Map<Matrix> inputValue((T*)m_inputValue, m_frame_size, m_node_size);
		Eigen::Map<Matrix> outputValue(m_outputValue, m_frame_size, m_node_size);

		auto valueExp = (inputValue.colwise() - inputValue.rowwise().maxCoeff()).array().exp();
		auto valueSum = valueExp.rowwise().sum();
		outputValue = valueExp.array().colwise() / valueSum.array();
	}

	void Backward(void)
	{
		Eigen::Map<Matrix> outputError((T*)m_outputError, m_frame_size, m_node_size);
		Eigen::Map<Matrix> inputError(m_inputError, m_frame_size, m_node_size);

		inputError = outputError;
	}

	void Update(double learning_rate)
	{
	}

};

