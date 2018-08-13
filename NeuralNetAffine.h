


#pragma once


#include <Eigen/Core>
#include <iostream>

// NeuralNetの抽象クラス
template <typename T=float, typename ET = float, typename INDEX=size_t>
class NeuralNetAffine
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;
	typedef Eigen::Matrix<T, 1, -1>						Vector;

public:
	INDEX	m_frame_size;
	INDEX	m_input_size;
	INDEX	m_output_size;

	Matrix		m_W;
	Vector		m_b;
	Matrix		m_dW;
	Vector		m_db;

	const T*	m_inputValue;
	T*			m_outputValue;

	T*			m_inputError;
	const T*	m_outputError;

public:
	NeuralNetAffine() {}

	NeuralNetAffine(INDEX input_size, INDEX output_size, INDEX frame_size=1)
	{
		Setup(input_size, output_size, frame_size);
	}

	~NeuralNetAffine() {}		// デストラクタ

	void Setup(INDEX input_size, INDEX output_size, INDEX frame_size=1)
	{
		m_frame_size  = frame_size;
		m_input_size = input_size;
		m_output_size = output_size;
		m_W = Matrix::Random(input_size, output_size);
		m_b = Vector::Random(output_size);
		m_dW = Matrix::Zero(input_size, output_size);
		m_db = Vector::Zero(output_size);
	}

	void SetInputValuePtr(const T* inputValue) { m_inputValue = inputValue; }
	void SetOutputValuePtr(T* outputValue) { m_outputValue = outputValue; }
	
	void SetOutputErrorPtr(const ET* outputError) { m_outputError = outputError; }
	void SetInputErrorPtr(ET* inputError) { m_inputError = inputError; }

	T& W(INDEX input, INDEX output) { return m_W(input, output); }
	T& b(INDEX output) { return m_b(output); }
	T& dW(INDEX input, INDEX output) { return m_dW(input, output); }
	T& db(INDEX output) { return m_db(output); }
	
	INDEX GetFrameSize(void) { return m_frame_size; }
	INDEX GetInputSize(void) { return m_input_size; }
	INDEX GetOutputSize(void) { return m_output_size; }

	void Forward(void)
	{
		Eigen::Map<Matrix> inputValue((T*)m_inputValue, m_frame_size, m_input_size);
		Eigen::Map<Matrix> outputValue(m_outputValue, m_frame_size, m_output_size);

		outputValue = inputValue * m_W;
		outputValue.rowwise() += m_b;
	}

	void Backward(void)
	{
		Eigen::Map<Matrix> outputError((T*)m_outputError, m_frame_size, m_output_size);
		Eigen::Map<Matrix> inputError(m_inputError, m_frame_size, m_input_size);
		Eigen::Map<Matrix> inputValue((T*)m_inputValue, m_frame_size, m_input_size);

		inputError = outputError * m_W.transpose();
		m_dW = inputValue.transpose() * outputError;
		m_db = outputError.colwise().sum();
	}

	void Update(double learning_rate)
	{
		m_W -= m_dW * learning_rate;
		m_b -= m_db * learning_rate;
	}

};

