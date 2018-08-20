// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <Eigen/Core>
#include "NeuralNetLayer.h"


namespace bb {


// Affineレイヤー
template <typename T = float, typename INDEX = size_t>
class NeuralNetAffine : public NeuralNetLayer<T, INDEX>
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;
	typedef Eigen::Matrix<T, 1, -1>						Vector;

	INDEX		m_frame_size;
	INDEX		m_input_size;
	INDEX		m_output_size;

	Matrix		m_W;
	Vector		m_b;
	Matrix		m_dW;
	Vector		m_db;

public:
	NeuralNetAffine() {}

	NeuralNetAffine(INDEX input_size, INDEX output_size, INDEX batch_size = 1)
	{
		Setup(input_size, output_size, batch_size);
	}

	~NeuralNetAffine() {}		// デストラクタ

	void Setup(INDEX input_size, INDEX output_size, INDEX batch_size = 1)
	{
		m_frame_size = batch_size;
		m_input_size = input_size;
		m_output_size = output_size;
		m_W = Matrix::Random(input_size, output_size);
		m_b = Vector::Random(output_size);
		m_dW = Matrix::Zero(input_size, output_size);
		m_db = Vector::Zero(output_size);
	}

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_input_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_output_size; }

	int   GetInputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	T& W(INDEX input, INDEX output) { return m_W(input, output); }
	T& b(INDEX output) { return m_b(output); }
	T& dW(INDEX input, INDEX output) { return m_dW(input, output); }
	T& db(INDEX output) { return m_db(output); }

	void  SetBatchSize(INDEX batch_size)
	{
		m_frame_size = batch_size;
	}

	void Forward(void)
	{
		Eigen::Map<Matrix> inputValue((T*)m_input_value_buffer.GetBuffer(), m_input_value_buffer.GetStride() / sizeof(T), m_input_size);
		Eigen::Map<Matrix> outputValue((T*)m_output_value_buffer.GetBuffer(), m_output_value_buffer.GetStride() / sizeof(T), m_output_size);

		outputValue = inputValue * m_W;
		outputValue.rowwise() += m_b;
	}

	void Backward(void)
	{
		Eigen::Map<Matrix> outputError((T*)m_output_error_buffer.GetBuffer(), m_output_error_buffer.GetStride() / sizeof(T), m_output_size);
		Eigen::Map<Matrix> inputError((T*)m_input_error_buffer.GetBuffer(), m_input_error_buffer.GetStride() / sizeof(T), m_input_size);
		Eigen::Map<Matrix> inputValue((T*)m_input_value_buffer.GetBuffer(), m_input_value_buffer.GetStride() / sizeof(T), m_input_size);

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

}