// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#ifndef EIGEN_MPL2_ONLY
#define EIGEN_MPL2_ONLY
#endif
#include <Eigen/Core>

#include "NeuralNetLayerBuf.h"


namespace bb {


// Softmax(活性化関数)
template <typename T = float, typename INDEX = size_t>
class NeuralNetSoftmax : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;
	typedef Eigen::Matrix<T, -1, 1>						Vector;

	INDEX		m_frame_size = 1;
	INDEX		m_node_size = 0;

public:
	NeuralNetSoftmax() {}

	NeuralNetSoftmax(INDEX node_size)
	{
		Resize(node_size);
	}

	~NeuralNetSoftmax() {}		// デストラクタ

	void Resize(INDEX node_size)
	{
		m_node_size = node_size;
	}

	void SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	void Forward(bool train = true)
	{
		Eigen::Map<Matrix> inputValue((T*)m_input_value_buffer.GetBuffer(), m_input_value_buffer.GetFrameStride() / sizeof(T), m_node_size);
		Eigen::Map<Matrix> outputValue((T*)m_output_value_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);

		auto valueExp = (inputValue.colwise() - inputValue.rowwise().maxCoeff()).array().exp();
		auto valueSum = valueExp.rowwise().sum();
		outputValue = valueExp.array().colwise() / valueSum.array();
	}

	void Backward(void)
	{
		Eigen::Map<Matrix> outputError((T*)m_output_error_buffer.GetBuffer(), m_output_error_buffer.GetFrameStride() / sizeof(T), m_node_size);
		Eigen::Map<Matrix> inputError((T*)m_input_error_buffer.GetBuffer(), m_input_error_buffer.GetFrameStride() / sizeof(T), m_node_size);

		inputError = outputError;
	}

	void Update(double learning_rate)
	{
	}

};


}