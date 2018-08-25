// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <Eigen/Core>


namespace bb {


// NeuralNetの抽象クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNetSigmoid : public NeuralNetLayer<T, INDEX>
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;

	INDEX		m_frame_size;
	INDEX		m_node_size;

public:
	NeuralNetSigmoid() {}

	NeuralNetSigmoid(INDEX node_size, INDEX batch_size = 1)
	{
		Setup(node_size, batch_size);
	}

	~NeuralNetSigmoid() {}		// デストラクタ

	void Setup(INDEX node_size, INDEX batch_size = 1)
	{
		m_frame_size = batch_size;
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

	void Forward(void)
	{
		Eigen::Map<Matrix> inputValue((T*)m_input_value_buffer.GetBuffer(), m_input_value_buffer.GetFrameStride() / sizeof(T), m_node_size);
		Eigen::Map<Matrix> outputValue((T*)m_output_value_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);

		outputValue = ((inputValue * -1).array().exp() + 1.0).inverse();
	}

	void Backward(void)
	{
		Eigen::Map<Matrix> outputValue((T*)m_output_value_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);
		Eigen::Map<Matrix> outputError((T*)m_output_error_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);
		Eigen::Map<Matrix> inputError((T*)m_input_error_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);

		inputError = outputError.array() * (-outputValue.array() + 1) * outputValue.array();
	}

	void Update(double learning_rate)
	{
	}

};

}
