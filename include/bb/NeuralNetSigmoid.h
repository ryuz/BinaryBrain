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


// NeuralNetの抽象クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNetSigmoid : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;

	INDEX		m_frame_size = 1;
	INDEX		m_node_size = 0;

public:
	NeuralNetSigmoid() {}

	NeuralNetSigmoid(INDEX node_size)
	{
		Resize(node_size);
	}

	~NeuralNetSigmoid() {}		// デストラクタ

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


//	std::mt19937_64 m_mt;

	void Forward(void)
	{
		Eigen::Map<Matrix> inputValue((T*)m_input_value_buffer.GetBuffer(), m_input_value_buffer.GetFrameStride() / sizeof(T), m_node_size);
		Eigen::Map<Matrix> outputValue((T*)m_output_value_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);

		outputValue = ((inputValue * -1).array().exp() + 1.0).inverse();

#if 0
		// ２値化実験
		{
			std::uniform_real_distribution<T> dist(0, 1);

			auto buf = GetOutputValueBuffer();
			auto node_size = GetOutputNodeSize();
			auto frame_size = GetOutputFrameSize();
			for (size_t node = 0; node < node_size; ++node) {
				for (size_t frame = 0; frame < frame_size; ++frame) {
					if (m_mt() % 2 == 0) {
						T val = buf.Get<float>(frame, node);
						if (val < 0 || val > 1) {
							std::cout << "over " << val << std::endl;
						}
			//			buf.Set<float>(frame, node, val > dist(m_mt) ? 1.0 : 0.0);
						buf.Set<float>(frame, node, val > 0.5 ? 1.0 : 0.0);
					}
				}
			}
		}
#endif
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
