// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

//#ifndef EIGEN_MPL2_ONLY
//#define EIGEN_MPL2_ONLY
//#endif
#include <Eigen/Core>

#include "NeuralNetLayerBuf.h"


namespace bb {


// シグモイド(活性化関数)
template <typename T = float, typename INDEX = size_t>
class NeuralNetSigmoid : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;

	INDEX		m_mux_size = 1;
	INDEX		m_frame_size = 1;
	INDEX		m_node_size = 0;
	bool		m_binary_mode = false;

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

	void  SetBinaryMode(bool enable)
	{
		m_binary_mode = enable;
	}

	void  SetMuxSize(INDEX mux_size) { m_mux_size = mux_size; }
	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size * m_mux_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	void Forward(bool train = true)
	{
		if (m_binary_mode) {
			// Binarize
			auto x = GetInputSignalBuffer();
			auto y = GetOutputSignalBuffer();
			for (INDEX node = 0; node < m_node_size; ++node) {
				for (INDEX frame = 0; frame < m_frame_size; ++frame) {
					y.Set<T>(frame, node, x.Get<T>(frame, node) >(T)0.0 ? (T)1.0 : (T)0.0);
				}
			}
		}
		else {
			// Sigmoid
			Eigen::Map<Matrix> x((T*)m_input_signal_buffer.GetBuffer(), m_input_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
			Eigen::Map<Matrix> y((T*)m_output_signal_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);

			y = ((x * -1).array().exp() + 1.0).inverse();
		}
	}

	void Backward(void)
	{
		if (m_binary_mode) {
			// Binarize
			auto dx = GetInputErrorBuffer();
			auto dy = GetOutputErrorBuffer();
			auto x = GetInputSignalBuffer();
			auto y = GetOutputSignalBuffer();

			for (INDEX node = 0; node < m_node_size; ++node) {
				for (INDEX frame = 0; frame < m_frame_size; ++frame) {
					// hard-tanh
					auto err = dy.Get<T>(frame, node);
					auto sig = x.Get<T>(frame, node);
					dx.Set<T>(frame, node, (sig >= (T)-1.0 && sig <= (T)1.0) ? err : 0);
				}
			}
		}
		else {
			// Sigmoid
			Eigen::Map<Matrix> y((T*)m_output_signal_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
			Eigen::Map<Matrix> dy((T*)m_output_error_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
			Eigen::Map<Matrix> dx((T*)m_input_error_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);

			dx = dy.array() * (-y.array() + 1) * y.array();
		}
	}

	void Update(void)
	{
	}

};

}
