// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <Eigen/Core>

#include "bb/NeuralNetLayerBuf.h"


namespace bb {


// シグモイド(活性化関数)
template <typename T = float>
class NeuralNetSigmoid : public NeuralNetLayerBuf<T>
{
protected:
//	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;

	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using Stride = Eigen::Stride<Eigen::Dynamic, 1>;
	using MatMap = Eigen::Map<Matrix, 0, Stride>;

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

	std::string GetClassName(void) const { return "NeuralNetSigmoid"; }

	void Resize(INDEX node_size)
	{
		m_node_size = node_size;
	}

	void  SetBinaryMode(bool enable)
	{
		m_binary_mode = enable;
	}

	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		if (m_binary_mode) {
			for (auto& v : input_value) {
				v = v > (T)0 ? (T)1 : (T)0;
			}
		}
		else {
			for (auto& v : input_value) {
				v = (T)1 / ((T)1 + exp(-v));
			}
		}
		return input_value;
	}

	void Forward(bool train = true)
	{
		if (m_binary_mode) {
			// Binarize
			auto x = this->GetInputSignalBuffer();
			auto y = this->GetOutputSignalBuffer();

			#pragma omp parallel for
			for (int node = 0; node < (int)m_node_size; ++node) {
				for (INDEX frame = 0; frame < m_frame_size; ++frame) {
					y.template Set<T>(frame, node, x.template Get<T>(frame, node) >(T)0.0 ? (T)1.0 : (T)0.0);
				}
			}
		}
		else {
			// Sigmoid
//			Eigen::Map<Matrix> x((T*)this->m_input_signal_buffer.GetBuffer(), this->m_input_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
//			Eigen::Map<Matrix> y((T*)this->m_output_signal_buffer.GetBuffer(), this->m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
			MatMap x((T*)this->m_input_signal_buffer.GetBuffer(), m_frame_size, m_node_size, Stride(this->m_input_signal_buffer.GetFrameStride() / sizeof(T), 1));
			MatMap y((T*)this->m_output_signal_buffer.GetBuffer(), m_frame_size, m_node_size, Stride(this->m_output_signal_buffer.GetFrameStride() / sizeof(T), 1));

			y = ((x * -1).array().exp() + 1.0).inverse();
		}
	}

	void Backward(void)
	{
		if (m_binary_mode) {
			// Binarize
			auto dx = this->GetInputErrorBuffer();
			auto dy = this->GetOutputErrorBuffer();
			auto x = this->GetInputSignalBuffer();

			#pragma omp parallel for
			for (int node = 0; node < (int)m_node_size; ++node) {
				for (INDEX frame = 0; frame < m_frame_size; ++frame) {
					// hard-tanh
					auto err = dy.template Get<T>(frame, node);
					auto sig = x.template Get<T>(frame, node);
					dx.template Set<T>(frame, node, (sig >= (T)-1.0 && sig <= (T)1.0) ? err : 0);
				}
			}
		}
		else {
			// Sigmoid
	//		Eigen::Map<Matrix> y((T*)this->m_output_signal_buffer.GetBuffer(), this->m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
	//		Eigen::Map<Matrix> dy((T*)this->m_output_error_buffer.GetBuffer(), this->m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
	//		Eigen::Map<Matrix> dx((T*)this->m_input_error_buffer.GetBuffer(), this->m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
			MatMap y((T*)this->m_output_signal_buffer.GetBuffer(), m_frame_size, m_node_size, Stride(this->m_output_signal_buffer.GetFrameStride() / sizeof(T), 1));
			MatMap dy((T*)this->m_output_error_buffer.GetBuffer(), m_frame_size, m_node_size, Stride(this->m_output_error_buffer.GetFrameStride() / sizeof(T), 1));
			MatMap dx((T*)this->m_input_error_buffer.GetBuffer(), m_frame_size, m_node_size, Stride(this->m_input_error_buffer.GetFrameStride() / sizeof(T), 1));

			dx = dy.array() * (-y.array() + 1) * y.array();
		}
	}

	void Update(void)
	{
	}

};

}
