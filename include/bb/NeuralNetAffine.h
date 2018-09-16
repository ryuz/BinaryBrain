// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>

#ifndef EIGEN_MPL2_ONLY
#define EIGEN_MPL2_ONLY
#endif
#include <Eigen/Core>

#include "NeuralNetLayerBuf.h"


namespace bb {


// Affineレイヤー
template <typename T = float, typename INDEX = size_t>
class NeuralNetAffine : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;
	typedef Eigen::Matrix<T, 1, -1>						Vector;

	INDEX		m_mux_size = 1;
	INDEX		m_frame_size = 1;
	INDEX		m_input_size = 0;
	INDEX		m_output_size = 0;

	Matrix		m_W;
	Vector		m_b;
	Matrix		m_dW;
	Vector		m_db;

public:
	NeuralNetAffine() {}

	NeuralNetAffine(INDEX input_size, INDEX output_size, std::uint64_t seed=1)
	{
		Resize(input_size, output_size);
		InitializeCoeff(seed);
	}

	~NeuralNetAffine() {}		// デストラクタ

	void Resize(INDEX input_size, INDEX output_size)
	{
		m_input_size = input_size;
		m_output_size = output_size;
		m_W = Matrix::Random(input_size, output_size);
		m_b = Vector::Random(output_size);
		m_dW = Matrix::Zero(input_size, output_size);
		m_db = Vector::Zero(output_size);
	}

	void InitializeCoeff(std::uint64_t seed)
	{
		std::mt19937_64 mt(seed);
//		std::uniform_real_distribution<T> real_dist((T)-1, (T)+1);
		std::normal_distribution<T>		real_dist((T)0.0, (T)1.0);

		for (INDEX i = 0; i < m_input_size; ++i) {
			for (INDEX j = 0; j < m_output_size; ++j) {
				m_W(i, j) = real_dist(mt);
			}
		}

		for (INDEX j = 0; j < m_output_size; ++j) {
			m_b(j) = real_dist(mt);
		}
	}

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_input_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_output_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	T& W(INDEX input, INDEX output) { return m_W(input, output); }
	T& b(INDEX output) { return m_b(output); }
	T& dW(INDEX input, INDEX output) { return m_dW(input, output); }
	T& db(INDEX output) { return m_db(output); }

	void  SetMuxSize(INDEX mux_size) { m_mux_size = mux_size; }

	void  SetBatchSize(INDEX batch_size)
	{
		m_frame_size = batch_size * m_mux_size;
	}

	void Forward(bool train = true)
	{
		Eigen::Map<Matrix> x((T*)m_input_signal_buffer.GetBuffer(), m_input_signal_buffer.GetFrameStride() / sizeof(T), m_input_size);
		Eigen::Map<Matrix> y((T*)m_output_signal_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_output_size);
		
		y = x * m_W;
		y.rowwise() += m_b;
	}
	
	void Backward(void)
	{
		Eigen::Map<Matrix> dy((T*)m_output_error_buffer.GetBuffer(), m_output_error_buffer.GetFrameStride() / sizeof(T), m_output_size);
		Eigen::Map<Matrix> dx((T*)m_input_error_buffer.GetBuffer(), m_input_error_buffer.GetFrameStride() / sizeof(T), m_input_size);
		Eigen::Map<Matrix> x((T*)m_input_signal_buffer.GetBuffer(), m_input_signal_buffer.GetFrameStride() / sizeof(T), m_input_size);

		dx = dy * m_W.transpose();
		m_dW = x.transpose() * dy;
		m_db = dy.colwise().sum();
	}

	void Update(double learning_rate)
	{
		m_W -= m_dW * learning_rate;
		m_b -= m_db * learning_rate;
	}
};

}