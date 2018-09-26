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
#include "NeuralNetOptimizerSgd.h"

namespace bb {


// NeuralNetの抽象クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNetBatchNormalization : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using Stride = Eigen::Stride<Eigen::Dynamic, 1>;
	using MatMap = Eigen::Map<Matrix, 0, Stride>;

	INDEX		m_mux_size = 1;
	INDEX		m_frame_size = 1;
	INDEX		m_node_size = 0;
	
	Vector		m_gamma;
	Vector		m_beta;
	Vector		m_dgamma;
	Vector		m_dbeta;

	std::unique_ptr< ParamOptimizer<T, INDEX> >	m_optimizer_gamma;
	std::unique_ptr< ParamOptimizer<T, INDEX> >	m_optimizer_beta;

	Matrix		m_xn;
	Matrix		m_xc;
	Vector		m_std;

	T			m_momentum = (T)0.01;
	Vector		m_running_mean;
	Vector		m_running_var;

public:
	NeuralNetBatchNormalization() {}

	NeuralNetBatchNormalization(INDEX node_size, const NeuralNetOptimizer<T, INDEX>* optimizer = &NeuralNetOptimizerSgd<>())
	{
		Resize(node_size);
		SetOptimizer(optimizer);
	}

	~NeuralNetBatchNormalization() {}		// デストラクタ


	T& gamma(INDEX node) { return m_gamma(node); }
	T& beta(INDEX node) { return m_beta(node); }
	T& dgamma(INDEX node) { return m_dgamma(node); }
	T& dbeta(INDEX node) { return m_dbeta(node); }
	T& mean(INDEX node) { return m_running_mean(node); }
	T& var(INDEX node) { return m_running_var(node); }


	void Resize(INDEX node_size)
	{
		m_node_size = node_size;
		m_gamma = Vector::Ones(m_node_size);
		m_beta = Vector::Zero(m_node_size);
		m_dgamma = Vector::Zero(m_node_size);
		m_dbeta = Vector::Zero(m_node_size);
		m_running_mean = Vector::Zero(m_node_size);
		m_running_var = Vector::Ones(m_node_size);
	}

	void  SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		m_optimizer_gamma.reset(optimizer->Create(m_node_size));
		m_optimizer_beta.reset(optimizer->Create(m_node_size));
	}

	void  SetMuxSize(INDEX mux_size) {
		m_mux_size = mux_size;
	}

	void SetBatchSize(INDEX batch_size) {
		m_frame_size = batch_size * m_mux_size;
	}

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	T CalcNode(INDEX node, std::vector<T> input_signals) const
	{
		T sig = input_signals[0];
		sig -= m_running_mean(node);
		sig /= (T)sqrt(m_running_var(node) + 10e-7);
		sig = sig * m_gamma(node) + m_beta(node);
		return sig;
	}

	void Forward(bool train = true)
	{
//		Eigen::Map<Matrix> x((T*)m_input_signal_buffer.GetBuffer(), m_input_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
//		Eigen::Map<Matrix> y((T*)m_output_signal_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
		MatMap x((T*)m_input_signal_buffer.GetBuffer(),  m_frame_size, m_node_size, Stride(m_input_signal_buffer.GetFrameStride() / sizeof(T), 1));
		MatMap y((T*)m_output_signal_buffer.GetBuffer(), m_frame_size, m_node_size, Stride(m_output_signal_buffer.GetFrameStride() / sizeof(T), 1));
		Matrix xc;
		Matrix xn;
		
		if (train) {
			Vector mu = x.colwise().mean();
			//	std::cout << "mu =\n" << mu << std::endl;

			xc = x.rowwise() - mu;
			//	std::cout << "xc =\n" << xc << std::endl;

			Vector var = (xc.array() * xc.array()).colwise().mean();
			//	std::cout << "var =\n" << var << std::endl;

			Vector std = (var.array() + (T)10e-7).array().sqrt();
			//	std::cout << "std =\n" << std << std::endl;

			xn = xc.array().rowwise() / std.array();
			//	std::cout << "xn =\n" << xn << std::endl;

			m_xn = xn;
			m_xc = xc;
			m_std = std;
			
			m_running_mean = m_running_mean * m_momentum + mu * (1 - m_momentum);
			m_running_var = m_running_var * m_momentum + var * (1 - m_momentum);
		}
		else{
			xc = x.rowwise() - m_running_mean;
			xn = xc.array().rowwise() / (m_running_var.array() + 10e-7).array().sqrt();
		}
		y = (xn.array().rowwise() * m_gamma.array()).array().rowwise() + m_beta.array();
		//	std::cout << "y =\n" << y << std::endl;
	}

	void Backward(void)
	{
//		Eigen::Map<Matrix> y((T*)m_output_signal_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
//		Eigen::Map<Matrix> dy((T*)m_output_error_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
//		Eigen::Map<Matrix> dx((T*)m_input_error_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
		MatMap y((T*)m_output_signal_buffer.GetBuffer(), m_frame_size, m_node_size, Stride(m_output_signal_buffer.GetFrameStride() / sizeof(T), 1));
		MatMap dy((T*)m_output_error_buffer.GetBuffer(), m_frame_size, m_node_size, Stride(m_output_error_buffer.GetFrameStride() / sizeof(T), 1));
		MatMap dx((T*)m_input_error_buffer.GetBuffer(), m_frame_size, m_node_size, Stride(m_input_error_buffer.GetFrameStride() / sizeof(T), 1));

		
		INDEX frame_size = GetOutputFrameSize();
		T reciprocal_frame_size = (T)1 / (T)frame_size;

//		std::cout << "dy =\n" << dy << std::endl;
		
		Vector dbeta = dy.colwise().sum();
//		std::cout << "dbeta =\n" << dbeta << std::endl;

		Vector dgamma = (m_xn.array() * dy.array()).colwise().sum();
//		std::cout << "dgamma =\n" << dgamma << std::endl;

		Matrix dxn  = dy.array().rowwise() * m_gamma.array();
//		std::cout << "dxn =\n" << dxn << std::endl;

		Matrix dxc = dxn.array().rowwise() / m_std.array();
//		std::cout << "dxc =\n" << dxc << std::endl;

		Vector dstd = -((dxn.array() * m_xc.array()).array().rowwise() / (m_std.array() * m_std.array()).array()).array().colwise().sum();
//		std::cout << "dstd =\n" << dstd << std::endl;

		Vector dvar = m_std.array().inverse() * dstd.array();// *(T)0.5;
//		std::cout << "dvar =\n" << dvar << std::endl;

		dxc = dxc.array() + (m_xc.array().rowwise() * dvar.array() * reciprocal_frame_size).array();	// 2.0f / frame_size
//		std::cout << "dxc =\n" << dxc << std::endl;

		Vector dmu = dxc.colwise().sum();
//		std::cout << "dmu =\n" << dmu << std::endl;

		dx = dxc.array().rowwise() - (dmu.array() * reciprocal_frame_size);
//		std::cout << "dx =\n" << dx << std::endl;
		
		m_dgamma = dgamma;
		m_dbeta = dbeta;
	}

	void Update(void)
	{
		// update
		m_optimizer_gamma->Update(m_gamma, m_dgamma);
		m_optimizer_beta->Update(m_beta, m_dbeta);

#if 0
		std::vector<T> vec_gamma(m_node_size);
		std::vector<T> vec_dgamma(m_node_size);
		std::vector<T> vec_beta(m_node_size);
		std::vector<T> vec_dbeta(m_node_size);

		// copy
		for (INDEX node = 0; node < m_node_size; ++node) {
			vec_gamma[node] = m_gamma(node);
			vec_dgamma[node] = m_dgamma(node);
			vec_beta[node] = m_beta(node);
			vec_dbeta[node] = m_dbeta(node);
		}

		// update
		m_optimizer_gamma->Update(vec_gamma, vec_dgamma);
		m_optimizer_beta->Update(vec_beta, vec_dbeta);

		// copy back
		for (INDEX node = 0; node < m_node_size; ++node) {
			m_gamma(node) = vec_gamma[node];
			m_beta(node) = vec_beta[node];
		}
#endif

//		m_gamma -= m_dgamma * learning_rate;
//		m_beta -= m_dbeta * learning_rate;

		// clear
//		m_dgamma *= 0;
//		m_dbeta *= 0;
	}

};

}
