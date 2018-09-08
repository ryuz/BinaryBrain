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
class NeuralNetBatchNormalization : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;
	typedef Eigen::Matrix<T, 1, -1>						Vector;

	INDEX		m_frame_size = 1;
	INDEX		m_node_size = 0;
	
	Vector		m_gamma;
	Vector		m_beta;
	Vector		m_dgamma;
	Vector		m_dbeta;

	Matrix		m_xn;
	Matrix		m_xc;
	Vector		m_std;

public:
	NeuralNetBatchNormalization() {}

	NeuralNetBatchNormalization(INDEX node_size)
	{
		Resize(node_size);
	}

	~NeuralNetBatchNormalization() {}		// デストラクタ

	void Resize(INDEX node_size)
	{
		m_node_size = node_size;
		m_gamma = Vector::Ones(m_node_size);
		m_beta = Vector::Zero(m_node_size);
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
		Eigen::Map<Matrix> x((T*)m_input_value_buffer.GetBuffer(), m_input_value_buffer.GetFrameStride() / sizeof(T), m_node_size);
		Eigen::Map<Matrix> y((T*)m_output_value_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);

		Vector mu = x.colwise().mean();
	//	std::cout << "mu =\n" << mu << std::endl;

		Matrix xc = x.array().rowwise() - mu.array();
	//	std::cout << "xc =\n" << xc << std::endl;

		Vector var = (xc.array() * xc.array()).colwise().mean();
	//	std::cout << "var =\n" << var << std::endl;

		Vector std = (var.array() + (T)10e-7).array().sqrt();
	//	std::cout << "std =\n" << std << std::endl;

		Matrix xn = xc.array().rowwise() / std.array();
	//	std::cout << "xn =\n" << xn << std::endl;

		y = (xn.array().rowwise() * m_gamma.array()).array().rowwise() + m_beta.array();
	//	std::cout << "y =\n" << y << std::endl;

		m_xn = xn;
		m_xc = xc;
		m_std = std;
	}

	void Backward(void)
	{
		Eigen::Map<Matrix> y((T*)m_output_value_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);
		Eigen::Map<Matrix> dy((T*)m_output_error_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);
		Eigen::Map<Matrix> dx((T*)m_input_error_buffer.GetBuffer(), m_output_value_buffer.GetFrameStride() / sizeof(T), m_node_size);
		INDEX frame_szie = GetOutputFrameSize();

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

		Vector dvar = m_std.array().inverse() * dstd.array() * (T)0.5;
//		std::cout << "dvar =\n" << dvar << std::endl;

		dxc = dxc.array() + (m_xc.array().rowwise() * dvar.array() * ((T)2.0 / (T)frame_szie)).array();
//		std::cout << "dxc =\n" << dxc << std::endl;

		Vector dmu = dxc.colwise().sum();
//		std::cout << "dmu =\n" << dmu << std::endl;

		dx = dxc.array().rowwise() - (dmu.array() / (T)frame_szie);
//		std::cout << "dx =\n" << dx << std::endl;
		
		m_dgamma = dgamma;
		m_dbeta = dbeta;
	}

	void Update(double learning_rate)
	{
		m_gamma -= m_dgamma * learning_rate;
		m_beta -= m_dbeta * learning_rate;
	}

};

}
