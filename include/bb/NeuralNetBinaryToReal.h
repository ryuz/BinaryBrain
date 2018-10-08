// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>
#include "NeuralNetLayerBuf.h"


namespace bb {


// NeuralNetの抽象クラス
template <typename BT = Bit, typename T = float, typename INDEX = size_t>
class NeuralNetBinaryToReal : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	INDEX				m_input_node_size = 0;
	INDEX				m_output_node_size = 0;
	INDEX				m_batch_size = 1;
	INDEX				m_mux_size = 1;
	
public:
	NeuralNetBinaryToReal() {}

	NeuralNetBinaryToReal(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1)
	{
		Resize(input_node_size, output_node_size);
		InitializeCoeff(seed);
	}
	
	~NeuralNetBinaryToReal() {}

	std::string GetClassName(void) const { return "NeuralNetBinaryToReal"; }

	void Resize(INDEX input_node_size, INDEX output_node_size)
	{
		m_input_node_size = input_node_size;
		m_output_node_size = output_node_size;
	}

	void  InitializeCoeff(std::uint64_t seed)
	{
	}

	void  SetMuxSize(INDEX mux_size) { m_mux_size = mux_size; }
	INDEX GetMuxSize(void) { return m_mux_size; }
	void  SetBatchSize(INDEX batch_size) { m_batch_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_batch_size * m_mux_size; }
	INDEX GetInputNodeSize(void) const { return m_input_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_batch_size; }
	INDEX GetOutputNodeSize(void) const { return m_output_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<BT>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }
	
	
	void Forward(bool train = true)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();

//		T	reciprocal = (T)1.0 / (T)m_mux_size;

		INDEX node_size = std::max(m_input_node_size, m_output_node_size);

		std::vector<T>		vec_v(m_output_node_size, (T)0);
		std::vector<int>	vec_n(m_output_node_size, 0);
		for (INDEX frame = 0; frame < m_batch_size; frame++) {
			std::fill(vec_v.begin(), vec_v.end(), (T)0);
			std::fill(vec_n.begin(), vec_n.end(), 0);
			for (INDEX node = 0; node < node_size; node++) {
				for (INDEX i = 0; i < m_mux_size; i++) {
					BT bin_sig = in_sig_buf.template Get<BT>(frame*m_mux_size + i, node);
					vec_v[node % m_output_node_size] += bin_sig;
					vec_n[node % m_output_node_size] += 1;
				}
			}

			for (INDEX node = 0; node < m_output_node_size; node++) {
				out_sig_buf.template Set<T>(frame, node, (T)vec_v[node] / vec_n[node]);
			}
		}
	}
	
	void Backward(void)
	{
		auto in_err = this->GetInputErrorBuffer();
		auto out_err = this->GetOutputErrorBuffer();

		for (INDEX node = 0; node < m_input_node_size; node++) {
			for (INDEX frame = 0; frame < m_batch_size; ++frame) {
				for (INDEX i = 0; i < m_mux_size; i++) {
					in_err.template Set<T>(frame*m_mux_size + i, node, out_err.template Get<T>(frame, node % m_output_node_size));
				}
			}
		}
	}

	void Update(void)
	{
	}

};

}