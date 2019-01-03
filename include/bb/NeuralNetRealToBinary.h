// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>

#include "bb/NeuralNetLayerBuf.h"


namespace bb {


// NeuralNetの抽象クラス
template <typename BT = Bit, typename T = float>
class NeuralNetRealToBinary : public NeuralNetLayerBuf<T>
{
protected:
	std::mt19937_64		m_mt;

	INDEX				m_input_node_size = 0;
	INDEX				m_output_node_size = 0;
	INDEX				m_batch_size = 1;
	INDEX				m_mux_size = 1;

	T					m_real_range_lo = (T)0.0;
	T					m_real_range_hi = (T)1.0;

public:
	NeuralNetRealToBinary() {}

	NeuralNetRealToBinary(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1)
	{
		Resize(input_node_size, output_node_size);
		InitializeCoeff(seed);
	}

	~NeuralNetRealToBinary() {}

	std::string GetClassName(void) const { return "NeuralNetRealToBinary"; }

	void Resize(INDEX input_node_size, INDEX output_node_size)
	{
		m_input_node_size = input_node_size;
		m_output_node_size = output_node_size;
	}

	void  InitializeCoeff(std::uint64_t seed)
	{
		m_mt.seed(seed);
	}

	void  SetMuxSize(INDEX mux_size) { m_mux_size = mux_size; }
	INDEX GetMuxSize(void) { return m_mux_size; }
	void  SetBatchSize(INDEX batch_size) { m_batch_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_batch_size; }
	INDEX GetInputNodeSize(void) const { return m_input_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_batch_size * m_mux_size; }
	INDEX GetOutputNodeSize(void) const { return m_output_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<BT>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	void Forward(bool train = true)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();

		// ノード数の倍率を確認
		int mul = std::max((int)(m_output_node_size / m_input_node_size), 1);
		T th_range = (T)(1.0 / mul);

		std::uniform_real_distribution<T>	dist_rand(0, th_range);

		INDEX node_size = std::max(m_input_node_size, m_output_node_size);
		for (INDEX frame = 0; frame < m_batch_size; frame++) {
			std::vector<T>		vec_v(m_output_node_size, (T)0.0);
			std::vector<int>	vec_n(m_output_node_size, 0);
			for (INDEX node = 0; node < node_size; node++) {
				vec_v[node % m_output_node_size] += in_sig_buf.template Get<T>(frame, node % m_input_node_size);
				vec_n[node % m_output_node_size] += 1;
			}

			for (INDEX node = 0; node < m_output_node_size; node++) {
				INDEX th_step   = node / m_input_node_size;
				T     th_offset = (T)(th_range * th_step);

				T rand_th = dist_rand(m_mt);

				T		real_sig = vec_v[node] / (T)vec_n[node];
				Binary	bin_sig = (real_sig > (rand_th + th_offset));
				for (INDEX i = 0; i < m_mux_size; i++) {
					out_sig_buf.template Set<BT>(frame*m_mux_size + i, node, bin_sig);
				}
			}
		}
	}

	void Backward(void)
	{
		auto in_err_buf = this->GetInputErrorBuffer();
		auto out_err_buf = this->GetOutputErrorBuffer();

		in_err_buf.Clear();
		for (INDEX node = 0; node < m_output_node_size; node++) {
			for (INDEX frame = 0; frame < m_batch_size; ++frame) {
				auto err = in_err_buf.template Get<T>(frame, node % m_input_node_size);
				for (INDEX i = 0; i < m_mux_size; i++) {
					err += out_err_buf.template Get<T>(frame*m_mux_size + i, node);
				}
				in_err_buf.template Set<T>(frame, node % m_input_node_size, err);
			}
		}
	}

	void Update(void)
	{
	}

};

}