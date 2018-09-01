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


// 実数<->バイナリ変換
template <typename T = float, typename INDEX = size_t>
class NeuralNetRealBinaryConverter : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	std::mt19937_64		m_mt;

	INDEX				m_binary_node_size = 0;
	INDEX				m_real_node_size = 0;
	INDEX				m_batch_size = 1;
	INDEX				m_mux_size = 1;

	T					m_real_range_lo = (T)0.0;
	T					m_real_range_hi = (T)1.0;
//	T					m_real_range_lo = (T)0.2;
//	T					m_real_range_hi = (T)0.8;

public:
	NeuralNetRealBinaryConverter() {}

	NeuralNetRealBinaryConverter(INDEX real_node_size, INDEX binary_node_size, std::uint64_t seed = 1)
	{
		Resize(real_node_size, binary_node_size);
		InitializeCoeff(seed);
	}

	~NeuralNetRealBinaryConverter() {}		// デストラクタ

	void Resize(INDEX real_node_size, INDEX binary_node_size)
	{
		m_binary_node_size = binary_node_size;
		m_real_node_size = real_node_size;
	}

	void  InitializeCoeff(std::uint64_t seed)
	{
		m_mt.seed(seed);
	}

	INDEX GetRealFrameSize(void) const { return m_batch_size; }
	INDEX GetBinaryFrameSize(void) const { return m_batch_size * m_mux_size; }
	INDEX GetBinaryNodeSize(void) const { return m_binary_node_size; }
	INDEX GetRealNodeSize(void) const { return m_real_node_size; }
	INDEX GetMuxSize(void) const { return m_mux_size; }

	void  SetMuxSize(INDEX mux_size) { m_mux_size = mux_size; }
	INDEX GetMuxSize(void) { return m_mux_size; }

	void  SetBatchSize(INDEX batch_size) { m_batch_size = batch_size; }

	void RealToBinary(NeuralNetBuffer<T, INDEX> real_buf, NeuralNetBuffer<T, INDEX> binary_buf)
	{
		std::uniform_real_distribution<T>	rand(m_real_range_lo, m_real_range_hi);

		INDEX node_size = std::max(m_real_node_size, m_binary_node_size);
		for (INDEX frame = 0; frame < m_batch_size; frame++) {
			std::vector<T>		vec_v(m_binary_node_size, (T)0.0);
			std::vector<int>	vec_n(m_binary_node_size, 0);
			for (INDEX node = 0; node < node_size; node++) {
				vec_v[node % m_binary_node_size] += real_buf.Get<T>(frame, node % m_real_node_size);
				vec_n[node % m_binary_node_size] += 1;
			}

			for (INDEX node = 0; node < m_binary_node_size; node++) {
				T		realVal = vec_v[node] / (T)vec_n[node];
				bool	binVal  = (realVal > rand(m_mt));
				for (INDEX i = 0; i < m_mux_size; i++) {
					binary_buf.Set<bool>(frame*m_mux_size + i, node, binVal);
				}
			}
		}
	}

	void BinaryToReal(NeuralNetBuffer<T, INDEX> binary_buf, NeuralNetBuffer<T, INDEX> real_buf)
	{
		T	reciprocal = (T)1.0 / (T)m_mux_size;

		INDEX node_size = std::max(m_real_node_size, m_binary_node_size);

//		#pragma omp parallel for
		std::vector<int>	vec_v(m_real_node_size, 0);
		std::vector<int>	vec_n(m_real_node_size, 0);
		for (INDEX frame = 0; frame < m_batch_size; frame++) {
			std::fill(vec_v.begin(), vec_v.end(), 0);
			std::fill(vec_n.begin(), vec_n.end(), 0);
			for (INDEX node = 0; node < node_size; node++) {
				for (INDEX i = 0; i < m_mux_size; i++) {
					bool binVal = binary_buf.Get<bool>(frame*m_mux_size + i, node);
					vec_v[node %m_real_node_size] += binVal ? 1 : 0;
					vec_n[node %m_real_node_size] += 1;
				}
			}

			for (INDEX node = 0; node < m_real_node_size; node++) {
				real_buf.Set<T>(frame, node, (T)vec_v[node] / vec_n[node]);
			}
		}
	}
	
	void RealToRealDemux(NeuralNetBuffer<T, INDEX> src_buf, NeuralNetBuffer<T, INDEX> dst_buf)
	{
		INDEX src_node_size = src_buf.GetNodeSize();
		INDEX dst_node_size = dst_buf.GetNodeSize();
		INDEX node_size = std::max(src_node_size, dst_node_size);
		std::vector<int>	vec_v(dst_node_size, 0);
		std::vector<int>	vec_n(dst_node_size, 0);
		for (INDEX frame = 0; frame < m_batch_size; frame++) {
			std::fill(vec_v.begin(), vec_v.end(), 0);
			std::fill(vec_n.begin(), vec_n.end(), 0);
			for (INDEX node = 0; node < node_size; node++) {
				for (INDEX i = 0; i < m_mux_size; i++) {
					bool binVal = binary_buf.Get<bool>(frame*m_mux_size + i, node);
					vec_v[node %m_real_node_size] += binVal ? 1 : 0;
					vec_n[node %m_real_node_size] += 1;
				}
			}

			for (INDEX node = 0; node < m_real_node_size; node++) {
				real_buf.Set<T>(frame, node, (T)vec_v[node] / vec_n[node]);
			}
		}
	}


	void Update(double learning_rate)
	{
	}

};


}
