


#pragma once

#include <random>


#include "NeuralNetBufferAccessorBinary.h"
#include "NeuralNetBufferAccessorReal.h"


// 実数<->バイナリ変換
template <typename T=float, typename INDEX=size_t>
class NeuralNetRealBinaryConverter : public NeuralNetLayer<INDEX>
{
protected:
	std::mt19937_64		m_mt;
	
	INDEX				m_node_size;
	INDEX				m_frame_size;
	INDEX				m_mux_size;

	T					m_real_range_lo = (T)0.0;
	T					m_real_range_hi = (T)1.0;
	
public:
	NeuralNetRealBinaryConverter() {}
	
	NeuralNetRealBinaryConverter(INDEX node_size, INDEX mux_size, INDEX frame_size, std::uint64_t seed=1)
	{
		Setup(node_size, mux_size, frame_size, std::uint64_t seed);
	}
	
	~NeuralNetRealBinaryConverter() {}		// デストラクタ
	
	void Setup(INDEX node_size, INDEX mux_size, INDEX frame_size, std::uint64_t seed = 1)
	{
		m_node_size = node_size;
		m_mux_size = mux_size;
		m_frame_size = frame_size;
		m_mt.seed(seed);
	}

	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetFrameSize(void) const { return m_frame_size; }
	INDEX GetNodeSize(void) const { return m_node_size; }
	INDEX GetMuxSize(void) const { return m_mux_size; }


	void RealToBinary(const void* real_buf, void *binary_buf)
	{
		NeuralNetBufferAccessorReal<T, INDEX>		accReal((void*)real_buf, m_frame_size);
		NeuralNetBufferAccessorBinary<T, INDEX>		accBin(binary_buf, m_frame_size*m_mux_size);
		std::uniform_real_distribution<T>			rand(m_real_range_lo, m_real_range_hi);

		for (INDEX node = 0; node < m_node_size; node++) {
			for (INDEX frame = 0; frame < m_frame_size; frame++) {
				for (INDEX i = 0; i < m_mux_size; i++) {
					T		realVal = accReal.Get(frame, node);
					bool	binVal = (realVal > rand(m_mt));
					accBin.Set(frame*m_mux_size + i, node, binVal);
				}
			}
		}
	}

	void BinaryToReal(const void *binary_buf, void* real_buf)
	{
		NeuralNetBufferAccessorBinary<T, INDEX>		accBin((void*)binary_buf, m_frame_size*m_mux_size);
		NeuralNetBufferAccessorReal<T, INDEX>		accReal(real_buf, m_frame_size);
		T	reciprocal = (T)1.0 / (T)m_mux_size;
//		#pragma omp parallel for
		for (INDEX node = 0; node < m_node_size; node++) {
			for (INDEX frame = 0; frame < m_frame_size; frame++) {
				INDEX count = 0;
				for (INDEX i = 0; i < m_mux_size; i++) {
					bool binVal = accBin.Get(frame*m_mux_size + i, node);
					count += binVal ? 1 : 0;
				}
				T	realVal = (T)count * reciprocal;
				accReal.Set(frame, node, realVal);
			}
		}
	}

	void Update(double learning_rate)
	{
	}

};

