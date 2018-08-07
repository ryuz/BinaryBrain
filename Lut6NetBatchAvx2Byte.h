


#pragma once

#include <array>
#include <vector>
#include <intrin.h>
#include "BinaryNetBatch.h"


// 6入力LUT固定(バッチ版)、データバイト配置 AVX2命令利用版
class Lut6NetBatchAvx2Byte : public BinaryNetBatch
{
protected:
	struct LutNode {
		std::array< int, 6 >		input;
		std::uint32_t				table_lo;
		std::uint32_t				table_hi;
	};

	int										m_batch_size_raw = 0;
	int										m_batch_size     = 0;
	std::vector< std::vector<LutNode> >		m_lut;
	std::vector<std::uint64_t*>				m_value;

	inline std::int8_t& at(int frame, int layer, int node) const
	{
		auto ptr = (std::int8_t*)m_value[layer];

		return ptr[node * (m_batch_size*8) + frame];
	}


	inline void CalcForwardUnit(int layer, int node) {
		auto& lut = m_lut[layer][node];

		__m256i		lut_lo = _mm256_set1_epi32(lut.table_lo);
		__m256i		lut_hi = _mm256_set1_epi32(lut.table_hi);
		uint64_t*	in_ptr0 = &m_value[layer - 1][lut.input[0] * m_batch_size];
		uint64_t*	in_ptr1 = &m_value[layer - 1][lut.input[1] * m_batch_size];
		uint64_t*	in_ptr2 = &m_value[layer - 1][lut.input[2] * m_batch_size];
		uint64_t*	in_ptr3 = &m_value[layer - 1][lut.input[3] * m_batch_size];
		uint64_t*	in_ptr4 = &m_value[layer - 1][lut.input[4] * m_batch_size];
		uint64_t*	in_ptr5 = &m_value[layer - 1][lut.input[5] * m_batch_size];
		uint64_t*	out_ptr = &m_value[layer][node * m_batch_size];

		__m128i in5_mem = _mm_set1_epi64x(0);
		__m128i in4_mem = _mm_set1_epi64x(0);
		__m128i in3_mem = _mm_set1_epi64x(0);
		__m128i in2_mem = _mm_set1_epi64x(0);
		__m128i in1_mem = _mm_set1_epi64x(0);
		__m128i in0_mem = _mm_set1_epi64x(0);
		__m256i in4_msk;
		__m256i in3_msk;
		__m256i in2_msk;
		__m256i in1_msk;
		__m256i in0_msk;
		__m256i in5_val;
		__m256i in4_val;
		__m256i in3_val;
		__m256i in2_val;
		__m256i in1_val;
		__m256i in0_val;
		__m256i lut_val;
		__m256i out_val;

		for (int i = 0; i < m_batch_size; i++) {
			in5_mem.m128i_u64[0] = in_ptr5[i];
			in5_val = _mm256_cvtepi8_epi32(in5_mem);
			lut_val = _mm256_blendv_epi8(lut_lo, lut_hi, in5_val);

			in4_mem.m128i_u64[0] = in_ptr4[i];
			in4_val = _mm256_cvtepi8_epi32(in4_mem);
			in4_msk = _mm256_xor_si256(in4_val, _mm256_set1_epi32(0x0000ffff));
			lut_val = _mm256_and_si256(lut_val, in4_msk);

			in3_mem.m128i_u64[0] = in_ptr3[i];
			in3_val = _mm256_cvtepi8_epi32(in3_mem);
			in3_msk = _mm256_xor_si256(in3_val, _mm256_set1_epi32(0x00ff00ff));
			lut_val = _mm256_and_si256(lut_val, in3_msk);

			in2_mem.m128i_u64[0] = in_ptr2[i];
			in2_val = _mm256_cvtepi8_epi32(in2_mem);
			in2_msk = _mm256_xor_si256(in2_val, _mm256_set1_epi32(0x0f0f0f0f));
			lut_val = _mm256_and_si256(lut_val, in2_msk);

			in1_mem.m128i_u64[0] = in_ptr1[i];
			in1_val = _mm256_cvtepi8_epi32(in1_mem);
			in1_msk = _mm256_xor_si256(in1_val, _mm256_set1_epi32(0x33333333));
			lut_val = _mm256_and_si256(lut_val, in1_msk);

			in0_mem.m128i_u64[0] = in_ptr0[i];
			in0_val = _mm256_cvtepi8_epi32(in0_mem);
			in0_msk = _mm256_xor_si256(in0_val, _mm256_set1_epi32(0x55555555));
			lut_val = _mm256_and_si256(lut_val, in0_msk);

			out_val = _mm256_cmpeq_epi32(lut_val, _mm256_set1_epi32(0x00000000));
			out_val = _mm256_andnot_si256(out_val, _mm256_set1_epi32(0xffffffff));

			out_val = _mm256_packs_epi32(out_val, out_val);
			out_val = _mm256_packs_epi16(out_val, out_val);
			out_val.m256i_u32[1] = out_val.m256i_u32[4];
			out_ptr[i] = out_val.m256i_u64[0];
		}
	}


public:
	Lut6NetBatchAvx2Byte()
	{
	}

	Lut6NetBatchAvx2Byte(std::vector<int> vec_layer_size)
	{
		Setup(vec_layer_size);
	}

	~Lut6NetBatchAvx2Byte()
	{
		for (uint64_t* v : m_value) {
			_mm_free(v);
		}
	}

	void Setup(std::vector<int> vec_layer_size)
	{
		int layer_num = (int)vec_layer_size.size();
		m_lut.resize(layer_num);
		for (int i = 0; i < layer_num; i++) {
			m_lut[i].resize(vec_layer_size[i]);
		}
	}
	
	int  GetLayerNum(void) const
	{
		return (int)m_lut.size();
	}

	int  GetNodeNum(int layer) const
	{
		return (int)m_lut[layer].size();
	}

	int  GetInputNum(int layer, int node) const
	{
		return 6;
	}

	void SetConnection(int layer, int node, int input_index, int input_node)
	{
		m_lut[layer][node].input[input_index] = input_node;
	}

	int GetConnection(int layer, int node, int input_index) const
	{
		return m_lut[layer][node].input[input_index];
	}
	
	void SetLutBit(int layer, int node, int bit, bool value)
	{
		if (bit < 32) {
			if (value) {
				m_lut[layer][node].table_lo |= (1 << bit);
			}
			else {
				m_lut[layer][node].table_lo &= ~(1 << bit);
			}
		}
		else {
			bit -= 32;
			if (value) {
				m_lut[layer][node].table_hi |= (1 << bit);
			}
			else {
				m_lut[layer][node].table_hi &= ~(1 << bit);
			}
		}
	}

	bool GetLutBit(int layer, int node, int bit) const
	{
		if (bit < 32) {
			return (((m_lut[layer][node].table_lo >> bit) & 1) != 0);
		}
		else {
			bit -= 32;
			return (((m_lut[layer][node].table_hi >> bit) & 1) != 0);
		}
	}

	void SetBatchSize(int batch_size)
	{
		// 既存メモリ開放
		for (uint64_t* v : m_value) {
			_mm_free(v);
		}

		// メモリ確保
		m_batch_size_raw = batch_size;
		m_batch_size = (batch_size + 7) / 8;

		int layer_num = GetLayerNum();
		m_value.resize(layer_num);
		for (int layer = 0; layer < layer_num; layer++) {
			int node_num = GetNodeNum(layer);
			m_value[layer] = (uint64_t*)_mm_malloc(sizeof(uint64_t) * node_num * m_batch_size, 32);
		}
	}
	
	int  GetBatchSize(void)
	{
		return m_batch_size_raw;
	}

	bool GetValue(int frame, int layer, int node) const
	{
		return (at(frame, layer, node) != 0);
	}
	
	void SetValue(int frame, int layer, int node, bool value)
	{
		at(frame, layer, node) = value ? -1 : 0;
	}

	bool GetInputValue(int frame, int layer, int node, int index) const
	{
		return (at(frame, layer - 1, m_lut[layer][node].input[index]) != 0);
	}

	void CalcForward(int start_layer = 0)
	{
		int layer_num = GetLayerNum();
		for (int layer = start_layer + 1; layer < layer_num; layer++) {
			int node_num = GetNodeNum(layer);
#pragma omp parallel for
			for (int node = 0; node < node_num; node++) {
				CalcForwardUnit(layer, node);
//				for (int frame = 0; frame < m_batch_size; frame++) {
//					at(frame, layer, node) = m_lut[layer][node].table[GetInputLutIndex(frame, layer, node)];
//				}
			}
		}
	}
};

