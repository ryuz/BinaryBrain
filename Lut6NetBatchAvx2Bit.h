


#pragma once

#include <array>
#include <vector>
#include <intrin.h>
#include "BinaryNetBatch.h"


// 6入力LUT固定(バッチ版)、データビット配置 AVX2命令利用版
class Lut6NetBatchAvx2Bit : public BinaryNetBatch
{
protected:
	struct LutNode {
		std::int8_t					table[64];
		std::array< int, 6 >		input;
	};

	int										m_batch_size_raw = 0;
	int										m_batch_size     = 0;
	std::vector< std::vector<LutNode> >		m_lut;
	std::vector<__m256i*>					m_value;

	
	inline void set_val(int frame, int layer, int node, bool val)
	{
		auto ptr = (std::int8_t*)m_value[layer];
		int bit = (1 << (frame % 8));
		if (val) {
			ptr[node * (m_batch_size * 32) + (frame / 8)] |= bit;
		}
		else {
			ptr[node * (m_batch_size * 32) + (frame / 8)] &= ~bit;
		}
	}

	inline bool get_val(int frame, int layer, int node) const
	{
		auto ptr = (std::int8_t*)m_value[layer];
		int bit = (1 << (frame % 8));
		return ((ptr[node * (m_batch_size * 32) + (frame / 8)] & bit) != 0);
	}


	template<int LUT, int VAL>
	inline __m256i lut_mask_unit(__m256i& val, __m256i& lut)
	{
		if ((LUT & (1 << VAL)) == 0 ) {
			return _mm256_andnot_si256(val, lut);
		}
		else {
			return _mm256_and_si256(val, lut);
		}
	}

	template<int LUT>
	inline void lut_mask(__m256i& msk, __m256i& lut, __m256i val[6])
	{
		lut = lut_mask_unit<LUT, 0>(val[0], lut);
		lut = lut_mask_unit<LUT, 1>(val[1], lut);
		lut = lut_mask_unit<LUT, 2>(val[2], lut);
		lut = lut_mask_unit<LUT, 3>(val[3], lut);
		lut = lut_mask_unit<LUT, 4>(val[4], lut);
		lut = lut_mask_unit<LUT, 5>(val[5], lut);
		msk = _mm256_or_si256(msk, lut);
	}

	inline void CalcForwardUnit(int layer, int node) {
		auto& lut = m_lut[layer][node];

		__m256i*	in_ptr[6];
		__m256i*	out_ptr;
		__m256i		in_val[6];

		in_ptr[0] = &m_value[layer - 1][lut.input[0] * m_batch_size];
		in_ptr[1] = &m_value[layer - 1][lut.input[1] * m_batch_size];
		in_ptr[2] = &m_value[layer - 1][lut.input[2] * m_batch_size];
		in_ptr[3] = &m_value[layer - 1][lut.input[3] * m_batch_size];
		in_ptr[4] = &m_value[layer - 1][lut.input[4] * m_batch_size];
		in_ptr[5] = &m_value[layer - 1][lut.input[5] * m_batch_size];
		out_ptr = &m_value[layer][node * m_batch_size];
		
		for (int i = 0; i < m_batch_size; i++) {
			// input
			in_val[0] = _mm256_loadu_si256(&in_ptr[0][i]);
			in_val[1] = _mm256_loadu_si256(&in_ptr[1][i]);
			in_val[2] = _mm256_loadu_si256(&in_ptr[2][i]);
			in_val[3] = _mm256_loadu_si256(&in_ptr[3][i]);
			in_val[4] = _mm256_loadu_si256(&in_ptr[4][i]);
			in_val[5] = _mm256_loadu_si256(&in_ptr[5][i]);

			// LUT
			__m256i msk = _mm256_set1_epi8(0);
			lut_mask<0>(msk, _mm256_set1_epi8(lut.table[0]), in_val);
			lut_mask<1>(msk, _mm256_set1_epi8(lut.table[1]), in_val);
			lut_mask<2>(msk, _mm256_set1_epi8(lut.table[2]), in_val);
			lut_mask<3>(msk, _mm256_set1_epi8(lut.table[3]), in_val);
			lut_mask<4>(msk, _mm256_set1_epi8(lut.table[4]), in_val);
			lut_mask<5>(msk, _mm256_set1_epi8(lut.table[5]), in_val);
			lut_mask<6>(msk, _mm256_set1_epi8(lut.table[6]), in_val);
			lut_mask<7>(msk, _mm256_set1_epi8(lut.table[7]), in_val);
			lut_mask<8>(msk, _mm256_set1_epi8(lut.table[8]), in_val);
			lut_mask<9>(msk, _mm256_set1_epi8(lut.table[9]), in_val);
			lut_mask<10>(msk, _mm256_set1_epi8(lut.table[10]), in_val);
			lut_mask<11>(msk, _mm256_set1_epi8(lut.table[11]), in_val);
			lut_mask<12>(msk, _mm256_set1_epi8(lut.table[12]), in_val);
			lut_mask<13>(msk, _mm256_set1_epi8(lut.table[13]), in_val);
			lut_mask<14>(msk, _mm256_set1_epi8(lut.table[14]), in_val);
			lut_mask<15>(msk, _mm256_set1_epi8(lut.table[15]), in_val);
			lut_mask<16>(msk, _mm256_set1_epi8(lut.table[16]), in_val);
			lut_mask<17>(msk, _mm256_set1_epi8(lut.table[17]), in_val);
			lut_mask<18>(msk, _mm256_set1_epi8(lut.table[18]), in_val);
			lut_mask<19>(msk, _mm256_set1_epi8(lut.table[19]), in_val);
			lut_mask<20>(msk, _mm256_set1_epi8(lut.table[20]), in_val);
			lut_mask<21>(msk, _mm256_set1_epi8(lut.table[21]), in_val);
			lut_mask<22>(msk, _mm256_set1_epi8(lut.table[22]), in_val);
			lut_mask<23>(msk, _mm256_set1_epi8(lut.table[23]), in_val);
			lut_mask<24>(msk, _mm256_set1_epi8(lut.table[24]), in_val);
			lut_mask<25>(msk, _mm256_set1_epi8(lut.table[25]), in_val);
			lut_mask<26>(msk, _mm256_set1_epi8(lut.table[26]), in_val);
			lut_mask<27>(msk, _mm256_set1_epi8(lut.table[27]), in_val);
			lut_mask<28>(msk, _mm256_set1_epi8(lut.table[28]), in_val);
			lut_mask<29>(msk, _mm256_set1_epi8(lut.table[29]), in_val);
			lut_mask<30>(msk, _mm256_set1_epi8(lut.table[30]), in_val);
			lut_mask<31>(msk, _mm256_set1_epi8(lut.table[31]), in_val);
			lut_mask<32>(msk, _mm256_set1_epi8(lut.table[32]), in_val);
			lut_mask<33>(msk, _mm256_set1_epi8(lut.table[33]), in_val);
			lut_mask<34>(msk, _mm256_set1_epi8(lut.table[34]), in_val);
			lut_mask<35>(msk, _mm256_set1_epi8(lut.table[35]), in_val);
			lut_mask<36>(msk, _mm256_set1_epi8(lut.table[36]), in_val);
			lut_mask<37>(msk, _mm256_set1_epi8(lut.table[37]), in_val);
			lut_mask<38>(msk, _mm256_set1_epi8(lut.table[38]), in_val);
			lut_mask<39>(msk, _mm256_set1_epi8(lut.table[39]), in_val);
			lut_mask<40>(msk, _mm256_set1_epi8(lut.table[40]), in_val);
			lut_mask<41>(msk, _mm256_set1_epi8(lut.table[41]), in_val);
			lut_mask<42>(msk, _mm256_set1_epi8(lut.table[42]), in_val);
			lut_mask<43>(msk, _mm256_set1_epi8(lut.table[43]), in_val);
			lut_mask<44>(msk, _mm256_set1_epi8(lut.table[44]), in_val);
			lut_mask<45>(msk, _mm256_set1_epi8(lut.table[45]), in_val);
			lut_mask<46>(msk, _mm256_set1_epi8(lut.table[46]), in_val);
			lut_mask<47>(msk, _mm256_set1_epi8(lut.table[47]), in_val);
			lut_mask<48>(msk, _mm256_set1_epi8(lut.table[48]), in_val);
			lut_mask<49>(msk, _mm256_set1_epi8(lut.table[49]), in_val);
			lut_mask<50>(msk, _mm256_set1_epi8(lut.table[50]), in_val);
			lut_mask<51>(msk, _mm256_set1_epi8(lut.table[51]), in_val);
			lut_mask<52>(msk, _mm256_set1_epi8(lut.table[52]), in_val);
			lut_mask<53>(msk, _mm256_set1_epi8(lut.table[53]), in_val);
			lut_mask<54>(msk, _mm256_set1_epi8(lut.table[54]), in_val);
			lut_mask<55>(msk, _mm256_set1_epi8(lut.table[55]), in_val);
			lut_mask<56>(msk, _mm256_set1_epi8(lut.table[56]), in_val);
			lut_mask<57>(msk, _mm256_set1_epi8(lut.table[57]), in_val);
			lut_mask<58>(msk, _mm256_set1_epi8(lut.table[58]), in_val);
			lut_mask<59>(msk, _mm256_set1_epi8(lut.table[59]), in_val);
			lut_mask<60>(msk, _mm256_set1_epi8(lut.table[60]), in_val);
			lut_mask<61>(msk, _mm256_set1_epi8(lut.table[61]), in_val);
			lut_mask<62>(msk, _mm256_set1_epi8(lut.table[62]), in_val);
			lut_mask<63>(msk, _mm256_set1_epi8(lut.table[63]), in_val);

			_mm256_storeu_si256(&out_ptr[i], msk);
		}
	}


public:
	Lut6NetBatchAvx2Bit()
	{
	}

	Lut6NetBatchAvx2Bit(std::vector<int> vec_layer_size)
	{
		Setup(vec_layer_size);
	}

	~Lut6NetBatchAvx2Bit()
	{
		for (__m256i* v : m_value) {
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
		m_lut[layer][node].table[bit] = value ? -1 : 0;
	}

	bool GetLutBit(int layer, int node, int bit) const
	{
		return (m_lut[layer][node].table[bit] != 0);
	}

	void SetBatchSize(int batch_size)
	{
		// 既存メモリ開放
		for (__m256i* v : m_value) {
			_mm_free(v);
		}

		// メモリ確保
		m_batch_size_raw = batch_size;
		m_batch_size = (batch_size + 255) / 256;

		int layer_num = GetLayerNum();
		m_value.resize(layer_num);
		for (int layer = 0; layer < layer_num; layer++) {
			int node_num = GetNodeNum(layer);
			m_value[layer] = (__m256i*)_mm_malloc(sizeof(__m256i) * node_num * m_batch_size, 32);
		}
	}
	
	int  GetBatchSize(void)
	{
		return m_batch_size_raw;
	}

	bool GetValue(int frame, int layer, int node) const
	{
		return get_val(frame, layer, node);
	}
	
	void SetValue(int frame, int layer, int node, bool value)
	{
		set_val(frame, layer, node, value);
	}

	bool GetInputValue(int frame, int layer, int node, int index) const
	{
		return get_val(frame, layer - 1, m_lut[layer][node].input[index]);
	}

	void CalcForward(int start_layer = 0)
	{
		int layer_num = GetLayerNum();
		for (int layer = start_layer + 1; layer < layer_num; layer++) {
			int node_num = GetNodeNum(layer);
#pragma omp parallel for
			for (int node = 0; node < node_num; node++) {
				CalcForwardUnit(layer, node);
			}
		}
	}
};

