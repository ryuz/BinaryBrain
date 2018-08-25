// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <array>
#include <vector>
#include <intrin.h>
#include "NeuralNetBinaryLut.h"


namespace bb {


// 6入力LUT固定
template <bool feedback_bitwise = false, typename T = float, typename INDEX = size_t>
class NeuralNetBinaryLut6 : public NeuralNetBinaryLut<feedback_bitwise, T, INDEX>
{
protected:
	struct LutNode {
		std::int8_t				table[64];
		std::array< INDEX, 6 >	input;
	};

	std::vector<LutNode>	m_lut;

public:
	NeuralNetBinaryLut6() {}

	NeuralNetBinaryLut6(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1)
	{
		Resize(input_node_size, output_node_size);
		InitializeCoeff(seed);
	}

	~NeuralNetBinaryLut6() {}		// デストラクタ

	void Resize(INDEX input_node_size, INDEX output_node_size)
	{
		NeuralNetBinaryLut<feedback_bitwise, T, INDEX>::Resize(input_node_size, output_node_size);
		
		m_lut.resize(m_output_node_size);
	}

	void Resize(std::vector<INDEX> sizes)
	{
		BB_ASSERT(sizes.size() == 2);
		Resize(sizes[1], sizes[0]);
	}

	int   GetLutInputSize(void) const { return 6; }
	int   GetLutTableSize(void) const { return (1 << 6); }
	void  SetLutInput(INDEX node, int input_index, INDEX input_node) { m_lut[node].input[input_index] = input_node; }
	INDEX GetLutInput(INDEX node, int input_index) const { return m_lut[node].input[input_index]; };
	void  SetLutTable(INDEX node, int bit, bool value) { m_lut[node].table[bit] = value ? -1 : 0; }
	bool  GetLutTable(INDEX node, int bit) const { return (m_lut[node].table[bit] != 0); }



protected:
	inline __m256i my_andnot_si256(__m256i& val, __m256i& lut)
	{
#ifdef __AVX2__
		return _mm256_andnot_si256(val, lut);
#else
		__m256 res = _mm256_andnot_ps(*(__m256 *)&val, *(__m256 *)&lut);
		return *(__m256i *)&res;
#endif
	}

	inline __m256i my_and_si256(__m256i& val, __m256i& lut)
	{
#ifdef __AVX2__
		return _mm256_and_si256(val, lut);
#else
		__m256 res = _mm256_and_ps(*(__m256 *)&val, *(__m256 *)&lut);
		return *(__m256i *)&res;
#endif
	}

	inline __m256i my_or_si256(__m256i& val, __m256i& lut)
	{
#ifdef __AVX2__
		return _mm256_or_si256(val, lut);
#else
		__m256 res = _mm256_or_ps(*(__m256 *)&val, *(__m256 *)&lut);
		return *(__m256i *)&res;
#endif
	}

	template<int LUT, int VAL>
	inline __m256i lut_mask_unit(__m256i& val, __m256i& lut)
	{
		if ((LUT & (1 << VAL)) == 0) {
			return my_andnot_si256(val, lut);
		}
		else {
			return my_and_si256(val, lut);
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
		msk = my_or_si256(msk, lut);
	}

	void ForwardNode(INDEX node) {
		INDEX frame_size = (m_frame_size + 255) / 256;

		auto in_buf = GetInputValueBuffer();
		auto out_buf = GetOutputValueBuffer();

		auto& lut = m_lut[node];

		__m256i*	in_ptr[6];
		__m256i*	out_ptr;
		__m256i		in_val[6];

		in_ptr[0] = (__m256i*)in_buf.GetPtr(lut.input[0]);
		in_ptr[1] = (__m256i*)in_buf.GetPtr(lut.input[1]);
		in_ptr[2] = (__m256i*)in_buf.GetPtr(lut.input[2]);
		in_ptr[3] = (__m256i*)in_buf.GetPtr(lut.input[3]);
		in_ptr[4] = (__m256i*)in_buf.GetPtr(lut.input[4]);
		in_ptr[5] = (__m256i*)in_buf.GetPtr(lut.input[5]);
		out_ptr = (__m256i*)out_buf.GetPtr(node);

		for (int i = 0; i < frame_size; i++) {
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
	void Forward(void)
	{
		int node_size = (int)m_output_node_size;
#pragma omp parallel for
		for (int node = 0; node < node_size; ++node) {
			ForwardNode(node);
		}
	}

	void Backward(void)
	{
	}

	void Update(double learning_rate)
	{
	}

};


}