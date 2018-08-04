


#pragma once

#include<memory>
#include <memory.h>
#include <intrin.h>


class Lut6LayerAvx2
{
public:
	int			m_size;

	const int *	m_input_vector;
	__m256i*	m_output_vector;

	__m256i*	m_lut_lo;
	__m256i*	m_lut_hi;

	__m256i*	m_input_index[6];
	
	Lut6LayerAvx2(int output_num)
	{
		m_size = (output_num + 7) / 8;
		m_input_vector = nullptr;

		m_output_vector = (__m256i*)_mm_malloc(sizeof(__m256i) * m_size, 32);
		m_lut_lo = (__m256i*)_mm_malloc(sizeof(__m256i) * m_size, 32);
		m_lut_hi = (__m256i*)_mm_malloc(sizeof(__m256i) * m_size, 32);
		m_input_index[0] = (__m256i*)_mm_malloc(sizeof(__m256i) * m_size, 32);
		m_input_index[1] = (__m256i*)_mm_malloc(sizeof(__m256i) * m_size, 32);
		m_input_index[2] = (__m256i*)_mm_malloc(sizeof(__m256i) * m_size, 32);
		m_input_index[3] = (__m256i*)_mm_malloc(sizeof(__m256i) * m_size, 32);
		m_input_index[4] = (__m256i*)_mm_malloc(sizeof(__m256i) * m_size, 32);
		m_input_index[5] = (__m256i*)_mm_malloc(sizeof(__m256i) * m_size, 32);

		memset(m_input_index[0], 0, sizeof(__m256i) * m_size);
		memset(m_input_index[1], 0, sizeof(__m256i) * m_size);
		memset(m_input_index[2], 0, sizeof(__m256i) * m_size);
		memset(m_input_index[3], 0, sizeof(__m256i) * m_size);
		memset(m_input_index[4], 0, sizeof(__m256i) * m_size);
		memset(m_input_index[5], 0, sizeof(__m256i) * m_size);
	}

	~Lut6LayerAvx2()
	{
		_mm_free(m_output_vector);
		_mm_free(m_lut_lo);
		_mm_free(m_lut_hi);
		_mm_free(m_input_index[0]);
		_mm_free(m_input_index[1]);
		_mm_free(m_input_index[2]);
		_mm_free(m_input_index[3]);
		_mm_free(m_input_index[4]);
		_mm_free(m_input_index[5]);
	}

	void SetInputVector(const int* input_vector)
	{
		m_input_vector = input_vector;
	}

	const int *GetOutputVector(void) const
	{
		return (int *)m_output_vector;
	}
	
	void SetConnection(int n, int idx, int in_inx)
	{
		int* input_index = (int*)m_input_index[idx];
		input_index[n] = in_inx;
	}
	
	bool GetValue(int n) const
	{
		int* output_vector = (int*)m_output_vector;
		return output_vector[n] != 0 ? true : false;
	}

	bool SetValue(int n, bool val)
	{
		int* output_vector = (int*)m_output_vector;
		output_vector[n] = val ? 0xffffffff : 0x00000000;
	}

	bool GetInputValue(int n, int idx) const
	{
		int* input_index = (int*)m_input_index[idx];
		return m_input_vector[input_index[n]] != 0 ? true : false;
	}
	
	int GetInputLutIndex(int n) const
	{
		int idx = 0;
		idx |= GetInputValue(n, 0) ? 0x000000001 : 0x000000000;
		idx |= GetInputValue(n, 1) ? 0x000000002 : 0x000000000;
		idx |= GetInputValue(n, 2) ? 0x000000004 : 0x000000000;
		idx |= GetInputValue(n, 3) ? 0x000000008 : 0x000000000;
		idx |= GetInputValue(n, 4) ? 0x000000010 : 0x000000000;
		idx |= GetInputValue(n, 5) ? 0x000000020 : 0x000000000;

		return idx;
	}

	bool GetLutBit(int n, int bit) const
	{
		int* lut_vector;
		if (bit >= 32) {
			bit -= 32;
			lut_vector = (int *)m_lut_hi;
		}
		else {
			lut_vector = (int *)m_lut_lo;
		}

		return ((lut_vector[n] >> bit) & 1) ? true : false;
	}

	void SetLutBit(int n, int bit, bool val)
	{
		int* lut_vector;
		if (bit >= 32) {
			bit -= 32;
			lut_vector = (int *)m_lut_hi;
		}
		else {
			lut_vector = (int *)m_lut_lo;
		}

		if (val) {
			lut_vector[n] |= (1 << bit);
		}
		else {
			lut_vector[n] &= ~(1 << bit);
		}
	}

	void SetInputConnection(int n, int idx, int in_inx)
	{
		int* input_index = (int*)m_input_index[idx];
		input_index[n] = in_inx;
	}
	
	void CalcForward(void)
	{
		for (int i = 0; i < m_size; i++) {
			__m256i in5_val = _mm256_i32gather_epi32(m_input_vector, _mm256_load_si256(&m_input_index[5][i]), 4);
			__m256i lut_lo = _mm256_load_si256(&m_lut_lo[i]);
			__m256i lut_hi = _mm256_load_si256(&m_lut_hi[i]);
			__m256i lut_val = _mm256_blendv_epi8(lut_lo, lut_hi, in5_val);

			__m256i in4_val = _mm256_i32gather_epi32(m_input_vector, _mm256_load_si256(&m_input_index[4][i]), 4);
			__m256i in4_msk = _mm256_xor_si256(in4_val, _mm256_set1_epi32(0xffff0000));
					lut_val = _mm256_and_si256(lut_val, in4_msk);

			__m256i in3_val = _mm256_i32gather_epi32(m_input_vector, _mm256_load_si256(&m_input_index[3][i]), 4);
			__m256i in3_msk = _mm256_xor_si256(in3_val, _mm256_set1_epi32(0xff00ff00));
					lut_val = _mm256_and_si256(lut_val, in3_msk);

			__m256i in2_val = _mm256_i32gather_epi32(m_input_vector, _mm256_load_si256(&m_input_index[2][i]), 4);
			__m256i in2_msk = _mm256_xor_si256(in2_val, _mm256_set1_epi32(0xf0f0f0f0));
					lut_val = _mm256_and_si256(lut_val, in2_msk);

			__m256i in1_val = _mm256_i32gather_epi32(m_input_vector, _mm256_load_si256(&m_input_index[1][i]), 4);
			__m256i in1_msk = _mm256_xor_si256(in1_val, _mm256_set1_epi32(0xcccccccc));
					lut_val = _mm256_and_si256(lut_val, in1_msk);

			__m256i in0_val = _mm256_i32gather_epi32(m_input_vector, _mm256_load_si256(&m_input_index[0][i]), 4);
			__m256i in0_msk = _mm256_xor_si256(in0_val, _mm256_set1_epi32(0xaaaaaaaa));
					lut_val = _mm256_and_si256(lut_val, in0_msk);

			__m256i out_val = _mm256_cmpeq_epi32(lut_val, _mm256_set1_epi32(0x00000000));
					out_val = _mm256_andnot_si256(out_val, _mm256_set1_epi32(0xffffffff));

					_mm256_store_si256(&m_output_vector[i], out_val);
		}
	}
};



class Lut6Net
{
public:
	Lut6Net(std::vector<int> layer_num)
	{
		const int* in_vec = nullptr;
		for (size_t i = 0; i < layer_num.size(); i++ ) {
			m_layer.push_back(std::make_unique<Lut6LayerAvx2>(layer_num[i]));
			m_layer[i]->SetInputVector(in_vec);
			in_vec = m_layer[i]->GetOutputVector();
		}
	}

	Lut6LayerAvx2& operator[](size_t i) {
		return *m_layer[i].get();
	}

protected:
	std::vector< std::unique_ptr<Lut6LayerAvx2> >	m_layer;
};

