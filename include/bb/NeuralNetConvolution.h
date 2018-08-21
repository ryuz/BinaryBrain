// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <vector>
#include <random>
#include <intrin.h>


namespace bb {


// NeuralNetの抽象クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNetConvolution : public NeuralNetLayer<T, INDEX>
{
protected:
	INDEX			m_frame_size;
	int				m_input_h_size;
	int				m_input_w_size;
	int				m_input_c_size;
	int				m_filter_h_size;
	int				m_filter_w_size;
	int				m_output_h_size;
	int				m_output_w_size;
	int				m_output_c_size;
	std::vector <T>	m_coeff;

public:
	NeuralNetConvolution() {}
	
	NeuralNetConvolution(INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX output_c_size, INDEX filter_h_size, INDEX filter_w_size, INDEX batch_size = 1, std::uint64_t seed = 1)
	{
		Setup(input_c_size, input_h_size, input_w_size, output_c_size, filter_h_size, filter_w_size, batch_size, seed);
	}
	
	~NeuralNetConvolution() {}		// デストラクタ
	
	void Setup(INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX output_c_size, INDEX filter_h_size, INDEX filter_w_size, INDEX batch_size = 1, std::uint64_t seed = 1)
	{
		m_frame_size = batch_size;
		m_input_c_size = (int)input_c_size;
		m_input_h_size = (int)input_h_size;
		m_input_w_size = (int)input_w_size;
		m_filter_h_size = (int)filter_h_size;
		m_filter_w_size = (int)filter_w_size;
		m_output_c_size = (int)output_c_size;
		m_output_h_size = m_input_h_size - m_filter_h_size + 1;
		m_output_w_size = m_input_w_size - m_filter_w_size + 1;

		m_coeff.resize(m_output_c_size*m_input_c_size*m_filter_h_size*m_filter_w_size);

		std::mt19937_64 mt(seed);
		std::uniform_real_distribution<T> uniform_rand((T)0, (T)1);
		for (auto& v : m_coeff) {
			v = uniform_rand(mt);
		}
	}
	
	T& coeff(INDEX n, INDEX c, INDEX y, INDEX x) {
		return m_coeff[((n*m_input_c_size + c)*m_filter_h_size + y)*m_filter_w_size + x];
	}


	void SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }
	
	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_input_c_size * m_input_h_size * m_input_w_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_output_c_size * m_output_h_size * m_output_w_size; }
	
	int   GetInputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }
	
protected:

	inline T* GetInputPtr(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
	{
		return buf.GetPtr<T>((c*m_input_h_size + y)*m_input_w_size + x);
	}

	inline T* GetOutputPtr(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
	{
		return buf.GetPtr<T>((c*m_output_h_size + y)*m_output_w_size + x);
	}

public:
	void Forward(void)
	{
		int  m256_frame_size = (int)(((m_frame_size + 7) / 8) * 8);
		auto in_buf = GetInputValueBuffer();
		auto out_buf = GetOutputValueBuffer();

		for (int n = 0; n < m_output_c_size; ++n) {
			for (int y = 0; y < m_output_h_size; ++y) {
				for (int x = 0; x < m_output_w_size; ++x) {
					float* out_ptr = GetOutputPtr(out_buf, n, y, x);
					for (size_t frame = 0; frame < m256_frame_size; frame += 8) {
						__m256 sum = _mm256_set1_ps(0);
						for (int c = 0; c < m_input_c_size; ++c) {
							for (int fy = 0; fy < m_filter_h_size; ++fy) {
								for (int fx = 0; fx < m_filter_w_size; ++fx) {
									int ix = x + fx;
									int iy = y + fy;
									float* in_ptr = GetInputPtr(in_buf, c, ix, iy);
									__m256 coeff_val = _mm256_set1_ps(coeff(n, c, fy, fx));
									__m256 in_val = _mm256_load_ps(&in_ptr[frame]);
									__m256 mul_val = _mm256_mul_ps(coeff_val, in_val);
									sum = _mm256_add_ps(sum, mul_val);
								}
							}
						}
						_mm256_store_ps(&out_ptr[frame], sum);
					}
				}
			}
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