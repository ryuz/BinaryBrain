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
	int				m_filter_h_size;
	int				m_filter_w_size;
	std::vector <T>	m_coeff;

public:
	NeuralNetConvolution() {}
	
	NeuralNetConvolution(INDEX input_h_size, INDEX input_w_size, INDEX filter_h_size, INDEX filter_w_size, INDEX batch_size = 1, std::uint64_t seed = 1)
	{
		Setup(input_h_size, input_w_size, filter_h_size, filter_w_size, batch_size, seed);
	}
	
	~NeuralNetConvolution() {}		// デストラクタ
	
	void Setup(INDEX input_h_size, INDEX input_w_size, INDEX filter_h_size, INDEX filter_w_size, INDEX batch_size = 1, std::uint64_t seed=1)
	{
		m_frame_size = batch_size;
		m_input_h_size = (int)input_h_size;
		m_input_w_size = (int)input_w_size;
		m_filter_h_size = (int)filter_h_size;
		m_filter_w_size = (int)filter_w_size;
		m_coeff.resize(m_filter_h_size*m_filter_w_size);

		std::mt19937_64 mt(seed);
		std::uniform_real_distribution<T> uniform_rand((T)0, (T)1);
		for (auto& v : m_coeff) {
			v = uniform_rand(mt);
		}
	}
	
	T& coeff(INDEX y, INDEX x) { return m_coeff[y*m_filter_w_size + x]; }


	void SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }
	
	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_input_h_size * m_input_w_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return (m_input_h_size - m_filter_h_size + 1) * (m_input_w_size - m_filter_w_size + 1); }
	
	int   GetInputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }
	
	void Forward(void)
	{
		int widht  = m_input_w_size - m_filter_w_size + 1;
		int height = m_input_h_size - m_filter_h_size + 1;
		size_t frame_size = ((m_frame_size + 7) / 8) * 8;
		auto in_buf = GetInputValueBuffer();
		auto out_buf = GetOutputValueBuffer();

		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < height; ++x) {
				int output_node = y*widht + x;
				float* out_ptr = out_buf.GetPtr<float>(output_node);
				for (size_t frame = 0; frame < frame_size; frame += 8) {
					__m256 sum = _mm256_set1_ps(0);
					for (int iy = 0; iy < m_filter_h_size; ++iy) {
						for (int ix = 0; ix < m_filter_w_size; ++ix) {
							int xx = x + ix;
							int yy = y + iy;
							int filter_node = iy*m_filter_w_size + ix;
							int input_node = yy*m_input_w_size + xx;
							float* in_ptr = in_buf.GetPtr<float>(input_node);
							__m256 coeff = _mm256_set1_ps(m_coeff[filter_node]);
							__m256 in_val = _mm256_load_ps(&in_ptr[frame]);
							__m256 mul_val = _mm256_mul_ps(coeff, in_val);
							sum = _mm256_add_ps(sum, mul_val);
						}
					}
					_mm256_store_ps(&out_ptr[frame], sum);
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