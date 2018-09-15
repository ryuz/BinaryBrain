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
#include "NeuralNetLayerBuf.h"


namespace bb {


// Convolutionクラス
template <typename T = float, typename INDEX = size_t>
class NeuralNetMaxPooling : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	INDEX			m_mux_size = 1;
	INDEX			m_frame_size = 1;
	int				m_input_h_size;
	int				m_input_w_size;
	int				m_input_c_size;
	int				m_filter_h_size;
	int				m_filter_w_size;
	int				m_output_h_size;
	int				m_output_w_size;

public:
	NeuralNetMaxPooling() {}
	
	NeuralNetMaxPooling(INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX filter_h_size, INDEX filter_w_size, std::uint64_t seed = 1)
	{
		Resize(input_c_size, input_h_size, input_w_size, filter_h_size, filter_w_size, seed);
	}
	
	~NeuralNetMaxPooling() {}
	
	void Resize(INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX filter_h_size, INDEX filter_w_size, std::uint64_t seed = 1)
	{
		m_input_c_size = (int)input_c_size;
		m_input_h_size = (int)input_h_size;
		m_input_w_size = (int)input_w_size;
		m_filter_h_size = (int)filter_h_size;
		m_filter_w_size = (int)filter_w_size;
		m_output_h_size = m_input_h_size / m_filter_h_size;
		m_output_w_size = m_input_w_size / m_filter_w_size;


		std::mt19937_64 mt(seed);
		std::uniform_real_distribution<T> uniform_rand((T)0, (T)1);
		for (auto& w : m_W) {
			w = uniform_rand(mt);
		}
		for (auto& b : m_b) {
			b = uniform_rand(mt);
		}
	}
	
	void SetMuxSize(INDEX mux_size) { m_mux_size = mux_size; }

	void SetBatchSize(INDEX batch_size) {
		m_frame_size = batch_size * m_mux_size;
	}
	
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
		return (T*)buf.GetPtr((c*m_input_h_size + y)*m_input_w_size + x);
	}

	inline T* GetOutputPtr(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
	{
		return (T*)buf.GetPtr((c*m_output_h_size + y)*m_output_w_size + x);
	}

	inline T* GetOutputPtrWithRangeCheck(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
	{
		if (x < 0 || x >= m_output_w_size || y < 0 || y >= m_output_h_size) {
			return (T*)buf.GetZeroPtr();
		}

		return (T*)buf.GetPtr((c*m_output_h_size + y)*m_output_w_size + x);
	}

public:
	void Forward(bool train = true)
	{
		if (typeid(T) == typeid(float)) {
			// float用実装
			int  m256_frame_size = (int)(((m_frame_size + 7) / 8) * 8);
			auto in_buf = GetInputValueBuffer();
			auto out_buf = GetOutputValueBuffer();

			for (int n = 0; n < m_input_c_size; ++n) {
				for (int y = 0; y < m_output_h_size; ++y) {
					for (int x = 0; x < m_output_w_size; ++x) {
						float* out_ptr = GetOutputPtr(out_buf, n, y, x);

						for (size_t frame = 0; frame < m256_frame_size; frame += 8) {
							for (int fy = 0; fy < m_filter_h_size; ++fy) {
								int iy = y*m_filter_h_size + fy;
								__m256	max_val = _mm256_set1_ps(0.0f);	// 前段に活性化入れるから0がminだよね？
								for (int fx = 0; fx < m_filter_w_size; ++fx) {
									int ix = x*m_filter_w_size + fx;
									float* in_ptr = GetInputPtr(in_buf, c, iy, ix);
									__m256 in_val = _mm256_load_ps(&in_ptr[frame]);
									max_val = _mm256_max_ps(max_val, in_val);
								}
							}
							_mm256_store_ps(&out_ptr[frame], max_val);
						}
					}
				}
			}
		}
		else if (typeid(T) == typeid(double)) {
			// double用実装
		}
		else {
			assert(0);
		}
	}
	
	void Backward(void)
	{
		if (typeid(T) == typeid(float)) {
			// float用実装
			int  m256_frame_size = (int)(((m_frame_size + 7) / 8) * 8);
			auto in_sig_buf = GetInputValueBuffer();
			auto out_sig_buf = GetOutputValueBuffer();
			auto in_err_buf = GetInputErrorBuffer();
			auto out_err_buf = GetOutputErrorBuffer();

				__m256 sum_db = _mm256_set1_ps(0);
			for (int n = 0; n < m_input_c_size; ++n) {
				for (int y = 0; y < m_output_h_size; ++y) {
					for (int x = 0; x < m_output_w_size; ++x) {
					float* out_sig_ptr = GetOutputPtr(out_sig_buf, n, y, x);
					float* out_err_ptr = GetOutputPtr(out_err_buf, n, y, x);

					for (size_t frame = 0; frame < m256_frame_size; frame += 8) {
						__m256 out_sig = _mm256_load_ps(&out_sig_ptr[frame]);
						__m256 out_err = _mm256_load_ps(&out_err_ptr[frame]);
						for (int fy = 0; fy < m_filter_h_size; ++fy) {
							int iy = y*m_filter_h_size + fy;
							for (int fx = 0; fx < m_filter_w_size; ++fx) {
								int ix = x*m_filter_w_size + fx;
								float* in_sig_ptr = GetInputPtr(in_sig_buf, n, iy, ix);
								float* in_err_ptr = GetInputPtr(in_err_buf, n, iy, ix);
								__m256 in_sig = _mm256_load_ps(&in_sig_buf[frame]);
								__m256 mask = _mm256_cmp_ps(in_sig, out_sig, _CMP_EQ_OQ);
								__m256 in_err = _mm256_and_ps(mask, out_err);
								_mm256_store_ps(&in_err_ptr[frame], in_err);
							}
						}
					}
				}
			}
		}
	}

	void Update(double learning_rate)
	{
	}

};


}