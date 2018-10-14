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

#include "bb/NeuralNetLayerBuf.h"


namespace bb {


// MaxPoolingクラス
template <typename ST = float, typename ET = float, typename T = float, typename INDEX = size_t>
class NeuralNetMaxPooling : public NeuralNetLayerBuf<T, INDEX>
{
protected:
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
	
	NeuralNetMaxPooling(INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX filter_h_size, INDEX filter_w_size)
	{
		Resize(input_c_size, input_h_size, input_w_size, filter_h_size, filter_w_size);
	}
	
	~NeuralNetMaxPooling() {}

	std::string GetClassName(void) const { return "NeuralNetMaxPooling"; }

	void Resize(INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX filter_h_size, INDEX filter_w_size)
	{
		m_input_c_size = (int)input_c_size;
		m_input_h_size = (int)input_h_size;
		m_input_w_size = (int)input_w_size;
		m_filter_h_size = (int)filter_h_size;
		m_filter_w_size = (int)filter_w_size;
		m_output_h_size = m_input_h_size / m_filter_h_size;
		m_output_w_size = m_input_w_size / m_filter_w_size;
	}
	
	void SetBatchSize(INDEX batch_size) {
		m_frame_size = batch_size;
	}
	
	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_input_c_size * m_input_h_size * m_input_w_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_input_c_size * m_output_h_size * m_output_w_size; }
	
	int   GetInputSignalDataType(void) const { return NeuralNetType<ST>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<ST>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<ET>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<ET>::type; }
	
protected:

	inline void* GetInputPtr(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
	{
		return buf.GetPtr((c*m_input_h_size + y)*m_input_w_size + x);
	}

	inline void* GetOutputPtr(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
	{
		return buf.GetPtr((c*m_output_h_size + y)*m_output_w_size + x);
	}

public:
	void Forward(bool train = true)
	{
		if (typeid(ST) == typeid(float)) {
			// float用実装
			int  m256_frame_size = (int)(((m_frame_size + 7) / 8) * 8);
			auto in_sig_buf = this->GetInputSignalBuffer();
			auto out_sig_buf = this->GetOutputSignalBuffer();
			#pragma omp parallel for
			for (int c = 0; c < m_input_c_size; ++c) {
				for (int y = 0; y < m_output_h_size; ++y) {
					for (int x = 0; x < m_output_w_size; ++x) {
						float* out_sig_ptr = (float*)GetOutputPtr(out_sig_buf, c, y, x);

						for (size_t frame = 0; frame < m256_frame_size; frame += 8) {
							__m256	max_val = _mm256_set1_ps(0.0f);	// 前段に活性化入れるから0がminだよね？
							for (int fy = 0; fy < m_filter_h_size; ++fy) {
								int iy = y*m_filter_h_size + fy;
								for (int fx = 0; fx < m_filter_w_size; ++fx) {
									int ix = x*m_filter_w_size + fx;
									float* in_sig_ptr = (float*)GetInputPtr(in_sig_buf, c, iy, ix);
									__m256 in_sig = _mm256_load_ps(&in_sig_ptr[frame]);
									max_val = _mm256_max_ps(max_val, in_sig);
								}
							}
							_mm256_store_ps(&out_sig_ptr[frame], max_val);
						}
					}
				}
			}
		}
		else if (typeid(ST) == typeid(double)) {
			// double用実装
		}
		else if (typeid(ST) == typeid(Bit) || typeid(ST) == typeid(bool)) {
			// バイナリ用実装
			int  m256_frame_size = (int)((m_frame_size + 255) / 256);
			auto in_sig_buf = this->GetInputSignalBuffer();
			auto out_sig_buf = this->GetOutputSignalBuffer();

			#pragma omp parallel for
			for (int c = 0; c < m_input_c_size; ++c) {
				for (int y = 0; y < m_output_h_size; ++y) {
					for (int x = 0; x < m_output_w_size; ++x) {
						__m256i* out_sig_ptr = (__m256i*)GetOutputPtr(out_sig_buf, c, y, x);
						for (size_t frame = 0; frame < m256_frame_size; ++frame) {
							__m256i	or_val = _mm256_set1_epi32(0);
							for (int fy = 0; fy < m_filter_h_size; ++fy) {
								int iy = y*m_filter_h_size + fy;
								for (int fx = 0; fx < m_filter_w_size; ++fx) {
									int ix = x*m_filter_w_size + fx;
									__m256i* in_sig_ptr = (__m256i*)GetInputPtr(in_sig_buf, c, iy, ix);
									__m256i in_sig = _mm256_load_si256(&in_sig_ptr[frame]);
									or_val = _mm256_or_si256(or_val, in_sig);
								}
							}
							_mm256_store_si256(&out_sig_ptr[frame], or_val);
						}
					}
				}
			}
		}
		else {
			assert(0);
		}
	}
	
	void Backward(void)
	{
		if (typeid(ST) == typeid(float) && typeid(ET) == typeid(float)) {
			// float用実装
			int  m256_frame_size = (int)(((m_frame_size + 7) / 8) * 8);
			auto in_sig_buf = this->GetInputSignalBuffer();
			auto out_sig_buf = this->GetOutputSignalBuffer();
			auto in_err_buf = this->GetInputErrorBuffer();
			auto out_err_buf = this->GetOutputErrorBuffer();

			#pragma omp parallel for
			for (int n = 0; n < m_input_c_size; ++n) {
				for (int y = 0; y < m_output_h_size; ++y) {
					for (int x = 0; x < m_output_w_size; ++x) {
						float* out_sig_ptr = (float*)GetOutputPtr(out_sig_buf, n, y, x);
						float* out_err_ptr = (float*)GetOutputPtr(out_err_buf, n, y, x);

						for (size_t frame = 0; frame < m256_frame_size; frame += 8) {
							__m256 out_sig = _mm256_load_ps(&out_sig_ptr[frame]);
							__m256 out_err = _mm256_load_ps(&out_err_ptr[frame]);
							for (int fy = 0; fy < m_filter_h_size; ++fy) {
								int iy = y*m_filter_h_size + fy;
								for (int fx = 0; fx < m_filter_w_size; ++fx) {
									int ix = x*m_filter_w_size + fx;
									float* in_sig_ptr = (float*)GetInputPtr(in_sig_buf, n, iy, ix);
									float* in_err_ptr = (float*)GetInputPtr(in_err_buf, n, iy, ix);
									__m256 in_sig = _mm256_load_ps(&in_sig_ptr[frame]);
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
		else if (typeid(ET) == typeid(double)) {
			// double用実装
		}
		else {
			assert(0);
		}
	}

	void Update(void)
	{
	}

};


}