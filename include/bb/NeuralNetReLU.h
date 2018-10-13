// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "NeuralNetLayerBuf.h"


namespace bb {


// ReLU(活性化関数)
template <typename T = float, typename INDEX = size_t>
class NeuralNetReLU : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	INDEX		m_frame_size = 1;
	INDEX		m_node_size = 0;

public:
	NeuralNetReLU() {}

	NeuralNetReLU(INDEX node_size)
	{
		Resize(node_size);
	}

	~NeuralNetReLU() {}

	std::string GetClassName(void) const { return "NeuralNetReLU"; }

	void Resize(INDEX node_size)
	{
		m_node_size = node_size;
	}

	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	void Forward(bool train = true)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		int  m256_frame_size = (int)(((m_frame_size + 7) / 8) * 8);

		__m256 zero = _mm256_set1_ps(0);
		for (INDEX node = 0; node < m_node_size; ++node) {
			T* in_sig_ptr = (T*)in_sig_buf.GetPtr(node);
			T* out_sig_ptr = (T*)out_sig_buf.GetPtr(node);
			for (INDEX frame = 0; frame < m256_frame_size; frame += 8 ) {
				__m256 in_sig = _mm256_load_ps(&in_sig_ptr[frame]);
				in_sig = _mm256_max_ps(in_sig, zero);
				_mm256_store_ps(&out_sig_ptr[frame], in_sig);
			}
		}				
	}

	void Backward(void)
	{
		auto out_sig_buf = this->GetOutputSignalBuffer();
		auto out_err_buf = this->GetOutputErrorBuffer();
		auto in_err_buf = this->GetInputErrorBuffer();
		int  m256_frame_size = (int)(((m_frame_size + 7) / 8) * 8);

		__m256 zero = _mm256_set1_ps(0);
		for (INDEX node = 0; node < m_node_size; ++node) {
			T* out_sig_ptr = (T*)out_sig_buf.GetPtr(node);
			T* out_err_ptr = (T*)out_err_buf.GetPtr(node);
			T* in_err_ptr = (T*)in_err_buf.GetPtr(node);
			for (INDEX frame = 0; frame < m256_frame_size; frame += 8) {
				__m256 out_sig = _mm256_load_ps(&out_sig_ptr[frame]);
				__m256 out_err = _mm256_load_ps(&out_err_ptr[frame]);
				__m256 mask = _mm256_cmp_ps(out_sig, zero, _CMP_GT_OS);
				__m256 in_err = _mm256_and_ps(out_err, mask);
				_mm256_store_ps(&in_err_ptr[frame], in_err);
			}
		}
	}

	void Update(void)
	{
	}

};

}
