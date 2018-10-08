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
#include "NeuralNetLayerBuf.h"


namespace bb {



template <typename ST = float, typename ET = float, typename T = float, typename INDEX = size_t>
class NeuralNetConvCollapse : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	INDEX			m_input_frame_size = 1;
	INDEX			m_output_frame_size = 1;
	int				m_c_size;
	int				m_h_size;
	int				m_w_size;

public:
	NeuralNetConvCollapse() {}
	
	NeuralNetConvCollapse(INDEX c_size, INDEX h_size, INDEX w_size)
	{
		Resize(c_size, h_size, w_size);
	}
	
	~NeuralNetConvCollapse() {}		// デストラクタ

	std::string GetClassName(void) const { return "NeuralNetConvCollapse"; }

	void Resize(INDEX c_size, INDEX h_size, INDEX w_size)
	{
		m_c_size = (int)c_size;
		m_h_size = (int)h_size;
		m_w_size = (int)w_size;
	}

	void SetBatchSize(INDEX batch_size) {
		m_input_frame_size  = batch_size * m_h_size * m_w_size;
		m_output_frame_size = batch_size;
	}
	
	INDEX GetInputFrameSize(void) const { return m_input_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_c_size; }
	INDEX GetOutputFrameSize(void) const { return m_output_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_c_size * m_h_size * m_w_size; }
	
	int   GetInputSignalDataType(void) const { return NeuralNetType<ST>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<ST>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<ET>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<ET>::type; }
	
protected:

	/*
	inline T* GetInputPtr(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
	{
		return (T*)buf.GetPtr((c*m_input_h_size + y)*m_input_w_size + x);
	}

	inline T* GetOutputPtr(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
	{
		return (T*)buf.GetPtr((c*m_filter_h_size + y)*m_filter_w_size + x);
	}
	
	inline int GetInputNode(int c, int y, int x)
	{
		return (c*m_input_h_size + y)*m_input_w_size + x;
	}

	inline int GetOutputNode(int c, int y, int x)
	{
		return (c*m_filter_h_size + y)*m_filter_w_size + x;
	}
	*/

public:
	void Forward(bool train = true)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		
		INDEX input_frame = 0;
		for (INDEX output_frame = 0; output_frame < m_output_frame_size; ++output_frame) {
			for (int y = 0; y < m_h_size; ++y) {
				for (int x = 0; x < m_w_size; ++x) {
					#pragma omp parallel for
					for (int c = 0; c < m_c_size; ++c) {
		//				float* in_sig_ptr = (float *)in_sig_buf.GetPtr(c);
		//				float* out_sig_ptr = (float *)out_sig_buf.GetPtr((c*m_h_size + y)*m_w_size + x);
		//				out_sig_ptr[output_frame] = in_sig_ptr[input_frame];
						
						int input_node = c;
						int output_node = (c*m_h_size + y)*m_w_size + x;
						out_sig_buf.template Set<ST>(output_frame, output_node, in_sig_buf.template Get<ST>(input_frame, input_node));
					}
					++input_frame;
				}
			}
		}
	}
	
	void Backward(void)
	{
		auto out_err_buf = this->GetOutputErrorBuffer();
		auto in_err_buf = this->GetInputErrorBuffer();

		INDEX input_frame = 0;
		for (INDEX output_frame = 0; output_frame < m_output_frame_size; ++output_frame) {
			for (int y = 0; y < m_h_size; ++y) {
				for (int x = 0; x < m_w_size; ++x) {
					#pragma omp parallel for
					for (int c = 0; c < m_c_size; ++c) {
		//				float* out_err_ptr = (float *)out_err_buf.GetPtr((c*m_h_size + y)*m_w_size + x);
		//				float* in_err_ptr = (float *)in_err_buf.GetPtr(c);
		//				in_err_ptr[input_frame] = out_err_ptr[output_frame];

						int output_node = (c*m_h_size + y)*m_w_size + x;
						int input_node = c;
						in_err_buf.template Set<ET>(input_frame, input_node, out_err_buf.template Get<ET>(output_frame, output_node));
					}
					++input_frame;
				}
			}
		}
	}

	void Update(void)
	{
	}
};


}