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
template <typename ST = float, typename ET = float, typename T = float, typename INDEX = size_t>
class NeuralNetConvExpand : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	INDEX			m_mux_size = 1;
	INDEX			m_input_frame_size = 1;
	INDEX			m_output_frame_size = 1;
	int				m_input_h_size;
	int				m_input_w_size;
	int				m_input_c_size;
	int				m_filter_h_size;
	int				m_filter_w_size;
	int				m_output_h_size;
	int				m_output_w_size;

public:
	NeuralNetConvExpand() {}
	
	NeuralNetConvExpand(INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX filter_h_size, INDEX filter_w_size)
	{
		Resize(input_c_size, input_h_size, input_w_size, filter_h_size, filter_w_size);
	}
	
	~NeuralNetConvExpand() {}		// デストラクタ
	
	void Resize(INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX filter_h_size, INDEX filter_w_size)
	{
		m_input_c_size = (int)input_c_size;
		m_input_h_size = (int)input_h_size;
		m_input_w_size = (int)input_w_size;
		m_filter_h_size = (int)filter_h_size;
		m_filter_w_size = (int)filter_w_size;
		m_output_h_size = m_input_h_size - m_filter_h_size + 1;
		m_output_w_size = m_input_w_size - m_filter_w_size + 1;
	}
	
	void SetMuxSize(INDEX mux_size) {
		m_mux_size = mux_size;
	}

	void SetBatchSize(INDEX batch_size) {
		m_input_frame_size = batch_size * m_mux_size;
		m_output_frame_size = m_input_frame_size * m_output_h_size * m_output_w_size * m_mux_size;
	}
	
	INDEX GetInputFrameSize(void) const { return m_input_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_input_c_size * m_input_h_size * m_input_w_size; }
	INDEX GetOutputFrameSize(void) const { return m_output_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_input_c_size * m_filter_h_size * m_filter_w_size; }
	
	int   GetInputSignalDataType(void) const { return NeuralNetType<ST>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<ST>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<ET>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<ET>::type; }
	
protected:

//	inline float* GetInputPtr(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
//	{
//		return (float*)buf.GetPtr((c*m_input_h_size + y)*m_input_w_size + x);
//	}

//	inline float* GetOutputPtr(NeuralNetBuffer<T, INDEX>& buf, int c, int y, int x)
//	{
//		return (float*)buf.GetPtr((c*m_filter_h_size + y)*m_filter_w_size + x);
//	}

	inline int GetInputNode(int c, int y, int x)
	{
		return (c * m_input_h_size + y)*m_input_w_size + x;
	}

	inline int GetOutputNode(int c, int y, int x)
	{
		return (c*m_filter_h_size + y)*m_filter_w_size + x;
	}


public:
	void Forward(bool train = true)
	{
		auto in_sig_buf = GetInputSignalBuffer();
		auto out_sig_buf = GetOutputSignalBuffer();
		
		INDEX output_frame = 0;
		for (INDEX input_frame = 0; input_frame < m_input_frame_size; ++input_frame) {
			for (int y = 0; y < m_output_h_size; ++y) {
				for (int x = 0; x < m_output_w_size; ++x) {
					#pragma omp parallel for
					for (int c = 0; c < m_input_c_size; ++c) {
						for (int fy = 0; fy < m_filter_h_size; ++fy) {
							for (int fx = 0; fx < m_filter_w_size; ++fx) {
								int ix = x + fx;
								int iy = y + fy;
					//			const float* in_sig_ptr = GetInputPtr(in_sig_buf, c, iy, ix);
					//			float* out_sig_ptr = GetOutputPtr(out_sig_buf, c, fy, fx);
					//			out_sig_ptr[output_frame] = in_sig_ptr[input_frame];

								int input_node = GetInputNode(c, iy, ix);
								int output_node = GetOutputNode(c, fy, fx);
								out_sig_buf.Set<ST>(output_frame, output_node, in_sig_buf.Get<ST>(input_frame, input_node));
							}
						}
					}
					output_frame++;
				}
			}
		}
	}
	
	void Backward(void)
	{
		auto out_err_buf = GetOutputErrorBuffer();
		auto in_err_buf = GetInputErrorBuffer();

		in_err_buf.Clear();

		INDEX output_frame = 0;
		for (INDEX input_frame = 0; input_frame < m_input_frame_size; ++input_frame) {
			for (int y = 0; y < m_output_h_size; ++y) {
				for (int x = 0; x < m_output_w_size; ++x) {
					#pragma omp parallel for
					for (int c = 0; c < m_input_c_size; ++c) {
						for (int fy = 0; fy < m_filter_h_size; ++fy) {
							for (int fx = 0; fx < m_filter_w_size; ++fx) {
								int ix = x + fx;
								int iy = y + fy;
				//				const float* out_err_ptr = GetOutputPtr(out_err_buf, c, fy, fx);
				//				float* in_err_ptr = GetInputPtr(in_err_buf, c, iy, ix);
				//				in_err_ptr[input_frame] += out_err_ptr[output_frame];

								int output_node = GetOutputNode(c, fy, fx);
								int input_node = GetInputNode(c, iy, ix);
								ET err = in_err_buf.Get<ET>(input_frame, input_node);
								in_err_buf.Set<ET>(input_frame, input_node, err + out_err_buf.Get<ET>(output_frame, output_node));
							}
						}
					}
					output_frame++;
				}
			}
		}
	}

	void Update(void)
	{
	}
};


}