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


// Convolutionクラス
template <typename ST = float, typename ET = float, typename T = float, typename INDEX = size_t>
class NeuralNetConvExpand : public NeuralNetLayerBuf<T, INDEX>
{
protected:
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

	std::string GetClassName(void) const { return "NeuralNetConvExpand"; }

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
	
	void SetBatchSize(INDEX batch_size) {
		m_input_frame_size = batch_size;
		m_output_frame_size = m_input_frame_size * m_output_h_size * m_output_w_size;
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
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();

		const int frame_size = (int)out_sig_buf.GetFrameStride() * 8 / NeuralNetType<ST>::bit_size;
		const int frame_unit = 256 / NeuralNetType<ST>::bit_size;

		for (int c = 0; c < m_input_c_size; ++c) {
#pragma omp parallel for
			for (int frame_base = 0; frame_base < frame_size; frame_base += frame_unit) {
				for (int fy = 0; fy < m_filter_h_size; ++fy) {
					for (int fx = 0; fx < m_filter_w_size; ++fx) {
						int output_node = GetOutputNode(c, fy, fx);
						for (int frame_step = 0; frame_step < frame_unit; ++frame_step) {
							int output_frame = frame_base + frame_step;
							int input_frame = output_frame / (m_output_h_size * m_output_w_size);
							int f = output_frame % (m_output_h_size * m_output_w_size);
							int ix = f % m_output_w_size;
							int iy = f / m_output_w_size;
							ix += fx;
							iy += fy;
							int input_node = GetInputNode(c, iy, ix);
							ST sig = in_sig_buf.template Get<ST>(input_frame, input_node);
							out_sig_buf.template Set<ST>(output_frame, output_node, sig);
						}
					}
				}
			}
		}
	}


	/*
	void Forward(bool train = true)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		
		#pragma omp parallel for
		for (int input_frame = 0; input_frame < (int)m_input_frame_size; ++input_frame) {
			for (int y = 0; y < m_output_h_size; ++y) {
				for (int x = 0; x < m_output_w_size; ++x) {
					INDEX output_frame = (input_frame*m_output_h_size + y) * m_output_w_size + x;
					for (int c = 0; c < m_input_c_size; ++c) {
						for (int fy = 0; fy < m_filter_h_size; ++fy) {
							for (int fx = 0; fx < m_filter_w_size; ++fx) {
								int ix = x + fx;
								int iy = y + fy;
								int input_node = GetInputNode(c, iy, ix);
								int output_node = GetOutputNode(c, fy, fx);
								ST sig = in_sig_buf.template Get<ST>(input_frame, input_node);
								out_sig_buf.template Set<ST>(output_frame, output_node, sig);
							}
						}
					}
				}
			}
		}
	}
	*/
	

	void Backward(void)
	{
		auto out_err_buf = this->GetOutputErrorBuffer();
		auto in_err_buf = this->GetInputErrorBuffer();

		in_err_buf.Clear();

		const int frame_size = (int)out_err_buf.GetFrameStride() * 8 / NeuralNetType<ST>::bit_size;
		const int frame_unit = 256 / NeuralNetType<ST>::bit_size;

		for (int c = 0; c < m_input_c_size; ++c) {
#pragma omp parallel for
			for (int frame_base = 0; frame_base < frame_size; frame_base += frame_unit) {
				for (int fy = 0; fy < m_filter_h_size; ++fy) {
					for (int fx = 0; fx < m_filter_w_size; ++fx) {
						int output_node = GetOutputNode(c, fy, fx);
						for (int frame_step = 0; frame_step < frame_unit; ++frame_step) {
							int output_frame = frame_base + frame_step;
							int input_frame = output_frame / (m_output_h_size * m_output_w_size);
							int f = output_frame % (m_output_h_size * m_output_w_size);
							int ix = f % m_output_w_size;
							int iy = f / m_output_w_size;
							ix += fx;
							iy += fy;
							int input_node = GetInputNode(c, iy, ix);
							ET err = out_err_buf.template Get<ET>(output_frame, output_node);
							in_err_buf.template Set<ET>(input_frame, input_node, in_err_buf.template Get<ET>(input_frame, input_node) + err);
						}
					}
				}
			}
		}
	}

	/*
	void Backward(void)
	{
		auto out_err_buf = GetOutputErrorBuffer();
		auto in_err_buf = GetInputErrorBuffer();

		in_err_buf.Clear();

#pragma omp parallel for
		for (int input_frame = 0; input_frame < m_input_frame_size; ++input_frame) {
			for (int y = 0; y < m_output_h_size; ++y) {
				for (int x = 0; x < m_output_w_size; ++x) {
					INDEX output_frame = (input_frame*m_output_h_size + y) * m_output_w_size + x;
					for (int c = 0; c < m_input_c_size; ++c) {
						for (int fy = 0; fy < m_filter_h_size; ++fy) {
							for (int fx = 0; fx < m_filter_w_size; ++fx) {
								int ix = x + fx;
								int iy = y + fy;
								int output_node = GetOutputNode(c, fy, fx);
								int input_node = GetInputNode(c, iy, ix);
								ET err = out_err_buf.template Get<ET>(output_frame, output_node);
								in_err_buf.template Set<ET>(input_frame, input_node, in_err_buf.template Get<ET>(input_frame, input_node) + err);
							}
						}
					}
				}
			}
		}
	}
	*/

	void Update(void)
	{
	}
};


}