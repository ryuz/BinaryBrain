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


// Convolution用展開クラス マクロバージョン
template <int INPUT_C_SIZE, int INPUT_H_SIZE, int INPUT_W_SIZE, int FILTER_H_SIZE, int FILTER_W_SIZE,
			typename ST = float, typename ET = float, typename T = float, typename INDEX = size_t>
class NeuralNetConvExpandM : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	INDEX			m_mux_size = 1;
	INDEX			m_input_frame_size = 1;
	INDEX			m_output_frame_size = 1;
	const int		OUTPUT_H_SIZE = INPUT_H_SIZE - FILTER_H_SIZE + 1;
	const int		OUTPUT_W_SIZE = INPUT_W_SIZE - FILTER_W_SIZE + 1;
	
public:
	NeuralNetConvExpandM() {}
		
	~NeuralNetConvExpandM() {}		// デストラクタ
	
	void Resize()
	{
	}
	
	void SetMuxSize(INDEX mux_size) {
		m_mux_size = mux_size;
	}

	void SetBatchSize(INDEX batch_size) {
		m_input_frame_size = batch_size * m_mux_size;
		m_output_frame_size = m_input_frame_size * OUTPUT_H_SIZE * OUTPUT_W_SIZE * m_mux_size;
	}
	
	INDEX GetInputFrameSize(void) const { return m_input_frame_size; }
	INDEX GetInputNodeSize(void) const { return INPUT_C_SIZE * INPUT_H_SIZE * INPUT_W_SIZE; }
	INDEX GetOutputFrameSize(void) const { return m_output_frame_size; }
	INDEX GetOutputNodeSize(void) const { return INPUT_C_SIZE * FILTER_H_SIZE * FILTER_W_SIZE; }
	
	int   GetInputSignalDataType(void) const { return NeuralNetType<ST>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<ST>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<ET>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<ET>::type; }
	
protected:

	inline int GetInputNode(int c, int y, int x)
	{
		return (c * INPUT_H_SIZE + y)*INPUT_W_SIZE + x;
	}

	inline int GetOutputNode(int c, int y, int x)
	{
		return (c * FILTER_H_SIZE + y)*FILTER_W_SIZE + x;
	}


public:
	void Forward(bool train = true)
	{
		auto in_sig_buf = GetInputSignalBuffer();
		auto out_sig_buf = GetOutputSignalBuffer();
		
//		INDEX output_frame = 0;
#pragma omp parallel for
		for (int input_frame = 0; input_frame < (int)m_input_frame_size; ++input_frame) {
			for (int y = 0; y < OUTPUT_H_SIZE; ++y) {
				for (int x = 0; x < OUTPUT_W_SIZE; ++x) {
					INDEX output_frame = (input_frame*OUTPUT_H_SIZE + y) * OUTPUT_W_SIZE + x;
					for (int c = 0; c < INPUT_C_SIZE; ++c) {
						for (int fy = 0; fy < FILTER_H_SIZE; ++fy) {
							for (int fx = 0; fx < FILTER_W_SIZE; ++fx) {
								int ix = x + fx;
								int iy = y + fy;

								int input_node = GetInputNode(c, iy, ix);
								int output_node = GetOutputNode(c, fy, fx);
								out_sig_buf.Set<ST>(output_frame, output_node, in_sig_buf.Get<ST>(input_frame, input_node));
							}
						}
					}
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
			for (int y = 0; y < OUTPUT_H_SIZE; ++y) {
				for (int x = 0; x < OUTPUT_W_SIZE; ++x) {
					#pragma omp parallel for
					for (int c = 0; c < INPUT_C_SIZE; ++c) {
						for (int fy = 0; fy < FILTER_H_SIZE; ++fy) {
							for (int fx = 0; fx < FILTER_W_SIZE; ++fx) {
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