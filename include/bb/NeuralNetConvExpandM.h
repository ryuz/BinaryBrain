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


// Convolution用展開クラス マクロバージョン
template <int INPUT_C_SIZE, int INPUT_H_SIZE, int INPUT_W_SIZE, int FILTER_H_SIZE, int FILTER_W_SIZE,
			typename ST = float, typename ET = float, typename T = float, typename INDEX = size_t>
class NeuralNetConvExpandM : public NeuralNetLayerBuf<T, INDEX>
{
protected:
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
	
	void SetBatchSize(INDEX batch_size) {
		m_input_frame_size = batch_size;
		m_output_frame_size = m_input_frame_size * OUTPUT_H_SIZE * OUTPUT_W_SIZE;
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
		return (c * INPUT_H_SIZE + y) * INPUT_W_SIZE + x;
	}

	inline int GetOutputNode(int c, int y, int x)
	{
		return (c * FILTER_H_SIZE + y) * FILTER_W_SIZE + x;
	}

public:
	/*
	void Forward(bool train = true)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		
#pragma omp parallel for
		for (int input_frame = 0; input_frame < (int)m_input_frame_size; ++input_frame) {
			for (int y = 0; y < OUTPUT_H_SIZE; ++y) {
				for (int x = 0; x < OUTPUT_W_SIZE; ++x) {
					INDEX output_frame = (input_frame*OUTPUT_H_SIZE + y) * OUTPUT_W_SIZE + x;
					for (int fy = 0; fy < FILTER_H_SIZE; ++fy) {
						for (int fx = 0; fx < FILTER_W_SIZE; ++fx) {
							for (int c = 0; c < INPUT_C_SIZE; ++c) {
								int ix = x + fx;
								int iy = y + fy;

								int input_node = GetInputNode(c, iy, ix);
								int output_node = GetOutputNode(c, fy, fx);
								ST sig = in_sig_buf.Get<ST>(input_frame, input_node);
								out_sig_buf.Set<ST>(output_frame, output_node, sig);
							}
						}
					}
				}
			}
		}
	}
	*/
	
	void output_to_input(int c, int output_frame, int output_node, int& input_frame, int& input_node)
	{
		int x = output_frame % OUTPUT_W_SIZE;
		int y = output_frame / OUTPUT_W_SIZE;
		
	}

	void Forward(bool train = true)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();

		const int frame_size = (int)out_sig_buf.GetFrameStride() * 8 / NeuralNetType<ST>::bit_size;
		const int frame_unit = 256 / NeuralNetType<ST>::bit_size;

		for (int c = 0; c < INPUT_C_SIZE; ++c) {
#pragma omp parallel for
			for (int frame_base = 0; frame_base < frame_size; frame_base += frame_unit) {
				for (int fy = 0; fy < FILTER_H_SIZE; ++fy) {
					for (int fx = 0; fx < FILTER_W_SIZE; ++fx) {
						int output_node = GetOutputNode(c, fy, fx);
						for (int frame_step = 0; frame_step < frame_unit; ++frame_step) {
							int output_frame = frame_base + frame_step;
							int input_frame  = output_frame / (OUTPUT_H_SIZE * OUTPUT_W_SIZE);
							int f            = output_frame % (OUTPUT_H_SIZE * OUTPUT_W_SIZE);
							int ix = f % OUTPUT_W_SIZE;
							int iy = f / OUTPUT_W_SIZE;
							ix += fx;
							iy += fy;
							int input_node = GetInputNode(c, iy, ix);
							ST sig = in_sig_buf.Get<ST>(input_frame, input_node);
							out_sig_buf.Set<ST>(output_frame, output_node, sig);
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

		const int frame_size = (int)out_err_buf.GetFrameStride() * 8 / NeuralNetType<ST>::bit_size;
		const int frame_unit = 256 / NeuralNetType<ST>::bit_size;

		for (int c = 0; c < INPUT_C_SIZE; ++c) {
#pragma omp parallel for
			for (int frame_base = 0; frame_base < frame_size; frame_base += frame_unit) {
				for (int fy = 0; fy < FILTER_H_SIZE; ++fy) {
					for (int fx = 0; fx < FILTER_W_SIZE; ++fx) {
						int output_node = GetOutputNode(c, fy, fx);
						for (int frame_step = 0; frame_step < frame_unit; ++frame_step) {
							int output_frame = frame_base + frame_step;
							int input_frame = output_frame / (OUTPUT_H_SIZE * OUTPUT_W_SIZE);
							int f = output_frame % (OUTPUT_H_SIZE * OUTPUT_W_SIZE);
							int ix = f % OUTPUT_W_SIZE;
							int iy = f / OUTPUT_W_SIZE;
							ix += fx;
							iy += fy;
							int input_node = GetInputNode(c, iy, ix);
							ET err = out_err_buf.Get<ET>(output_frame, output_node);
							in_err_buf.Set<ET>(input_frame, input_node, in_err_buf.Get<ET>(input_frame, input_node) + err);
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
		for (int input_frame = 0; input_frame < (int)m_input_frame_size; ++input_frame) {
			for (int y = 0; y < OUTPUT_H_SIZE; ++y) {
				for (int x = 0; x < OUTPUT_W_SIZE; ++x) {
					INDEX output_frame = (input_frame*OUTPUT_H_SIZE + y) * OUTPUT_W_SIZE + x;
					for (int c = 0; c < INPUT_C_SIZE; ++c) {
						for (int fy = 0; fy < FILTER_H_SIZE; ++fy) {
							for (int fx = 0; fx < FILTER_W_SIZE; ++fx) {
								int ix = x + fx;
								int iy = y + fy;
								int output_node = GetOutputNode(c, fy, fx);
								int input_node = GetInputNode(c, iy, ix);
								ET err = in_err_buf.Get<ET>(input_frame, input_node);
								in_err_buf.Set<ET>(input_frame, input_node, err + out_err_buf.Get<ET>(output_frame, output_node));
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