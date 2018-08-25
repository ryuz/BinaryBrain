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

#include "NeuralNetLayer.h"

namespace bb {


// NeuralNetの抽象クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNetBinaryFilter : public NeuralNetLayer<T, INDEX>
{
protected:
	NeuralNetLayer<T, INDEX>* m_filter_net;
	INDEX			m_mux_size;
	INDEX			m_frame_size;
	int				m_input_h_size;
	int				m_input_w_size;
	int				m_input_c_size;
	int				m_filter_h_size;
	int				m_filter_w_size;
	int				m_y_step;
	int				m_x_step;
	int				m_output_h_size;
	int				m_output_w_size;
	int				m_output_c_size;

public:
	NeuralNetBinaryFilter() {}
	
	NeuralNetBinaryFilter(NeuralNetLayer<T, INDEX>* filter_net, INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX output_c_size, INDEX filter_h_size, INDEX filter_w_size, INDEX y_step, INDEX x_step, INDEX mux_size, INDEX batch_size = 1)
	{
		Setup(filter_net, input_c_size, input_h_size, input_w_size, output_c_size, filter_h_size, filter_w_size, y_step, x_step, mux_size, batch_size);
	}
	
	~NeuralNetBinaryFilter() {}		// デストラクタ
	
	void Setup(NeuralNetLayer<T, INDEX>* filter_net, INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX output_c_size, INDEX filter_h_size, INDEX filter_w_size, INDEX y_step, INDEX x_step, INDEX mux_size, INDEX batch_size = 1)
	{
		m_filter_net = filter_net;
		m_mux_size = mux_size;
		m_frame_size = batch_size * mux_size;
		m_input_c_size = (int)input_c_size;
		m_input_h_size = (int)input_h_size;
		m_input_w_size = (int)input_w_size;
		m_filter_h_size = (int)filter_h_size;
		m_filter_w_size = (int)filter_w_size;
		m_y_step = (int)y_step;
		m_x_step = (int)x_step;
		m_output_c_size = (int)output_c_size;
		m_output_h_size = ((m_input_h_size - m_filter_h_size + 1) + (m_y_step - 1)) / m_y_step;
		m_output_w_size = ((m_input_w_size - m_filter_w_size + 1) + (m_x_step - 1)) / m_x_step;
	}
	
	void SetBatchSize(INDEX batch_size) { m_frame_size = batch_size * m_mux_size; }
	
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
	}
	
	void Backward(void)
	{
	}

	void Update(double learning_rate)
	{
	}

};


}