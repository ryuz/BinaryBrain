// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>
#include "NeuralNetLayer.h"
#include "NeuralNetConvExpand.h"
#include "NeuralNetConvCollapse.h"


namespace bb {


// 入力数制限Affine Binary Connect版
template <int N = 6, typename T = float, typename INDEX = size_t>
class NeuralNetConvolutionPack : public NeuralNetLayer<T, INDEX>
{
protected:
	// 3層で構成
	NeuralNetConvExpand<T, INDEX>		m_expand;
	NeuralNetLayer<T, INDEX>*			m_layer;
	NeuralNetConvCollapse<T, INDEX>		m_collapse;
	
	INDEX	m_expand_size = 1;

public:
	NeuralNetConvolutionPack() {}

	NeuralNetConvolutionPack(NeuralNetLayer<T, INDEX>* layer, INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX output_c_size, INDEX filter_h_size, INDEX filter_w_size)
		: m_expand(input_c_size, input_h_size, input_w_size, filter_h_size, filter_w_size),
		m_collapse(output_c_size, input_h_size - filter_h_size + 1, input_w_size - filter_w_size + 1)
	{
		m_layer = layer;

		m_expand_size = (input_h_size - filter_h_size + 1) * (input_w_size - filter_w_size + 1);
	}

	~NeuralNetConvolutionPack() {}
	
	void InitializeCoeff(std::uint64_t seed)
	{
		m_layer->InitializeCoeff(seed);
	}
	
	void  SetBinaryMode(bool enable)
	{
		m_expand.SetBinaryMode(enable);
		m_layer->SetBinaryMode(enable);
		m_collapse.SetBinaryMode(enable);
	}

	int   GetNodeInputSize(INDEX node) const { return m_affine.GetNodeInputSize(node); }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { m_affine.SetNodeInput(node, input_index, input_node); }
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_affine.GetNodeInput(node, input_index); }

	void  SetMuxSize(INDEX mux_size)
	{
		m_expand.SetMuxSize(mux_size);
		m_layer->SetMuxSize(mux_size);
		m_collapse.SetMuxSize(mux_size);
	}
	
	void  SetBatchSize(INDEX batch_size) {
		m_expand.SetBatchSize(batch_size);
		m_layer->SetBatchSize(batch_size * m_expand_size);
		m_collapse.SetBatchSize(batch_size);

		m_expand.SetOutputSignalBuffer(m_expand.CreateOutputSignalBuffer());
		m_expand.SetOutputErrorBuffer(m_expand.CreateOutputErrorBuffer());
		m_layer->SetInputSignalBuffer(m_expand.GetOutputSignalBuffer());
		m_layer->SetInputErrorBuffer(m_expand.GetOutputErrorBuffer());

		m_layer->SetOutputSignalBuffer(m_layer->CreateOutputSignalBuffer());
		m_layer->SetOutputErrorBuffer(m_layer->CreateOutputErrorBuffer());
		m_collapse.SetInputSignalBuffer(m_layer->GetOutputSignalBuffer());
		m_collapse.SetInputErrorBuffer(m_layer->GetOutputErrorBuffer());
	}

	
	// 入出力バッファ
	void  SetInputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_expand.SetInputSignalBuffer(buffer); }
	void  SetOutputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_collapse.SetOutputSignalBuffer(buffer); }
	void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_expand.SetInputErrorBuffer(buffer); }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_collapse.SetOutputErrorBuffer(buffer); }

	const NeuralNetBuffer<T, INDEX>& GetInputSignalBuffer(void) const { return m_expand.GetInputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputSignalBuffer(void) const { return m_collapse.GetOutputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) const { return m_expand.GetInputErrorBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) const { return m_collapse.GetOutputErrorBuffer(); }


	INDEX GetInputFrameSize(void) const { return m_expand.GetInputFrameSize(); }
	INDEX GetOutputFrameSize(void) const { return m_collapse.GetOutputFrameSize(); }

	INDEX GetInputNodeSize(void) const { return m_expand.GetInputNodeSize(); }
	INDEX GetOutputNodeSize(void) const { return m_collapse.GetOutputNodeSize(); }

	int   GetInputSignalDataType(void) const { return m_expand.GetInputSignalDataType(); }
	int   GetInputErrorDataType(void) const { return m_expand.GetInputErrorDataType(); }
	int   GetOutputSignalDataType(void) const { return m_collapse.GetOutputSignalDataType(); }
	int   GetOutputErrorDataType(void) const { return m_collapse.GetOutputErrorDataType(); }


public:

	void Forward(bool train = true)
	{
		m_expand.Forward(train);
		m_layer->Forward(train);
		m_collapse.Forward(train);
	}

	void Backward(void)
	{
		m_collapse.Backward();
		m_layer->Backward();
		m_expand.Backward();
	}

	void Update(double learning_rate)
	{
		m_expand.Update(learning_rate);
		m_layer->Update(learning_rate);
		m_collapse.Update(learning_rate);
	}

};


}