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
template <typename ST = float, typename ET = float, typename T = float, typename INDEX = size_t>
class NeuralNetConvolutionPack : public NeuralNetLayer<T, INDEX>
{
protected:
	// 3層で構成
	NeuralNetConvExpand<ST, ET, T, INDEX>		m_expand;
	NeuralNetLayer<T, INDEX>*					m_layer;
	NeuralNetConvCollapse<ST, ET, T, INDEX>		m_collapse;
	
	INDEX	m_batch_size = 0;
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

	std::string GetClassName(void) const { return "NeuralNetConvolutionPack"; }

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

	int   GetNodeInputSize(INDEX node) const { return this->m_affine.GetNodeInputSize(node); }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { this->m_affine.SetNodeInput(node, input_index, input_node); }
	INDEX GetNodeInput(INDEX node, int input_index) const { return this->m_affine.GetNodeInput(node, input_index); }
	
	void  SetBatchSize(INDEX batch_size)
	{
		m_expand.SetBatchSize(batch_size);
		m_layer->SetBatchSize(batch_size * m_expand_size);
		m_collapse.SetBatchSize(batch_size);

		if (m_batch_size == batch_size) {
			return;
		}
		m_batch_size = batch_size;

		CheckConnection(m_expand, *m_layer);
		CheckConnection(*m_layer, m_collapse);

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

	void Update(void)
	{
		m_expand.Update();
		m_layer->Update();
		m_collapse.Update();
	}


	bool Feedback(const std::vector<double>& loss)
	{
		std::vector<double> exp_loss(loss.size() * m_expand_size);
		for (size_t i = 0; i < loss.size(); ++i) {
			for (INDEX j = 0; j < m_expand_size; ++j) {
				exp_loss[i*m_expand_size + j] = loss[i];
			}
		}
		return m_layer->Feedback(exp_loss);
	}

public:
	// Serialize
	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
	}

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
	}


	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		m_layer->Save(archive);
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		m_layer->Load(archive);
	}
};


}