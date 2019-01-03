// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>

#include "bb/NeuralNetFilter2d.h"
#include "bb/NeuralNetConvolutionIm2Col.h"
#include "bb/NeuralNetConvolutionCol2Im.h"


namespace bb {


// 入力数制限Affine Binary Connect版
template <typename ST = float, typename ET = float, typename T = float>
class NeuralNetLoweringConvolution : public NeuralNetFilter2d<T>
{
protected:
	// 3層で構成
	NeuralNetConvolutionIm2Col<ST, ET, T>	m_im2col;
	NeuralNetLayer<T>*						m_layer;
	NeuralNetConvolutionCol2Im<ST, ET, T>	m_col2im;
	
	INDEX	m_batch_size = 0;
	INDEX	m_im2col_size = 1;

public:
	NeuralNetLoweringConvolution() {}

	NeuralNetLoweringConvolution(NeuralNetLayer<T>* layer, INDEX input_c_size, INDEX input_h_size, INDEX input_w_size, INDEX output_c_size, INDEX filter_h_size, INDEX filter_w_size)
		: m_im2col(input_c_size, input_h_size, input_w_size, filter_h_size, filter_w_size),
		m_col2im(output_c_size, input_h_size - filter_h_size + 1, input_w_size - filter_w_size + 1)
	{
		m_layer = layer;

		m_im2col_size = (input_h_size - filter_h_size + 1) * (input_w_size - filter_w_size + 1);
	}

	~NeuralNetLoweringConvolution() {}

	std::string GetClassName(void) const { return "NeuralNetLoweringConvolution"; }

	int GetInputChannel(void)  const { return m_im2col.GetInputChannel(); }
	int GetInputHeight(void)   const { return m_im2col.GetInputHeight(); }
	int GetInputWidth(void)    const { return m_im2col.GetInputWidth(); }
	int GetFilterHeight(void)  const { return m_im2col.GetFilterHeight(); }
	int	GetFilterWidth(void)   const { return m_im2col.GetFilterWidth(); }
	int GetOutputChannel(void) const { return m_col2im.GetChannel(); }
	int	GetOutputHeight(void)  const { return m_col2im.GetHeight(); }
	int	GetOutputWidth(void)   const { return m_col2im.GetWidth(); }

	void InitializeCoeff(std::uint64_t seed)
	{
		m_layer->InitializeCoeff(seed);
	}
	
	void  SetBinaryMode(bool enable)
	{
		m_im2col.SetBinaryMode(enable);
		m_layer->SetBinaryMode(enable);
		m_col2im.SetBinaryMode(enable);
	}

	void SetOptimizer(const NeuralNetOptimizer<T>* optimizer)
	{
		m_im2col.SetOptimizer(optimizer);
		m_layer->SetOptimizer(optimizer);
		m_col2im.SetOptimizer(optimizer);
	}

	NeuralNetLayer<T>* GetLayer(void) const {
		return m_layer;
	}

	int   GetNodeInputSize(INDEX node) const { return this->m_affine.GetNodeInputSize(node); }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { this->m_affine.SetNodeInput(node, input_index, input_node); }
	INDEX GetNodeInput(INDEX node, int input_index) const { return this->m_affine.GetNodeInput(node, input_index); }
	
	void  SetBatchSize(INDEX batch_size)
	{
		m_im2col.SetBatchSize(batch_size);
		m_layer->SetBatchSize(batch_size * m_im2col_size);
		m_col2im.SetBatchSize(batch_size);

		if (m_batch_size == batch_size) {
			return;
		}
		m_batch_size = batch_size;

		CheckConnection(m_im2col, *m_layer);
		CheckConnection(*m_layer, m_col2im);

		m_im2col.SetOutputSignalBuffer(m_im2col.CreateOutputSignalBuffer());
		m_im2col.SetOutputErrorBuffer(m_im2col.CreateOutputErrorBuffer());
		m_layer->SetInputSignalBuffer(m_im2col.GetOutputSignalBuffer());
		m_layer->SetInputErrorBuffer(m_im2col.GetOutputErrorBuffer());

		m_layer->SetOutputSignalBuffer(m_layer->CreateOutputSignalBuffer());
		m_layer->SetOutputErrorBuffer(m_layer->CreateOutputErrorBuffer());
		m_col2im.SetInputSignalBuffer(m_layer->GetOutputSignalBuffer());
		m_col2im.SetInputErrorBuffer(m_layer->GetOutputErrorBuffer());
	}

	
	// 入出力バッファ
	void  SetInputSignalBuffer(NeuralNetBuffer<T> buffer) { m_im2col.SetInputSignalBuffer(buffer); }
	void  SetOutputSignalBuffer(NeuralNetBuffer<T> buffer) { m_col2im.SetOutputSignalBuffer(buffer); }
	void  SetInputErrorBuffer(NeuralNetBuffer<T> buffer) { m_im2col.SetInputErrorBuffer(buffer); }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T> buffer) { m_col2im.SetOutputErrorBuffer(buffer); }

	const NeuralNetBuffer<T>& GetInputSignalBuffer(void) const { return m_im2col.GetInputSignalBuffer(); }
	const NeuralNetBuffer<T>& GetOutputSignalBuffer(void) const { return m_col2im.GetOutputSignalBuffer(); }
	const NeuralNetBuffer<T>& GetInputErrorBuffer(void) const { return m_im2col.GetInputErrorBuffer(); }
	const NeuralNetBuffer<T>& GetOutputErrorBuffer(void) const { return m_col2im.GetOutputErrorBuffer(); }


	INDEX GetInputFrameSize(void) const { return m_im2col.GetInputFrameSize(); }
	INDEX GetOutputFrameSize(void) const { return m_col2im.GetOutputFrameSize(); }

	INDEX GetInputNodeSize(void) const { return m_im2col.GetInputNodeSize(); }
	INDEX GetOutputNodeSize(void) const { return m_col2im.GetOutputNodeSize(); }

	int   GetInputSignalDataType(void) const { return m_im2col.GetInputSignalDataType(); }
	int   GetInputErrorDataType(void) const { return m_im2col.GetInputErrorDataType(); }
	int   GetOutputSignalDataType(void) const { return m_col2im.GetOutputSignalDataType(); }
	int   GetOutputErrorDataType(void) const { return m_col2im.GetOutputErrorDataType(); }


public:

	void Forward(bool train = true)
	{
		m_im2col.Forward(train);
		m_layer->Forward(train);
		m_col2im.Forward(train);
	}

	void Backward(void)
	{
		m_col2im.Backward();
		m_layer->Backward();
		m_im2col.Backward();
	}

	void Update(void)
	{
		m_im2col.Update();
		m_layer->Update();
		m_col2im.Update();
	}


	bool Feedback(const std::vector<double>& loss)
	{
		std::vector<double> exp_loss(loss.size() * m_im2col_size);
		for (size_t i = 0; i < loss.size(); ++i) {
			for (INDEX j = 0; j < m_im2col_size; ++j) {
				exp_loss[i*m_im2col_size + j] = loss[i];
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