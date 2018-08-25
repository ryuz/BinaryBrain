// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>
#include "NeuralNetLayer.h"


namespace bb {

// NeuralNetLayerのグループ化
template <typename T = float, typename INDEX = size_t>
class NeuralNetGroup : public NeuralNetLayer<T, INDEX>
{
protected:
	typedef	NeuralNetLayer<T, INDEX>	LAYER;
	
	std::vector< LAYER* > m_layers;

	std::vector< NeuralNetBuffer<T, INDEX> > m_value_buffers;
	std::vector< NeuralNetBuffer<T, INDEX> > m_error_buffers;

	LAYER*				m_firstLayer;
	LAYER*				m_lastLayer;

	int					m_feedback_layer = -1;

public:
	// コンストラクタ
	NeuralNetGroup()
	{
	}

	// デストラクタ
	~NeuralNetGroup() {
	}

	INDEX GetInputFrameSize(void) const { return m_firstLayer->GetInputFrameSize(); }
	INDEX GetInputNodeSize(void) const { return m_firstLayer->GetInputNodeSize(); }
	int   GetInputValueDataType(void) const { return m_firstLayer->GetInputValueDataType(); }
	int   GetInputErrorDataType(void) const { return m_firstLayer->GetInputErrorDataType(); }
	INDEX GetOutputFrameSize(void) const { return m_lastLayer->GetOutputFrameSize(); }
	INDEX GetOutputNodeSize(void) const  { return m_lastLayer->GetOutputNodeSize(); }
	int   GetOutputValueDataType(void) const { return m_lastLayer->GetOutputValueDataType(); }
	int   GetOutputErrorDataType(void) const { return m_lastLayer->GetOutputErrorDataType(); }
	
	void  SetInputValueBuffer(NeuralNetBuffer<T, INDEX> buffer)
	{
		NeuralNetLayer<T, INDEX>::SetInputValueBuffer(buffer);
		m_firstLayer->SetInputValueBuffer(buffer);
	}

	void  SetOutputValueBuffer(NeuralNetBuffer<T, INDEX> buffer)
	{
		NeuralNetLayer<T, INDEX>::SetOutputValueBuffer(buffer);
		m_lastLayer->SetOutputValueBuffer(buffer);
	}

	void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer)
	{
		NeuralNetLayer<T, INDEX>::SetInputErrorBuffer(buffer);
		m_firstLayer->SetInputErrorBuffer(buffer);
	}

	void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer)
	{
		NeuralNetLayer<T, INDEX>::SetOutputErrorBuffer(buffer);
		m_lastLayer->SetOutputErrorBuffer(buffer);
	}
	
	void AddLayer(LAYER* layer)
	{
		if (m_layers.empty()) {
			m_firstLayer = layer;
		}
		m_layers.push_back(layer);
		m_lastLayer = layer;
	}

	void SetBatchSize(INDEX batch_size)
	{
		for (auto layer : m_layers) {
			layer->SetBatchSize(batch_size);
		}

		return SetupBuffer();
	}

	virtual void Forward(INDEX start_layer)
	{
		INDEX layer_size = m_layers.size();

		for (INDEX layer = start_layer; layer < layer_size; ++layer) {
			m_layers[layer]->Forward();
		}
	}
	
	void Forward(void)
	{
		Forward(0);
	}

	void Backward(void)
	{
		for (auto layer = m_layers.rbegin(); layer != m_layers.rend(); ++layer) {
			(*layer)->Backward();
		}
	}

	void Update(double learning_rate)
	{
		for (auto layer = m_layers.begin(); layer != m_layers.end(); ++layer) {
			(*layer)->Update(learning_rate);
		}
	}

	bool Feedback(std::vector<T>& loss)
	{
		if (m_feedback_layer < 0) {
			m_feedback_layer = (int)m_layers.size() - 1;	// 初回
		}

		while (m_feedback_layer >= 0) {
			if (m_layers[m_feedback_layer]->Feedback(loss)) {
				Forward(m_feedback_layer + 1);
				return true;
			}
			--m_feedback_layer;
		}

		return false;
	}


protected:
	void SetupBuffer(void)
	{
		if (m_layers.empty()) {
			assert(0);
			return;
		}

		// 整合性確認
		for (size_t i = 0; i < m_layers.size()-1; ++i) {
			if (m_layers[i]->GetOutputFrameSize() != m_layers[i+1]->GetInputFrameSize()) {
				assert(0);
				return;
			}
			if (m_layers[i]->GetOutputNodeSize() != m_layers[i+1]->GetInputNodeSize()) {
				std::cout << "node size mismatch" << std::endl;
				std::cout << "layer[" << i - 1 << "] : output node = : " << m_layers[i - 1]->GetOutputNodeSize() << std::endl;
				std::cout << "layer[" << i << "] : input node = : " << m_layers[i]->GetInputNodeSize() << std::endl;
				assert(0);
				return;
			}
			if (m_layers[i]->GetOutputValueDataType() != m_layers[i+1]->GetInputValueDataType()) {
				assert(0);
				return;
			}
			if (m_layers[i]->GetOutputErrorDataType() != m_layers[i+1]->GetInputErrorDataType()) {
				assert(0);
				return;
			}
		}

		// メモリ再確保
		m_value_buffers.clear();
		m_error_buffers.clear();

		for (size_t i = 0; i < m_layers.size()-1; ++i) {
			// バッファ生成
			m_value_buffers.push_back(m_layers[i]->CreateOutputValueBuffer());
			m_error_buffers.push_back(m_layers[i]->CreateOutputErrorBuffer());
			
			// バッファ設定
			m_layers[i]->SetOutputValueBuffer(m_value_buffers[i]);
			m_layers[i]->SetOutputErrorBuffer(m_error_buffers[i]);
			m_layers[i + 1]->SetInputValueBuffer(m_value_buffers[i]);
			m_layers[i + 1]->SetInputErrorBuffer(m_error_buffers[i]);
		}
	}
};


}