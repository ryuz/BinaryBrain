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
	int   GetInputSignalDataType(void) const { return m_firstLayer->GetInputSignalDataType(); }
	int   GetInputErrorDataType(void) const { return m_firstLayer->GetInputErrorDataType(); }
	INDEX GetOutputFrameSize(void) const { return m_lastLayer->GetOutputFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return m_lastLayer->GetOutputNodeSize(); }
	int   GetOutputSignalDataType(void) const { return m_lastLayer->GetOutputSignalDataType(); }
	int   GetOutputErrorDataType(void) const { return m_lastLayer->GetOutputErrorDataType(); }
	
	// 入出力バッファ
	void  SetInputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_firstLayer->SetInputSignalBuffer(buffer); }
	void  SetOutputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_lastLayer->SetOutputSignalBuffer(buffer); }
	void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_firstLayer->SetInputErrorBuffer(buffer); }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_lastLayer->SetOutputErrorBuffer(buffer); }

	const NeuralNetBuffer<T, INDEX>& GetInputSignalBuffer(void) const  { return m_firstLayer->GetInputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputSignalBuffer(void) const { return m_lastLayer->GetOutputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) const { return m_firstLayer->GetInputErrorBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) const { return m_lastLayer->GetOutputErrorBuffer(); }

	
	// 内部バッファアクセス
	void SetInputSignalBuffer(INDEX layer, NeuralNetBuffer<T, INDEX> buf) { return m_layers[layer]->SetInputSignalBuffer(buf); }
	void SetInputErrorBuffer(INDEX layer, NeuralNetBuffer<T, INDEX> buf) { return m_layers[layer]->SetInputErrorBuffer(buf); }
	void SetOutputSignalBuffer(INDEX layer, NeuralNetBuffer<T, INDEX> buf) { return m_layers[layer]->SetOutputSignalBuffer(buf); }
	void SetOutputErrorBuffer(INDEX layer, NeuralNetBuffer<T, INDEX> buf) { return m_layers[layer]->SetOutputErrorBuffer(buf); }

	NeuralNetBuffer<T, INDEX> GetInputSignalBuffer(INDEX layer) const { return m_layers[layer]->GetInputSignalBuffer(); }
	NeuralNetBuffer<T, INDEX> GetInputErrorBuffer(INDEX layer) const { return m_layers[layer]->GetInputErrorBuffer(); }
	NeuralNetBuffer<T, INDEX> GetOutputSignalBuffer(INDEX layer) const { return m_layers[layer]->GetOutputSignalBuffer(); }
	NeuralNetBuffer<T, INDEX> GetOutputErrorBuffer(INDEX layer) const { return m_layers[layer]->GetOutputErrorBuffer(); }


	void AddLayer(LAYER* layer)
	{
		if (m_layers.empty()) {
			m_firstLayer = layer;
		}
		m_layers.push_back(layer);
		m_lastLayer = layer;
	}

	void SetMuxSize(INDEX mux_size)
	{
		for (auto layer : m_layers) {
			layer->SetMuxSize(mux_size);
		}
	}

	void SetBatchSize(INDEX batch_size)
	{
		for (auto layer : m_layers) {
			layer->SetBatchSize(batch_size);
		}

		return SetupBuffer();
	}

	virtual void Forward(bool train, INDEX start_layer)
	{
		INDEX layer_size = m_layers.size();

		for (INDEX layer = start_layer; layer < layer_size; ++layer) {
			m_layers[layer]->Forward(train);
		}
	}
	
	void Forward(bool train = true)
	{
		Forward(train, 0);
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

	bool Feedback(const std::vector<double>& loss)
	{
		if (m_feedback_layer < 0) {
			m_feedback_layer = (int)m_layers.size() - 1;	// 初回
		}

		while (m_feedback_layer >= 0) {
			if (m_layers[m_feedback_layer]->Feedback(loss)) {
				Forward(true, m_feedback_layer + 1);
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
			BB_ASSERT(0);
			return;
		}

		// 整合性確認
		for (size_t i = 0; i < m_layers.size()-1; ++i) {
			if (m_layers[i]->GetOutputFrameSize() != m_layers[i+1]->GetInputFrameSize()) {
				std::cout << "frame size mismatch" << std::endl;
				std::cout << "layer[" << i << "] " << m_layers[i]->GetLayerName() << " : output frame = : " << m_layers[i]->GetOutputFrameSize() << std::endl;
				std::cout << "layer[" << i + 1 << "] " << m_layers[i+1]->GetLayerName() << " : input frame = : " << m_layers[i + 1]->GetInputFrameSize() << std::endl;
				BB_ASSERT(0);
				return;
			}
			if (m_layers[i]->GetOutputNodeSize() != m_layers[i+1]->GetInputNodeSize()) {
				std::cout << "node size mismatch" << std::endl;
				std::cout << "layer[" << i << "] " << m_layers[i]->GetLayerName() << ": output node = : " << m_layers[i]->GetOutputNodeSize() << std::endl;
				std::cout << "layer[" << i+1 << "] " << m_layers[i+1]->GetLayerName() << ": input node = : " << m_layers[i+1]->GetInputNodeSize() << std::endl;
				BB_ASSERT(0);
				return;
			}
			if (m_layers[i]->GetOutputSignalDataType() != m_layers[i+1]->GetInputSignalDataType()) {
				std::cout << "data type size mismatch" << std::endl;
				std::cout << "layer[" << i << "] " << m_layers[i]->GetLayerName() << ": output data type = : " << m_layers[i]->GetOutputSignalDataType() << std::endl;
				std::cout << "layer[" << i + 1 << "] " << m_layers[i + 1]->GetLayerName() << ": data input type = : " << m_layers[i + 1]->GetInputSignalDataType() << std::endl;
				BB_ASSERT(0);
				return;
			}
			if (m_layers[i]->GetOutputErrorDataType() != m_layers[i+1]->GetInputErrorDataType()) {
				std::cout << "error type size mismatch" << std::endl;
				std::cout << "layer[" << i << "] " << m_layers[i]->GetLayerName() << ": output error type = : " << m_layers[i]->GetOutputErrorDataType() << std::endl;
				std::cout << "layer[" << i + 1 << "] " << m_layers[i + 1]->GetLayerName() << ": error input type = : " << m_layers[i + 1]->GetInputErrorDataType() << std::endl;
				BB_ASSERT(0);
				return;
			}
		}

		// メモリ再確保
		m_value_buffers.clear();
		m_error_buffers.clear();

		for (size_t i = 0; i < m_layers.size()-1; ++i) {
			// バッファ生成
			m_value_buffers.push_back(m_layers[i]->CreateOutputSignalBuffer());
			m_error_buffers.push_back(m_layers[i]->CreateOutputErrorBuffer());
			
			// バッファ設定
			m_layers[i]->SetOutputSignalBuffer(m_value_buffers[i]);
			m_layers[i]->SetOutputErrorBuffer(m_error_buffers[i]);
			m_layers[i + 1]->SetInputSignalBuffer(m_value_buffers[i]);
			m_layers[i + 1]->SetInputErrorBuffer(m_error_buffers[i]);
		}
	}

public:
	// Serialize
	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
		INDEX layer_size = (INDEX)m_layers.size();
		archive(cereal::make_nvp("layer_size", layer_size));
	}

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
		INDEX layer_size;
		archive(cereal::make_nvp("layer_size", layer_size));
		m_layers.resize(layer_size);
	}

	
	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		archive(cereal::make_nvp("NeuralNetGroup", *this));
		for (auto l : m_layers) {
			l->Save(archive);
		}
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		archive(cereal::make_nvp("NeuralNetGroup", *this));
		for (auto l : m_layers) {
			l->Load(archive);
		}
	}
	
};


}