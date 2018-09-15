// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <vector>
#include <intrin.h>
#include <assert.h>
#include "NeuralNetGroup.h"


namespace bb {


// NeuralNet 最上位構成用クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNet : public NeuralNetGroup<T, INDEX>
{
protected:
	NeuralNetBuffer<T, INDEX>	m_input_value_buffers;
	NeuralNetBuffer<T, INDEX>	m_input_error_buffers;
	NeuralNetBuffer<T, INDEX>	m_output_value_buffers;
	NeuralNetBuffer<T, INDEX>	m_output_error_buffers;

public:
	// コンストラクタ
	NeuralNet()
	{
	}

	// デストラクタ
	~NeuralNet() {
	}

	void SetBatchSize(INDEX batch_size)
	{
		// 親クラス呼び出し
		NeuralNetGroup<T, INDEX>::SetBatchSize(batch_size);

		// 入出力のバッファも準備
		m_input_value_buffers = m_firstLayer->CreateInputSignalBuffer();
		m_input_error_buffers = m_firstLayer->CreateInputErrorBuffer();
		m_output_value_buffers = m_lastLayer->CreateOutputSignalBuffer();
		m_output_error_buffers = m_lastLayer->CreateOutputErrorBuffer();
		m_firstLayer->SetInputSignalBuffer(m_input_value_buffers);
		m_firstLayer->SetInputErrorBuffer(m_input_error_buffers);
		m_lastLayer->SetOutputSignalBuffer(m_output_value_buffers);
		m_lastLayer->SetOutputErrorBuffer(m_output_error_buffers);
	}

	void Forward(bool train = true, INDEX start_layer = 0)
	{
		INDEX layer_size = m_layers.size();

		for (INDEX layer = start_layer; layer < layer_size; ++layer) {
			m_layers[layer]->Forward(train);
		}
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


	// 入出力データへのアクセス補助
	void SetInputSignal(INDEX frame, INDEX node, T value) {
		return m_firstLayer->GetInputSignalBuffer().SetReal(frame, node, value);
	}

	void SetInputSignal(INDEX frame, std::vector<T> values) {
		for (INDEX node = 0; node < (INDEX)values.size(); ++node) {
			SetInputSignal(frame, node, values[node]);
		}
	}

	T GetOutputSignal(INDEX frame, INDEX node) {
		return m_lastLayer->GetOutputSignalBuffer().GetReal(frame, node);
	}

	std::vector<T> GetOutputSignal(INDEX frame) {
		std::vector<T> values(m_lastLayer->GetOutputNodeSize());
		for (INDEX node = 0; node < (INDEX)values.size(); ++node) {
			values[node] = GetOutputSignal(frame, node);
		}
		return values;
	}

	void SetOutputError(INDEX frame, INDEX node, T error) {
		m_lastLayer->GetOutputErrorBuffer().SetReal(frame, node, error);
	}

	void SetOutputError(INDEX frame, std::vector<T> errors) {
		for (INDEX node = 0; node < (INDEX)errors.size(); ++node) {
			SetOutputError(frame, node, errors[node]);
		}
	}
};


}