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
#include "NeuralNetLayer.h"


namespace bb {

// NeuralNetの抽象クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNet
{
protected:
	typedef	NeuralNetLayer<T, INDEX>	LAYER;

	std::vector< LAYER* > m_layers;
	//	std::vector< void* > m_values;
	//	std::vector< void* > m_errors;

	std::vector< NeuralNetBuffer<T, INDEX> > m_value_buffers;
	std::vector< NeuralNetBuffer<T, INDEX> > m_error_buffers;

	LAYER*				m_firstLayer;
	LAYER*				m_lastLayer;

public:
	// コンストラクタ
	NeuralNet()
	{
	}

	// デストラクタ
	~NeuralNet() {
		//		ClearBuffer();
	}

	void AddLayer(LAYER* layer)
	{
		if (m_layers.empty()) {
			m_firstLayer = layer;
		}
		m_layers.push_back(layer);
		m_lastLayer = layer;
	}

	bool SetBatchSize(INDEX batch_size)
	{
		for (auto layer : m_layers) {
			layer->SetBatchSize(batch_size);
		}

		return SetupBuffer();
	}

	void Forward(INDEX start_layer = 0)
	{
		INDEX layer_size = m_layers.size();

		for (INDEX layer = start_layer; layer < layer_size; ++layer) {
			m_layers[layer]->Forward();
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


	void SetInputValue(INDEX frame, INDEX node, T value) {
		//		T* buf = (T*)m_values.front();
		//		INDEX stride = m_firstLayer->GetInputFrameSize();
		//		buf[node*stride + frame] = value;

		//		NeuralNetBufferAccessor<T, INDEX>* acc = m_firstLayer->GetInputValueBuffer().GetAccessor();
		//		acc->SetReal(frame, node, value);
		return m_firstLayer->GetInputValueBuffer().SetReal(frame, node, value);
	}

	void SetInputValue(INDEX frame, std::vector<T> values) {
		for (INDEX node = 0; node < (INDEX)values.size(); ++node) {
			SetInputValue(frame, node, values[node]);
		}
	}

	T GetOutputValue(INDEX frame, INDEX node) {
		//		T* buf = (T*)m_values.back();
		//		INDEX stride = m_lastLayer->GetOutputFrameSize();
		//		return buf[node*stride + frame];

		//		NeuralNetBufferAccessor<T, INDEX>* acc = m_lastLayer->GetOutputValueBuffer().GetAccessor();
		//		return acc->GetReal(frame, node);

		return m_lastLayer->GetOutputValueBuffer().GetReal(frame, node);
	}

	std::vector<T> GetOutputValue(INDEX frame) {
		std::vector<T> values(m_lastLayer->GetOutputNodeSize());
		//		T* buf = (T*)m_values.back();
		for (INDEX node = 0; node < (INDEX)values.size(); ++node) {
			values[node] = GetOutputValue(frame, node);
		}
		return values;
	}

	void SetOutputError(INDEX frame, INDEX node, T error) {
		//		T* buf = (T*)m_errors.back();
		//		INDEX stride = m_lastLayer->GetOutputFrameSize();
		//		buf[node*stride + frame] = error;

		m_lastLayer->GetOutputErrorBuffer().SetReal(frame, node, error);
	}

	void SetOutputError(INDEX frame, std::vector<T> errors) {
		for (INDEX node = 0; node < (INDEX)errors.size(); ++node) {
			SetOutputError(frame, node, errors[node]);
		}
	}

protected:
	int		m_feedback_layer = -1;

public:
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
	size_t CalcBufferSize(INDEX frame_size, INDEX node_size, int bit_size)
	{
		size_t mm256_size = ((frame_size * bit_size) + 255) / 256;
		return 32 * mm256_size * node_size;
	}

	/*
	void ClearBuffer(void) {
		for (auto v : m_values) { if (v != nullptr) { _mm_free(v); } }
		m_values.clear();

		for (auto e : m_errors) { if (e != nullptr) { _mm_free(e); } }
		m_errors.clear();
	}
	*/

	bool SetupBuffer(void)
	{
		if (m_layers.empty()) {
			assert(0);
			return false;
		}

		// 整合性確認
		for (size_t i = 1; i < m_layers.size(); ++i) {
			if (m_layers[i - 1]->GetOutputFrameSize() != m_layers[i]->GetInputFrameSize()) {
				assert(0);
				return false;
			}
			if (m_layers[i - 1]->GetOutputNodeSize() != m_layers[i]->GetInputNodeSize()) {
				std::cout << "node size mismatch" << std::endl;
				std::cout << "layer[" << i - 1 << "] : output node = : " << m_layers[i - 1]->GetOutputNodeSize() << std::endl;
				std::cout << "layer[" << i << "] : input node = : " << m_layers[i]->GetInputNodeSize() << std::endl;
				assert(0);
				return false;
			}
			if (m_layers[i - 1]->GetOutputValueDataType() != m_layers[i]->GetInputValueDataType()) {
				assert(0);
				return false;
			}
			if (m_layers[i - 1]->GetOutputErrorDataType() != m_layers[i]->GetInputErrorDataType()) {
				assert(0);
				return false;
			}
		}

		// メモリ再確保
		m_value_buffers.clear();
		m_error_buffers.clear();

		// バッファ生成
		m_value_buffers.push_back(m_firstLayer->CreateInputValueBuffer());
		m_error_buffers.push_back(m_firstLayer->CreateInputErrorBuffer());
		for (auto layer : m_layers) {
			m_value_buffers.push_back(layer->CreateOutputValueBuffer());
			m_error_buffers.push_back(layer->CreateOutputErrorBuffer());
		}

		// バッファ設定
		for (size_t i = 0; i < m_layers.size(); ++i) {
			m_layers[i]->SetInputValueBuffer(m_value_buffers[i]);
			m_layers[i]->SetInputErrorBuffer(m_error_buffers[i]);
			m_layers[i]->SetOutputValueBuffer(m_value_buffers[i + 1]);
			m_layers[i]->SetOutputErrorBuffer(m_error_buffers[i + 1]);
		}

		/*
		ClearBuffer();

		m_values.push_back(
			_mm_malloc(
				CalcBufferSize(
					m_firstLayer->GetInputFrameSize(),
					m_firstLayer->GetInputNodeSize(),
					m_firstLayer->GetInputValueBitSize()),
				32));

		m_errors.push_back(
			_mm_malloc(
				CalcBufferSize(
					m_firstLayer->GetInputFrameSize(),
					m_firstLayer->GetInputNodeSize(),
					m_firstLayer->GetInputErrorBitSize()),
				32));

		for (auto layer : m_layers) {
			m_values.push_back(
				_mm_malloc(
					CalcBufferSize(
						layer->GetOutputFrameSize(),
						layer->GetOutputNodeSize(),
						layer->GetOutputValueBitSize()),
					32));

			m_errors.push_back(
				_mm_malloc(
					CalcBufferSize(
						layer->GetOutputFrameSize(),
						layer->GetOutputNodeSize(),
						layer->GetOutputErrorBitSize()),
					32));
		}

		// バッファ設定
		for (size_t i = 0; i < m_layers.size(); ++i) {
			m_layers[i]->SetInputValuePtr(m_values[i]);
			m_layers[i]->SetInputErrorPtr(m_errors[i]);
			m_layers[i]->SetOutputValuePtr(m_values[i+1]);
			m_layers[i]->SetOutputErrorPtr(m_errors[i+1]);
		}
		*/

		return true;
	}




#if 0
	virtual INDEX GetInputFrameSize(void) const = 0;	// 入力のフレーム数
	virtual INDEX GetInputNodeSize(void) const = 0;		// 入力のノード数
	virtual INDEX GetOutputFrameSize(void) const = 0;	// 出力のフレーム数
	virtual INDEX GetOutputNodeSize(void) const = 0;	// 出力のノード数

	virtual void  SetInputValue(INDEX frame, INDEX node, T value) = 0;
	virtual T     GetInputValue(INDEX frame, INDEX node) = 0;
	virtual void  SetInputError(INDEX frame, INDEX node, ET error) = 0;
	virtual ET    GetInputError(INDEX frame, INDEX node) = 0;

	virtual void  SetOutput(INDEX frame, INDEX node, T value) = 0;
	virtual T     GetOutput(INDEX frame, INDEX node) = 0;
	virtual void  SetOutputError(INDEX frame, INDEX node, ET error) = 0;
	virtual ET    GetOutputError(INDEX frame, INDEX node) = 0;

	virtual	void  Forward(void) = 0;
	virtual	void  Backward(void) = 0;

	virtual INDEX  GetLayerSize(void) const = 0;			// 
	virtual INDEX  GetNodeSize(int layer) const = 0;
	virtual INDEX  GetEdgeSize(int layer, int node) const = 0;

	virtual void   SetConnection(int layer, int node, int input_num, int input_node) = 0;
	virtual int    GetConnection(int layer, int node, int input_num) const = 0;

	virtual void CalcForward(int layer = 0) = 0;

	virtual bool GetValue(int layer, int node) const = 0;
	virtual void SetValue(int layer, int node, bool value) = 0;

	virtual bool GetInputValue(int layer, int node, int index) const = 0;

	virtual bool GetLutBit(int layer, int node, int bit) const = 0;
	virtual void SetLutBit(int layer, int node, int bit, bool value) = 0;

	virtual void InvertLut(int layer, int node)
	{
		int n = GetInputNum(layer, node);
		int size = (1 << n);
		for (int i = 0; i < size; i++) {
			SetLutBit(layer, node, i, !GetLutBit(layer, node, i));
		}
		SetValue(layer, node, !GetValue(layer, node));
	}

	void SetInput(std::vector<bool> input_vector)
	{
		for (int i = 0; i < (int)input_vector.size(); i++) {
			SetValue(0, i, input_vector[i]);
		}
	}

	std::vector<bool> GetOutput(void)
	{
		int layer = GetLayerNum() - 1;
		std::vector<bool> output_vector(GetNodeNum(layer));
		for (int i = 0; i < (int)output_vector.size(); i++) {
			output_vector[i] = GetValue(layer, i);
		}
		return output_vector;
	}

	int  GetInputLutIndex(int layer, int node) const
	{
		int num = GetInputNum(layer, node);

		int idx = 0;
		int bit = 1;
		for (int i = 0; i < num; i++) {
			idx |= GetInputValue(layer, node, i) ? bit : 0;
			bit <<= 1;
		}

		return idx;
	}


	// データエクスポート
	BinaryNetData ExportData(void) {
		BinaryNetData	bnd;
		for (int layer = 0; layer < GetLayerNum(); layer++) {
			if (layer == 0) {
				bnd.input_num = GetNodeNum(layer);
			}
			else {
				int node_num = GetNodeNum(layer);
				BinaryLayerData bld;
				bld.node.reserve(node_num);
				for (int node = 0; node < node_num; node++) {
					LutData ld;
					int num = GetInputNum(layer, node);
					ld.connect.reserve(num);
					for (int i = 0; i < num; i++) {
						ld.connect.push_back(GetConnection(layer, node, i));
					}
					ld.lut.reserve((size_t)1 << num);
					for (int i = 0; i < (1 << num); i++) {
						ld.lut.push_back(GetLutBit(layer, node, i));
					}
					bld.node.push_back(ld);
				}
				bnd.layer.push_back(bld);
			}
		}

		return bnd;
	}
#endif
};


template <typename T = float, typename INDEX = int>
INDEX argmax(std::vector<T> vec)
{
	auto maxIt = std::max_element(vec.begin(), vec.end());
	return (INDEX)std::distance(vec.begin(), maxIt);
}


}