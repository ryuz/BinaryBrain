

#pragma once

#include <vector>
#include <intrin.h>
#include "NeuralNetLayer.h"

// NeuralNetの抽象クラス
template <typename T=float, typename INDEX=size_t>
class NeuralNet
{
protected:
	typedef	NeuralNetLayer<INDEX>	LAYER;

	std::vector< LAYER* > m_layers;
	std::vector< void* > m_values;
	std::vector< void* > m_errors;
	LAYER*				m_firstLayer;
	LAYER*				m_lastLayer;

public:
	// コンストラクタ
	NeuralNet()
	{
	}

	// デストラクタ
	~NeuralNet() {
		ClearBuffer();
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
	
protected:
	size_t CalcBufferSize(INDEX frame_size, INDEX node_size, int bit_size)
	{
		size_t mm256_size = ((frame_size * bit_size)) + 255 / 256;
		return 32 * mm256_size * node_size;
	}

	void ClearBuffer(void) {
		for (auto v : m_values) { if (v != nullptr) { _mm_free(v); } }
		m_values.clear();

		for (auto e : m_errors) { if (e != nullptr) { _mm_free(e); } }
		m_errors.clear();
	}

	bool SetupBuffer(void)
	{
		if (m_layers.empty()) {
			return false;
		}

		// 整合性確認
		for (size_t i = 1; i < m_layers.size(); ++i) {
			if (m_layers[i - 1]->GetOutputFrameSize() != m_layers[i]->GetInputFrameSize()) {
				return false;
			}
			if (m_layers[i - 1]->GetOutputNodeSize() != m_layers[i]->GetInputNodeSize()) {
				return false;
			}
			if (m_layers[i - 1]->GetOutputValueBitSize() != m_layers[i]->GetInputValueBitSize()) {
				return false;
			}
			if (m_layers[i - 1]->GetOutputErrorBitSize() != m_layers[i]->GetInputErrorBitSize()) {
				return false;
			}
		}

		// メモリ再確保
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
		for ( int i = 0; i < (int)input_vector.size(); i++ ) {
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
					ld.lut.reserve((size_t)1<<num);
					for (int i = 0; i < (1<<num); i++) {
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

