

#pragma once

#include <vector>
#include "NeuralNetBuffer.h"


// NeuralNetの抽象クラス
template <typename T=float, typename INDEX = size_t>
class NeuralNetLayer
{
public:
	// 基本機能
	virtual ~NeuralNetLayer() {}									// デストラクタ

//	virtual void  SetInputValuePtr(const void* ptr) = 0;			// 入力側値アドレス設定
//	virtual void  SetOutputValuePtr(void* ptr) = 0;					// 出力側値アドレス設定
//	virtual void  SetOutputErrorPtr(const void* ptr) = 0;			// 出力側誤差アドレス設定
//	virtual void  SetInputErrorPtr(void* ptr) = 0;					// 入力側誤差アドレス設定

	virtual INDEX GetInputFrameSize(void) const = 0;				// 入力のフレーム数
	virtual INDEX GetInputNodeSize(void) const = 0;					// 入力のノード数
	virtual INDEX GetOutputFrameSize(void) const = 0;				// 出力のフレーム数
	virtual INDEX GetOutputNodeSize(void) const = 0;				// 出力のノード数
	virtual int   GetInputValueBitSize(void) const = 0;				// 入力値のサイズ
	virtual int   GetInputErrorBitSize(void) const = 0;				// 出力値のサイズ
	virtual int   GetOutputValueBitSize(void) const = 0;			// 入力値のサイズ
	virtual int   GetOutputErrorBitSize(void) const = 0;			// 入力値のサイズ

	virtual void  SetBatchSize(INDEX batch_size) = 0;				// バッチサイズの設定
	virtual	void  Forward(void) = 0;								// 予測
	virtual	void  Backward(void) = 0;								// 誤差逆伝播
	virtual	void  Update(double learning_rate) {};					// 学習

	virtual	bool  Feedback(const std::vector<T>& loss) { return false; }			// 直接フィードバック


protected:
	NeuralNetBuffer<T, INDEX>	m_input_value_buffer;
	NeuralNetBuffer<T, INDEX>	m_output_value_buffer;
	NeuralNetBuffer<T, INDEX>	m_input_error_buffer;
	NeuralNetBuffer<T, INDEX>	m_output_error_buffer;

public:
	
	// バッファ設定
	void  SetInputValueBuffer(NeuralNetBuffer<T, INDEX> buffer)  { m_input_value_buffer = buffer; }
	void  SetOutputValueBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_output_value_buffer = buffer; }
	void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer)  { m_input_error_buffer = buffer; }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_output_error_buffer = buffer; }
	
	// バッファ取得
	NeuralNetBuffer<T, INDEX>& GetInputValueBuffer(void) { return m_input_value_buffer; }
	NeuralNetBuffer<T, INDEX>& GetOutputValueBuffer(void) { return m_output_value_buffer; }
	NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) { return m_input_error_buffer; }
	NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) { return m_output_error_buffer; }

	// バッファ生成補助
	NeuralNetBuffer<T, INDEX> CreateInputValueBuffer(void) { 
		return NeuralNetBuffer<T, INDEX>(GetInputFrameSize(), GetInputNodeSize(), GetInputValueBitSize()); 
	}
	NeuralNetBuffer<T, INDEX> CreateOutputValueBuffer(void) {
		return NeuralNetBuffer<T, INDEX>(GetOutputFrameSize(), GetOutputNodeSize(), GetOutputValueBitSize());
	}
	NeuralNetBuffer<T, INDEX> CreateInputErrorBuffer(void) {
		return NeuralNetBuffer<T, INDEX>(GetInputFrameSize(), GetInputNodeSize(), GetInputErrorBitSize());
	}
	NeuralNetBuffer<T, INDEX> CreateOutputErrorBuffer(void) {
		return NeuralNetBuffer<T, INDEX>(GetOutputFrameSize(), GetOutputNodeSize(), GetOutputErrorBitSize());
	}

};

