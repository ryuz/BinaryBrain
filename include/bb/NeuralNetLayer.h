// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <vector>
#include "NeuralNetBuffer.h"


namespace bb {


// NeuralNetの抽象クラス
template <typename T=float, typename INDEX = size_t>
class NeuralNetLayer
{
public:
	// 基本機能
	virtual ~NeuralNetLayer() {}											// デストラクタ

	virtual void  Resize(std::vector<INDEX> size) {};						// サイズ設定
	virtual void  InitializeCoeff(std::uint64_t seed) {}					// 内部係数の乱数初期化
	
	virtual INDEX GetInputFrameSize(void) const = 0;						// 入力のフレーム数
	virtual INDEX GetInputNodeSize(void) const = 0;							// 入力のノード数
	virtual INDEX GetOutputFrameSize(void) const = 0;						// 出力のフレーム数
	virtual INDEX GetOutputNodeSize(void) const = 0;						// 出力のノード数
	virtual int   GetInputValueDataType(void) const = 0;					// 入力値のサイズ
	virtual int   GetInputErrorDataType(void) const = 0;					// 出力値のサイズ
	virtual int   GetOutputValueDataType(void) const = 0;					// 入力値のサイズ
	virtual int   GetOutputErrorDataType(void) const = 0;					// 入力値のサイズ

	virtual void  SetMuxSize(INDEX mux_size) {}								// 多重化サイズの設定
	virtual void  SetBatchSize(INDEX batch_size) = 0;						// バッチサイズの設定
	virtual	void  Forward(void) = 0;										// 予測
	virtual	void  Backward(void) = 0;										// 誤差逆伝播
	virtual	void  Update(double learning_rate) {};							// 学習
	virtual	bool  Feedback(const std::vector<double>& loss) { return false; }	// 直接フィードバック
	

protected:
	// バッファ情報
	NeuralNetBuffer<T, INDEX>	m_input_value_buffer;
	NeuralNetBuffer<T, INDEX>	m_output_value_buffer;
	NeuralNetBuffer<T, INDEX>	m_input_error_buffer;
	NeuralNetBuffer<T, INDEX>	m_output_error_buffer;

public:
	// バッファ設定
	virtual void  SetInputValueBuffer(NeuralNetBuffer<T, INDEX> buffer)  { m_input_value_buffer = buffer; }
	virtual void  SetOutputValueBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_output_value_buffer = buffer; }
	virtual void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer)  { m_input_error_buffer = buffer; }
	virtual void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_output_error_buffer = buffer; }
	
	// バッファ取得
	NeuralNetBuffer<T, INDEX>& GetInputValueBuffer(void) { return m_input_value_buffer; }
	NeuralNetBuffer<T, INDEX>& GetOutputValueBuffer(void) { return m_output_value_buffer; }
	NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) { return m_input_error_buffer; }
	NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) { return m_output_error_buffer; }

	// バッファ生成補助
	NeuralNetBuffer<T, INDEX> CreateInputValueBuffer(void) { 
		return NeuralNetBuffer<T, INDEX>(GetInputFrameSize(), GetInputNodeSize(), GetInputValueDataType());
	}
	NeuralNetBuffer<T, INDEX> CreateOutputValueBuffer(void) {
		return NeuralNetBuffer<T, INDEX>(GetOutputFrameSize(), GetOutputNodeSize(), GetOutputValueDataType());
	}
	NeuralNetBuffer<T, INDEX> CreateInputErrorBuffer(void) {
		return NeuralNetBuffer<T, INDEX>(GetInputFrameSize(), GetInputNodeSize(), GetInputErrorDataType());
	}
	NeuralNetBuffer<T, INDEX> CreateOutputErrorBuffer(void) {
		return NeuralNetBuffer<T, INDEX>(GetOutputFrameSize(), GetOutputNodeSize(), GetOutputErrorDataType());
	}
};


}