// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <vector>

#include "cereal/types/array.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/json.hpp"

#include "bb/NeuralNetLayer.h"


namespace bb {


// NeuralNetのバッファ付き抽象クラス
template <typename T=float>
class NeuralNetLayerBuf : public NeuralNetLayer<T>
{
protected:
	// バッファ情報
	NeuralNetBuffer<T>	m_input_signal_buffer;
	NeuralNetBuffer<T>	m_output_signal_buffer;
	NeuralNetBuffer<T>	m_input_error_buffer;
	NeuralNetBuffer<T>	m_output_error_buffer;
	
public:
	// バッファ設定
	void  SetInputSignalBuffer(NeuralNetBuffer<T> buffer)  { m_input_signal_buffer = buffer; }
	void  SetOutputSignalBuffer(NeuralNetBuffer<T> buffer) { m_output_signal_buffer = buffer; }
	void  SetInputErrorBuffer(NeuralNetBuffer<T> buffer)  { m_input_error_buffer = buffer; }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T> buffer) { m_output_error_buffer = buffer; }
	
	// バッファ取得
	const NeuralNetBuffer<T>& GetInputSignalBuffer(void) const { return m_input_signal_buffer; }
	const NeuralNetBuffer<T>& GetOutputSignalBuffer(void) const { return m_output_signal_buffer; }
	const NeuralNetBuffer<T>& GetInputErrorBuffer(void) const { return m_input_error_buffer; }
	const NeuralNetBuffer<T>& GetOutputErrorBuffer(void) const { return m_output_error_buffer; }
};


}