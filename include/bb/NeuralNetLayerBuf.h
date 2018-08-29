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

#include "NeuralNetLayer.h"


namespace bb {


// NeuralNetのバッファ付き抽象クラス
template <typename T=float, typename INDEX = size_t>
class NeuralNetLayerBuf : public NeuralNetLayer<T, INDEX>
{
protected:
	// バッファ情報
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
	const NeuralNetBuffer<T, INDEX>& GetInputValueBuffer(void) const { return m_input_value_buffer; }
	const NeuralNetBuffer<T, INDEX>& GetOutputValueBuffer(void) const { return m_output_value_buffer; }
	const NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) const { return m_input_error_buffer; }
	const NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) const { return m_output_error_buffer; }
};


}