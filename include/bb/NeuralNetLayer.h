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

#include "NeuralNetBuffer.h"
#include "NeuralNetOptimizer.h"


namespace bb {


// abstract neura-network layer class
template <typename T=float, typename INDEX = size_t>
class NeuralNetLayer
{
protected:
	std::string					m_layer_name;

public:
	// basic functions
	virtual ~NeuralNetLayer() {}												// デストラクタ

	virtual void  SetLayerName(const std::string name) {						// レイヤー名設定
		m_layer_name = name;
	}
	virtual std::string GetLayerName(void)
	{
		return m_layer_name;
	}

	virtual void  Resize(std::vector<INDEX> size) {}							// サイズ設定
	virtual void  InitializeCoeff(std::uint64_t seed) {}						// 内部係数の乱数初期化
	
	virtual void  SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer) {}	//オプティマイザの設定

	virtual INDEX GetInputFrameSize(void) const = 0;							// 入力のフレーム数
	virtual INDEX GetInputNodeSize(void) const = 0;								// 入力のノード数
	virtual INDEX GetOutputFrameSize(void) const = 0;							// 出力のフレーム数
	virtual INDEX GetOutputNodeSize(void) const = 0;							// 出力のノード数
	
	virtual void  SetBinaryMode(bool enable) {}									// バイナリモードを設定

	virtual void  SetMuxSize(INDEX mux_size) = 0;								// 多重化サイズの設定
	virtual void  SetBatchSize(INDEX batch_size) = 0;							// バッチサイズの設定
	
	virtual T     CalcNode(INDEX node, std::vector<T> input_value) const { return input_value[0]; }	// 1ノードだけ個別計算
	
	virtual	void  Forward(bool train=true) = 0;									// 予測
	virtual	void  Backward(void) = 0;											// 誤差逆伝播
	virtual	void  Update(void) = 0;												// 学習
	virtual	bool  Feedback(const std::vector<double>& loss) { return false; }	// 直接フィードバック
	
	
	// forward propagation of signals
	virtual int   GetInputSignalDataType(void) const = 0;
	virtual int   GetOutputSignalDataType(void) const = 0;
	
	virtual void  SetInputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) = 0;
	virtual void  SetOutputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) = 0;
	
	virtual const NeuralNetBuffer<T, INDEX>& GetInputSignalBuffer(void) const = 0;
	virtual const NeuralNetBuffer<T, INDEX>& GetOutputSignalBuffer(void) const = 0;
	
	NeuralNetBuffer<T, INDEX> CreateInputSignalBuffer(void) { 
		return NeuralNetBuffer<T, INDEX>(GetInputFrameSize(), GetInputNodeSize(), GetInputSignalDataType());
	}
	NeuralNetBuffer<T, INDEX> CreateOutputSignalBuffer(void) {
		return NeuralNetBuffer<T, INDEX>(GetOutputFrameSize(), GetOutputNodeSize(), GetOutputSignalDataType());
	}
	

	// backward propagation of errors
	virtual int   GetInputErrorDataType(void) const = 0;
	virtual int   GetOutputErrorDataType(void) const = 0;

	virtual void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) = 0;
	virtual void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) = 0;

	virtual const NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) const = 0;
	virtual const NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) const = 0;

	NeuralNetBuffer<T, INDEX> CreateInputErrorBuffer(void) {
		return NeuralNetBuffer<T, INDEX>(GetInputFrameSize(), GetInputNodeSize(), GetInputErrorDataType());
	}
	NeuralNetBuffer<T, INDEX> CreateOutputErrorBuffer(void) {
		return NeuralNetBuffer<T, INDEX>(GetOutputFrameSize(), GetOutputNodeSize(), GetOutputErrorDataType());
	}
	
	
	// Serialize(CEREAL)
	template <class Archive>
	void save(Archive& archive, std::uint32_t const version) const
	{
		archive(cereal::make_nvp("layer_name", m_layer_name));
	}

	template <class Archive>
	void load(Archive& archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("layer_name", m_layer_name));
	}

	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		archive(cereal::make_nvp("NeuralNetLayer", *this));
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		archive(cereal::make_nvp("NeuralNetLayer", *this));
	}
};


}