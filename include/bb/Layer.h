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

#include "bb/FrameBuffer.h"


namespace bb {


//! layer class
class Layer
{
protected:
	std::string		m_layerName;

public:
    /**
     * @brief  デストラクタ(仮想関数)
     * @detail デストラクタ(仮想関数)
     */
	virtual ~Layer() {}

    /**
     * @brief  クラス名取得
     * @detail クラス名取得
     *         シリアライズ時などの便宜上、クラス名を返すようにする
     * @return クラス名
     */
	virtual std::string GetClassName(void) const = 0;

    /**
     * @brief  レイヤー名設定
     * @detail レイヤー名設定
     *         便宜上インスタンスの区別の付く命名を可能にする
     */
	virtual void  SetLayerName(const std::string name) {
		m_layerName = name;
	}
    
    /**
     * @brief  レイヤー名取得
     * @detail レイヤー名取得
     *         便宜上インスタンスの区別の付く命名を可能にする
     * @return レイヤー名取得名
     */
	virtual std::string GetLayerName(void) const
	{
		if (m_layerName.empty()) {
			return GetClassName();
		}
		return m_layerName;
	}
	
//	virtual GetVariables GetVariables(void) {} = 0;
	
	virtual void  SetBinaryMode(bool enable) {}									// バイナリモードを設定
	
	virtual	void  Forward(bool train=true) = 0;									// 予測
	virtual	void  Backward(void) = 0;											// 誤差逆伝播
	virtual	void  Update(void) = 0;												// 学習
	virtual	bool  Feedback(const std::vector<double>& loss) { return false; }	// 直接フィードバック
	
	
	// Serialize(CEREAL)
	template <class Archive>
	void save(Archive& archive, std::uint32_t const version) const
	{
		archive(cereal::make_nvp("layer_name", m_layerName));
	}

	template <class Archive>
	void load(Archive& archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("layer_name", m_layerName));
	}

	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		archive(cereal::make_nvp("Layer", *this));
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		archive(cereal::make_nvp("Layer", *this));
	}
};



}