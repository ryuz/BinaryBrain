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
	
	
//	virtual void  Command(std::string command) {}								// コマンドを送信
	virtual void  SetBinaryMode(bool enable) {}									// バイナリモードを設定

    //	virtual GetVariables GetParameters(void) {} = 0;
    //	virtual GetVariables GetGradients(void) {} = 0;
    
    /**
     * @brief  入力形状設定
     * @detail 入力形状を設定する
     *         内部変数を初期化し、以降、GetOutputShape()で値取得可能となることとする
     *         同一形状を指定しても内部変数は初期化されるものとする
     * @param  shape      1フレームのノードを構成するshape
     * @return 出力形状を返す
     */
    virtual indices_t SetInputShape(indices_t shape) = 0;

   /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    virtual	FrameBuffer Forward(FrameBuffer x, bool train=true) = 0;

   /**
     * @brief  forward演算(複数入力対応)
     * @detail forward演算を行う
     *         分岐や合流演算を可能とするために汎用版を定義しておく
     * @return forward演算結果
     */
    virtual	std::vector<FrameBuffer> Forward(std::vector<FrameBuffer> vx, bool train = true)
    {
        BB_ASSERT(vx.size() == 1);
        auto& y = Forward(vx[0], train);
        return {y};
    }


   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
	virtual	FrameBuffer Backward(FrameBuffer dy) = 0;

   /**
     * @brief  backward演算(複数入力対応)
     * @detail backward演算を行う
     *         分岐や合流演算を可能とするために汎用版を定義しておく
     * @return backward演算結果
     */
    virtual	std::vector<FrameBuffer> Backward(std::vector<FrameBuffer> vdy)
    {
        BB_ASSERT(vdy.size() == 1);
        auto& dx = Backward(vdy[0]);
        return {dx};
    }
	
	
public:
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