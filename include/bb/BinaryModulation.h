// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Model.h"
#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"


namespace bb {


template <typename FRT = float, typename FBT = float, typename BT = float>
class BinaryModulation : public Model
{
    using _super = Model;

protected:

    // 3層で構成
    std::shared_ptr< RealToBinary<FRT, FBT, BT> >   m_real2bin;
    std::shared_ptr< Model                      >   m_layer;
    std::shared_ptr< BinaryToReal<FBT, FRT, BT> >   m_bin2real;
    
protected:
    BinaryModulation() {}

public:
    ~BinaryModulation() {}

    struct create_t
    {
        std::shared_ptr<Model>  layer;
    };

    static std::shared_ptr<BinaryModulation> Create(create_t const & create)
    {
        auto self = std::shared_ptr<BinaryModulation>(new BinaryModulation);

        self->m_real2bin = RealToBinary<FRT, FBT, BT>::Create();
        self->m_layer    = create.layer;
        self->m_bin2real =  BinaryToReal<FBT, FRT, BT>::Create();

        return self;
    }

    static std::shared_ptr<BinaryModulation> Create(std::shared_ptr<Model> layer)
    {
        auto self = std::shared_ptr<BinaryModulation>(new BinaryModulation);

        self->m_real2bin = RealToBinary<FRT, FBT, BT>::Create();
        self->m_layer    = create.layer;
        self->m_bin2real =  BinaryToReal<FBT, FRT, BT>::Create();

        return self;
    }

    std::string GetClassName(void) const { return "BinaryModulation"; }

    
    std::shared_ptr< Model > GetLayer(void)
    {
        return m_layer;
    }


    /**
     * @brief  コマンドを送る
     * @detail コマンドを送る
     */   
    void SendCommand(std::string command, std::string send_to = "all")
    {
        m_real2bin->SendCommand(command, send_to);
        m_layer->SendCommand(command, send_to);
        m_bin2real->SendCommand(command, send_to);
    }
    
    /**
     * @brief  パラメータ取得
     * @detail パラメータを取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetParameters(void)
    {
        Variables parameters;
        parameters.PushBack(m_real2bin->GetParameters());
        parameters.PushBack(m_layer->GetParameters());
        parameters.PushBack(m_bin2real->GetParameters());
        return parameters;
    }

    /**
     * @brief  勾配取得
     * @detail 勾配を取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    virtual Variables GetGradients(void)
    {
        Variables gradients;
        gradients.PushBack(m_real2bin->GetGradients());
        gradients.PushBack(m_layer->GetGradients());
        gradients.PushBack(m_bin2real->GetGradients());
        return gradients;
    }  

    /**
     * @brief  入力形状設定
     * @detail 入力形状を設定する
     *         内部変数を初期化し、以降、GetOutputShape()で値取得可能となることとする
     *         同一形状を指定しても内部変数は初期化されるものとする
     * @param  shape      1フレームのノードを構成するshape
     * @return 出力形状を返す
     */
    indices_t SetInputShape(indices_t shape)
    {
        shape = m_real2bin->SetInputShape(shape);
        shape = m_layer->SetInputShape(shape);
        shape = m_bin2real->SetInputShape(shape);

        return shape;
    }


    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_real2bin->GetInputShape();
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_bin2real->GetOutputShape();
    }
    

   /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        x = m_real2bin->Forward(x, train);
        x = m_layer->Forward(x, train);
        x = m_bin2real->Forward(x, train);
        return x;
    }

   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    FrameBuffer Backward(FrameBuffer dy)
    {
        dy = m_bin2real->Backward(dy);
        dy = m_layer->Backward(dy);
        dy = m_real2bin->Backward(dy);
        return dy; 
    }
    
protected:
    /**
     * @brief  モデルの情報を表示
     * @detail モデルの情報を表示する
     * @param  os     出力ストリーム
     * @param  indent インデント文字列
     */
    void PrintInfoText(std::ostream& os, std::string indent, int columns, int nest, int depth)
    {
        // これ以上ネストしないなら自クラス概要
        if ( depth > 0 && (nest+1) >= depth ) {
            Model::PrintInfoText(os, indent, columns, nest, depth);
        }
        else {
            // 子レイヤーの表示
            m_real2bin->PrintInfo(depth, os, columns, nest+1);
            m_layer->PrintInfo(depth, os, columns, nest+1);
            m_bin2real->PrintInfo(depth, os, columns, nest+1);
        }
    }

public:
    // Serialize
    void Save(std::ostream &os) const 
    {
        m_real2bin->Save(os);
        m_layer->Save(os);
        m_bin2real->Save(os);
    }

    void Load(std::istream &is)
    {
        m_real2bin->Load(is);
        m_layer->Load(is);
        m_bin2real->Load(is);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
    }

    void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("BinaryModulation", *this));
        m_real2bin->Save(archive);
        m_layer->Save(archive);
        m_bin2real->Save(archive);
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("BinaryModulation", *this));
        m_real2bin->Load(archive);
        m_layer->Load(archive);
        m_bin2real->Load(archive);
    }
#endif
};


}

