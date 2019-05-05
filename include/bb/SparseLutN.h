// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>
#include <random>

#include "bb/Model.h"
#include "bb/StochasticLutN.h"
#include "bb/StochasticBatchNormalization.h"
#include "bb/HardTanh.h"


namespace bb {


// Sparce Mini-MLP(Multilayer perceptron) Layer [Affine-ReLU-Affine-BatchNorm-Binarize]
template <int N = 6, typename T = float >
class SparseLutN : public SparseLayer<T, T>
{
    using _super = SparseLayer<T, T>;

protected:
    // 2層で構成
    std::shared_ptr< StochasticBatchNormalization<T>   >    m_norm_pre;
    std::shared_ptr< StochasticLutN<N, T> >                 m_lut;
    std::shared_ptr< StochasticBatchNormalization<T>   >    m_norm_post;
    std::shared_ptr< HardTanh<T>   >                        m_activate;

public:
    struct create_t
    {
        indices_t       output_shape;
        std::string     connection;     //< 結線ルール
        std::uint64_t   seed = 1;       //< 乱数シード
    };

protected:
    SparseLutN(create_t const &create)
    {
        m_norm_pre  = StochasticBatchNormalization<T>::Create(0.001f);

        typename StochasticLutN<N, T>::create_t lut_create;
        lut_create.output_shape = create.output_shape;
        lut_create.connection   = create.connection;
        lut_create.seed         = create.seed;
        m_lut = StochasticLutN<N, T>::Create(lut_create);

        m_norm_post = StochasticBatchNormalization<T>::Create(0.001f);

        m_activate = HardTanh<T>::Create((T)0, (T)1);
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args)
    {
    }



public:
    ~SparseLutN() {}

    static std::shared_ptr< SparseLutN > Create(create_t const &create)
    {
        return std::shared_ptr< SparseLutN >(new SparseLutN(create));
    }

    static std::shared_ptr< SparseLutN > Create(indices_t output_shape, std::string connection = "random", std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.connection   = connection;
        return Create(create);
    }

    static std::shared_ptr< SparseLutN > Create(index_t output_node_size, std::string connection = "random", std::uint64_t seed = 1)
    {
        return Create(indices_t({output_node_size}), connection, seed);
    }

    std::string GetClassName(void) const { return "SparseLutN"; }

    /**
     * @brief  コマンドを送る
     * @detail コマンドを送る
     */   
    void SendCommand(std::string command, std::string send_to = "all")
    {
        _super::SendCommand(command, send_to);

        m_norm_pre ->SendCommand(command, send_to);
        m_lut      ->SendCommand(command, send_to);
        m_norm_post->SendCommand(command, send_to);
        m_activate ->SendCommand(command, send_to);
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
        parameters.PushBack(m_norm_pre ->GetParameters());
        parameters.PushBack(m_lut      ->GetParameters());
        parameters.PushBack(m_norm_post->GetParameters());
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
        gradients.PushBack(m_norm_pre ->GetGradients());
        gradients.PushBack(m_lut      ->GetGradients());
        gradients.PushBack(m_norm_post->GetGradients());
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
        shape = m_norm_pre ->SetInputShape(shape);
        shape = m_lut      ->SetInputShape(shape);
        shape = m_norm_post->SetInputShape(shape);
        shape = m_activate ->SetInputShape(shape);
        return shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_norm_pre->GetInputShape();
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_activate->GetOutputShape();
    }
    
    index_t GetNodeInputSize(index_t node) const
    {
        return m_lut->GetNodeInputSize(node);
    }

    void SetNodeInput(index_t node, index_t input_index, index_t input_node)
    {
        m_lut->SetNodeInput(node, input_index, input_node);
    }

    index_t GetNodeInput(index_t node, index_t input_index) const
    {
        return m_lut->GetNodeInput(node, input_index);
    }

    std::vector<T> ForwardNode(index_t node, std::vector<T> x_vec) const
    {
        index_t input_size = this->GetNodeInputSize(node);
        BB_ASSERT(input_size == x_vec.size());

        std::vector<T> tmp(1);
        for (index_t i = 0; i < input_size; ++i) {
            index_t input_node = this->GetNodeInput(node, i);
            tmp[0] = x_vec[i];
            tmp = m_norm_pre->ForwardNode(input_node, tmp);
            x_vec[i] = tmp[0];
        }

        x_vec = m_lut->ForwardNode(node, x_vec);

        x_vec = m_norm_post->ForwardNode(node, x_vec);

        x_vec = m_activate->ForwardNode(node, x_vec);

        return x_vec;
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
        x = m_norm_pre ->Forward(x, train);
        x = m_lut      ->Forward(x, train);
        x = m_norm_post->Forward(x, train);
        x = m_activate ->Forward(x, train);
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
        dy = m_activate ->Backward(dy);
        dy = m_norm_post->Backward(dy);
        dy = m_lut      ->Backward(dy);
        dy = m_norm_pre ->Backward(dy);

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
            m_norm_pre ->PrintInfo(depth, os, columns, nest+1);
            m_lut      ->PrintInfo(depth, os, columns, nest+1);
            m_norm_post->PrintInfo(depth, os, columns, nest+1);
            m_activate ->PrintInfo(depth, os, columns, nest+1);
        }
    }

public:
    // Serialize
    void Save(std::ostream &os) const 
    {
        m_norm_pre ->Save(os);
        m_lut      ->Save(os);
        m_norm_post->Save(os);
        m_activate ->Save(os);
    }

    void Load(std::istream &is)
    {
        m_norm_pre ->Load(is);
        m_lut      ->Load(is);
        m_norm_post->Load(is);
        m_activate ->Load(is);
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
        archive(cereal::make_nvp("SparseLutN", *this));
        m_norm_pre ->Save(archive);
        m_lut      ->Save(archive);
        m_norm_post->Save(archive);
        m_activate ->Save(archive);
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("SparseLutN", *this));
        m_norm_pre ->Load(archive);
        m_lut      ->Load(archive);
        m_norm_post->Load(archive);
        m_activate ->Load(archive);
    }
#endif

};


}