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
#include "bb/ReLU.h"


namespace bb {


// Sparce Mini-MLP(Multilayer perceptron) Layer [Affine-ReLU-Affine-BatchNorm-Binarize]
template <int N = 6, typename T = float >
class StochasticLutBn : public SparseLayer
{
    using _super = SparseLayer;

protected:
    // 2層で構成
    std::shared_ptr< StochasticBatchNormalization<T>   >    m_norm;
    std::shared_ptr< SparseLayer >                          m_lut;

    bool                                                    m_bn_enable = true;


public:
    struct create_t
    {
        indices_t  output_shape;
        bool       bn_enable = true;
        T          momentum = (T)0.9;
        T          gamma    = (T)0.2;
        T          beta     = (T)0.5;
    };

protected:
    StochasticLutBn(create_t const &create)
    {
        m_lut = StochasticLutN<N, T>::Create(create.output_shape);
        m_bn_enable = create.bn_enable;

        typename StochasticBatchNormalization<T>::create_t bn_create;
        bn_create.momentum  = create.momentum;
        bn_create.gamma     = create.gamma; 
        bn_create.beta      = create.beta;
        m_norm = StochasticBatchNormalization<T>::Create(bn_create);
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args)
    {
        // BatchNormalization設定
        if ( args.size() == 2 && args[0] == "batch_normalization" )
        {
            m_bn_enable = EvalBool(args[1]);
        }
    }

public:
    ~StochasticLutBn() {}

    static std::shared_ptr< StochasticLutBn > Create(create_t const &create)
    {
        return std::shared_ptr<StochasticLutBn>(new StochasticLutBn(create));
    }

    static std::shared_ptr< StochasticLutBn > Create(indices_t output_shape, bool bn_enable = true, T momentum = (T)0.9, T gamma = (T)0.2, T beta = (T)0.5)
    {
        create_t create;
        create.output_shape = output_shape;
        create.bn_enable    = bn_enable;
        create.momentum     = momentum;
        create.gamma        = gamma;
        create.beta         = beta;
        return Create(create);
    }

    static std::shared_ptr< StochasticLutBn > Create(index_t output_node_size, bool bn_enable = true, T momentum = (T)0.9, T gamma = (T)0.2, T beta = (T)0.5)
    {
        create_t create;
        create.output_shape = indices_t({output_node_size});
        create.bn_enable    = bn_enable;
        create.momentum     = momentum;
        create.gamma        = gamma;
        create.beta         = beta;
        return Create(create);
    }

    static std::shared_ptr< StochasticLutBn > CreateEx(
                indices_t   output_shape,
                bool        bn_enable = true,
                double      momentum = 0.9,
                double      gamma = 0.2,
                double      beta = 0.5)
    {
        create_t create;
        create.output_shape = indices_t({output_node_size});
        create.bn_enable    = bn_enable;
        create.momentum     = (T)momentum;
        create.gamma        = (T)gamma;
        create.beta         = (T)beta;
        return Create(create);
    }


    std::string GetClassName(void) const { return "StochasticLutBn"; }

    /**
     * @brief  コマンドを送る
     * @detail コマンドを送る
     */   
    void SendCommand(std::string command, std::string send_to = "all")
    {
        _super::SendCommand(command, send_to);

        m_norm->SendCommand(command, send_to);
        m_lut       ->SendCommand(command, send_to);
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
        if ( m_bn_enable ) {
            parameters.PushBack(m_norm->GetParameters());
        }
        parameters.PushBack(m_lut->GetParameters());
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
        if ( m_bn_enable ) {
            gradients.PushBack(m_norm->GetGradients());
        }
        gradients.PushBack(m_lut       ->GetGradients());
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
        shape = m_norm->SetInputShape(shape);
        shape = m_lut->SetInputShape(shape);
        return shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_norm->GetInputShape();
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_lut->GetOutputShape();
    }
    


    index_t GetNodeConnectionSize(index_t output_node) const
    {
        return m_lut->GetNodeConnectionSize(output_node);
    }

    void SetNodeConnectionIndex(index_t output_node, index_t connection, index_t input_node)
    {
        m_lut->SetNodeConnectionIndex(output_node, connection, input_node);
    }

    index_t GetNodeConnectionIndex(index_t output_node, index_t connection) const
    {
        return m_lut->GetNodeConnectionIndex(output_node, connection);
    }

    /*
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
    */

    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {
        index_t input_size = this->GetNodeConnectionSize(node);
        BB_ASSERT(input_size == x_vec.size());

        if ( m_bn_enable ) {
            std::vector<double> tmp(1);
            for (index_t i = 0; i < input_size; ++i) {
                index_t input_node = this->GetNodeConnectionIndex(node, i);
                tmp[0] = x_vec[i];
                tmp = m_norm->ForwardNode(input_node, tmp);
                x_vec[i] = tmp[0];
            }
        }

        x_vec = m_lut->ForwardNode(node, x_vec);

        for (auto &x : x_vec) {
            x = (x - 0.5);
        }

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
        if ( m_bn_enable ) {
            x = m_norm->Forward(x, train);
        }
        x = m_lut->Forward(x, train);
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
        dy = m_lut->Backward(dy);
        if ( m_bn_enable ) {
            dy = m_norm->Backward(dy);
        }
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
            if ( m_bn_enable ) {
                m_norm->PrintInfo(depth, os, columns, nest+1);
            }
            m_lut->PrintInfo(depth, os, columns, nest+1);
        }
    }

public:
    // Serialize
    void Save(std::ostream &os) const 
    {
        m_norm->Save(os);
        m_lut->Save(os);
    }

    void Load(std::istream &is)
    {
        m_norm->Load(is);
        m_lut->Load(is);
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
        archive(cereal::make_nvp("StochasticLutBn", *this));
        m_norm->Save(archive);
        m_lut       ->Save(archive);
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("StochasticLutBn", *this));
        m_norm->Load(archive);
        m_lut       ->Load(archive);
    }
#endif

};


}