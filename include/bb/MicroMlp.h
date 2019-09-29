// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>
#include <random>

#include "bb/Model.h"
#include "bb/MicroMlpAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/Binarize.h"


namespace bb {


// Sparce Mini-MLP(Multilayer perceptron) Layer [Affine-ReLU-Affine-BatchNorm-Binarize]
template <int N = 6, int M = 16, typename FT = float, typename T = float>
class MicroMlp : public SparseLayer
{
    using _super = SparseLayer;

protected:
    bool                                            m_memory_saving = false;// true;

    // 3層で構成
    std::shared_ptr< MicroMlpAffine<N, M, FT, T> >  m_affine;
    std::shared_ptr< BatchNormalization<T>       >  m_batch_norm;
    std::shared_ptr< Binarize<FT, T>             >  m_activation;

public:
    // 生成情報
    struct create_t
    {
        std::string     name;
        indices_t       output_shape;
        std::string     connection;
        T               initialize_std = (T)0.01;
        std::string     initializer    = "";
        T               momentum       = (T)0.0;
        T               gamma          = (T)1.0;
        T               beta           = (T)0.0;
        std::uint64_t   seed           = 1;
    };

protected:
    // コンストラクタ
    MicroMlp(create_t const &create)
    {
        this->SetName(create.name);

        typename MicroMlpAffine<N, M, FT, T>::create_t affine_create;
        affine_create.output_shape   = create.output_shape;
        affine_create.connection     = create.connection;
        affine_create.initialize_std = create.initialize_std;
        affine_create.initializer    = create.initializer;
        affine_create.seed           = create.seed;
        m_affine = MicroMlpAffine<N, M, FT, T>::Create(affine_create);

        typename BatchNormalization<T>::create_t bn_create;
        bn_create.momentum  = create.momentum;
        bn_create.gamma     = create.gamma;
        bn_create.beta      = create.beta;
        m_batch_norm = BatchNormalization<T>::Create(bn_create);
        
        m_activation = Binarize<FT, T>::Create();
    }

    void CommandProc(std::vector<std::string> args)
    {
        if ( args.size() == 2 && args[0] == "memory_saving" )
        {
            m_memory_saving = EvalBool(args[1]);
        }
    }

public:
    // デストラクタ
    ~MicroMlp() {}
    
    // 生成
    static std::shared_ptr< MicroMlp > Create(create_t const &create)
    {
        return std::shared_ptr<MicroMlp>(new MicroMlp(create));
    }

    static std::shared_ptr< MicroMlp > Create(indices_t const &output_shape, std::string connection = "", T momentum = (T)0.0)
    {
        create_t create;
        create.output_shape = output_shape;
        create.connection   = connection;
        create.momentum     = momentum;
        return Create(create);
    }

    static std::shared_ptr< MicroMlp > Create(index_t output_node_size, std::string connection = "", T momentum = (T)0.0)
    {
        return Create(indices_t({output_node_size}), connection, momentum);
    }
    

    std::string GetClassName(void) const { return "MicroMlp"; }

    /**
     * @brief  コマンドを送る
     * @detail コマンドを送る
     */   
    void SendCommand(std::string command, std::string send_to = "all")
    {
        _super::SendCommand(command, send_to);

        m_affine    ->SendCommand(command, send_to);
        m_batch_norm->SendCommand(command, send_to);
        m_activation->SendCommand(command, send_to);
    }

    auto lock_W0(void)             { return m_affine->lock_W0(); }     
    auto lock_W0_const(void) const { return m_affine->lock_W0_const(); }
    auto lock_b0(void)             { return m_affine->lock_b0(); }      
    auto lock_b0_const(void) const { return m_affine->lock_b0_const(); }
    auto lock_W1(void)             { return m_affine->lock_W1(); }
    auto lock_W1_const(void) const { return m_affine->lock_W1_const(); }
    auto lock_b1(void)             { return m_affine->lock_b1(); }
    auto lock_b1_const(void) const { return m_affine->lock_b1_const(); }

    /**
     * @brief  パラメータ取得
     * @detail パラメータを取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetParameters(void)
    {
        Variables parameters;
        parameters.PushBack(m_affine    ->GetParameters());
        parameters.PushBack(m_batch_norm->GetParameters());
        parameters.PushBack(m_activation->GetParameters());
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
        gradients.PushBack(m_affine    ->GetGradients());
        gradients.PushBack(m_batch_norm->GetGradients());
        gradients.PushBack(m_activation->GetGradients());
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
        shape = m_affine    ->SetInputShape(shape);
        shape = m_batch_norm->SetInputShape(shape);
        shape = m_activation->SetInputShape(shape);
        return shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_affine->GetInputShape();
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_activation->GetOutputShape();
    }



    index_t GetNodeConnectionSize(index_t node) const
    {
        return m_affine->GetNodeConnectionSize(node);
    }

    void SetNodeConnectionIndex(index_t node, index_t input_index, index_t input_node)
    {
        m_affine->SetNodeConnectionIndex(node, input_index, input_node);
    }

    index_t GetNodeConnectionIndex(index_t node, index_t input_index) const
    {
        return m_affine->GetNodeConnectionIndex(node, input_index);
    }

    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {
        x_vec = m_affine    ->ForwardNode(node, x_vec);
        x_vec = m_batch_norm->ForwardNode(node, x_vec);
        x_vec = m_activation->ForwardNode(node, x_vec);
        return x_vec;
    }

   /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        x_buf = m_affine    ->Forward(x_buf, train);
        x_buf = m_batch_norm->Forward(x_buf, train);
        if (m_memory_saving || !train ) { m_batch_norm->SetFrameBufferX(FrameBuffer()); }
        x_buf = m_activation->Forward(x_buf, train);
        if (m_memory_saving || !train ) { m_activation->SetFrameBufferX(FrameBuffer()); }

        return x_buf;
    }

   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        if (m_memory_saving) {
            // 再計算
            FrameBuffer x_buf;
            x_buf = m_affine    ->ReForward(m_affine->GetFrameBufferX());
            x_buf = m_batch_norm->ReForward(x_buf);
            m_activation->SetFrameBufferX(x_buf);
        }

        dy_buf = m_activation->Backward(dy_buf);
        dy_buf = m_batch_norm->Backward(dy_buf);
        dy_buf = m_affine    ->Backward(dy_buf);
        return dy_buf; 
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
            m_affine->PrintInfo(depth, os, columns, nest+1);
            m_batch_norm->PrintInfo(depth, os, columns, nest+1);
            m_activation->PrintInfo(depth, os, columns, nest+1);
        }
    }

public:
    // Serialize
    void Save(std::ostream &os) const 
    {
        m_affine->Save(os);
        m_batch_norm->Save(os);
        m_activation->Save(os);
    }

    void Load(std::istream &is)
    {
        m_affine->Load(is);
        m_batch_norm->Load(is);
        m_activation->Load(is);
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
        archive(cereal::make_nvp("MicroMlp", *this));
        m_affine->Save(archive);
        m_batch_norm->Save(archive);
        m_activation->Save(archive);
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("MicroMlp", *this));
        m_affine->Load(archive);
        m_batch_norm->Load(archive);
        m_activation->Load(archive);
    }
#endif

};


}