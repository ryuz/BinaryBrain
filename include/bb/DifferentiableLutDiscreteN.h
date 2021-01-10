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

#include "bb/StochasticLutModel.h"
#include "bb/StochasticLutN.h"
#include "bb/StochasticBatchNormalization.h"
#include "bb/BatchNormalization.h"
#include "bb/HardTanh.h"


namespace bb {


// Differentiable LUT (Discrite version)
template <int N = 6, typename BinType = float, typename RealType = float>
class DifferentiableLutDiscreteN : public StochasticLutModel
{
    using _super =  StochasticLutModel;

public:
    static inline std::string ModelName(void) { return "DifferentiableLutDiscrete" + std::to_string(N); }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    bool                                                            m_memory_saving = false;
    bool                                                            m_bn_enable  = true;
    bool                                                            m_act_enable = true;

    // 2層で構成
    std::shared_ptr< StochasticLutN<N, BinType, RealType> >         m_lut;
    std::shared_ptr< StochasticBatchNormalization<RealType>   >     m_batch_norm;
    std::shared_ptr< HardTanh<BinType, RealType>   >                m_activation;

public:
    struct create_t
    {
        indices_t       output_shape;
        std::string     connection;     //< 結線ルール
        bool            batch_norm = true;
        bool            activation = true;
        RealType        momentum   = (RealType)0.0;
        RealType        gamma      = (RealType)0.3;
        RealType        beta       = (RealType)0.5;
        std::uint64_t   seed       = 1;       //< 乱数シード
    };

protected:
    DifferentiableLutDiscreteN(create_t const &create)
    {
        typename StochasticLutN<N, BinType, RealType>::create_t lut_create;
        lut_create.output_shape = create.output_shape;
        lut_create.connection   = create.connection;
        lut_create.seed         = create.seed;
        m_lut = StochasticLutN<N, BinType, RealType>::Create(lut_create);

        m_bn_enable  = create.batch_norm;
        m_batch_norm = StochasticBatchNormalization<RealType>::Create(create.momentum, create.gamma,  create.beta);

        m_act_enable = create.activation;
        m_activation = HardTanh<BinType, RealType>::Create((RealType)0, (RealType)1);
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args)
    {
        if ( args.size() == 2 && args[0] == "memory_saving" )
        {
            m_memory_saving = EvalBool(args[1]);
        }
    }



public:
    ~DifferentiableLutDiscreteN() {}

    static std::shared_ptr< DifferentiableLutDiscreteN > Create(create_t const &create)
    {
        return std::shared_ptr< DifferentiableLutDiscreteN >(new DifferentiableLutDiscreteN(create));
    }

    static std::shared_ptr< DifferentiableLutDiscreteN > Create(indices_t const &output_shape, bool batch_norm = true, std::string connection = "", std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.connection   = connection;
        create.batch_norm   = batch_norm;
        create.seed         = seed;
        return Create(create);
    }

    static std::shared_ptr< DifferentiableLutDiscreteN > Create(index_t output_node_size, bool batch_norm = true, std::string connection = "", std::uint64_t seed = 1)
    {
        return Create(indices_t({output_node_size}), batch_norm, connection, seed);
    }

    static std::shared_ptr<DifferentiableLutDiscreteN> Create(void)
    {
        return Create(create_t());
    }

#ifdef BB_PYBIND11
    static std::shared_ptr< DifferentiableLutDiscreteN > CreatePy(indices_t const &output_shape, bool batch_norm = true, bool activation = true, std::string connection = "",
                                                             RealType momentum = (RealType)0.0, RealType gamma = (RealType)0.3, RealType beta = (RealType)0.5,
                                                             std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.connection   = connection;
        create.batch_norm   = batch_norm;
        create.activation   = activation;
        create.momentum     = momentum;
        create.gamma        = gamma;
        create.beta         = beta;
        create.seed         = seed;
        return Create(create);
    }
#endif

    /**
     * @brief  コマンドを送る
     * @detail コマンドを送る
     */   
    void SendCommand(std::string command, std::string send_to = "all")
    {
        _super::SendCommand(command, send_to);

        m_lut       ->SendCommand(command, send_to);
        m_batch_norm->SendCommand(command, send_to);
        m_activation->SendCommand(command, send_to);
    }

    Tensor       &W(void)        { return m_lut->W(); }
    Tensor const &W(void) const  { return m_lut->W(); }
    Tensor       &dW(void)       { return m_lut->dW(); }
    Tensor const &dW(void) const { return m_lut->dW(); }

    auto lock_InputIndex(void)             { return m_lut->lock_InputIndex(); }
    auto lock_InputIndex_const(void) const { return m_lut->lock_InputIndex_const(); }
    auto lock_W(void)                      { return m_lut->lock_W(); }
    auto lock_W_const(void) const          { return m_lut->lock_W_const(); }
    auto lock_dW(void)                     { return m_lut->lock_dW(); }
    auto lock_dW_const(void) const         { return m_lut->lock_dW_const(); }
    
    auto lock_mean(void)               { return m_batch_norm->lock_mean(); }
    auto lock_mean_const(void)   const { return m_batch_norm->lock_mean_const(); }
    auto lock_var(void)                { return m_batch_norm->lock_var(); }
    auto lock_var_const(void)    const { return m_batch_norm->lock_var_const(); }
    
    // debug
    auto lock_tmp_mean_const(void)   const { return m_batch_norm->lock_tmp_mean_const(); }
    auto lock_tmp_rstd_const(void)   const { return m_batch_norm->lock_tmp_rstd_const(); }


    /**
     * @brief  パラメータ取得
     * @detail パラメータを取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetParameters(void)
    {
        Variables parameters;
        parameters.PushBack(m_lut       ->GetParameters());
        parameters.PushBack(m_batch_norm->GetParameters());
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
        gradients.PushBack(m_lut       ->GetGradients());
        gradients.PushBack(m_batch_norm->GetGradients());
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
        shape = m_lut->SetInputShape(shape);
        if ( m_bn_enable )  { shape = m_batch_norm->SetInputShape(shape); }
        if ( m_act_enable ) { shape = m_activation->SetInputShape(shape); }
        return shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_lut->GetInputShape();
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
        return m_lut->GetNodeConnectionSize(node);
    }

    void SetNodeConnectionIndex(index_t node, index_t input_index, index_t input_node)
    {
        m_lut->SetNodeConnectionIndex(node, input_index, input_node);
    }

    index_t GetNodeConnectionIndex(index_t node, index_t input_index) const
    {
        return m_lut->GetNodeConnectionIndex(node, input_index);
    }

    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {
        index_t input_size = this->GetNodeConnectionSize(node);
        BB_ASSERT(input_size == (index_t)x_vec.size());

        x_vec = m_lut->ForwardNode(node, x_vec);
        if ( m_bn_enable )   { x_vec = m_batch_norm->ForwardNode(node, x_vec); }
        if ( m_act_enable )  { x_vec = m_activation->ForwardNode(node, x_vec); }
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
        x_buf = m_lut->Forward(x_buf, train);
        if ( m_bn_enable ) {
            x_buf = m_batch_norm->Forward(x_buf, train);
            if (m_memory_saving || !train ) { m_batch_norm->SetFrameBufferX(FrameBuffer()); }
        }
        
        if ( m_act_enable ) {
            x_buf = m_activation->Forward(x_buf, train);
            if (m_memory_saving || !train ) { m_activation->SetFrameBufferX(FrameBuffer()); }
        }

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
        if ( m_memory_saving && (m_bn_enable || m_act_enable) ) {
            // 再計算
            FrameBuffer x_buf;
            x_buf = m_lut->ReForward(m_lut->GetFrameBufferX());
            if ( m_bn_enable ) {
                x_buf = m_batch_norm->ReForward(x_buf);
            }
            if ( m_act_enable ) {
                m_activation->SetFrameBufferX(x_buf);
            }
        }

        if ( m_act_enable ) { dy_buf = m_activation->Backward(dy_buf); }
        if ( m_bn_enable )  { dy_buf = m_batch_norm->Backward(dy_buf); }
        dy_buf = m_lut->Backward(dy_buf);
        return dy_buf; 
    }

protected:
    /**
     * @brief  モデルの情報を表示
     * @detail モデルの情報を表示する
     * @param  os     出力ストリーム
     * @param  indent インデント文字列
     */
    void PrintInfoText(std::ostream& os, std::string indent, int columns, int nest, int depth) const override
    {
        // これ以上ネストしないなら自クラス概要
        if ( depth > 0 && (nest+1) >= depth ) {
            Model::PrintInfoText(os, indent, columns, nest, depth);
        }
        else {
            // 子レイヤーの表示
            m_lut->PrintInfo(depth, os, columns, nest+1);
            if ( m_bn_enable )  { m_batch_norm->PrintInfo(depth, os, columns, nest+1); }
            if ( m_bn_enable )  { m_activation->PrintInfo(depth, os, columns, nest+1); }
        }
    }


    // シリアライズ
protected:
    void DumpObjectData(std::ostream &os) const override
    {
        // バージョン
        std::int64_t ver = 1;
        bb::SaveValue(os, ver);

        // 親クラス
        _super::DumpObjectData(os);

        // メンバ
        bb::SaveValue(os, m_memory_saving);
        bb::SaveValue(os, m_bn_enable);
        bb::SaveValue(os, m_act_enable);
        m_lut->DumpObject(os);
        if ( m_bn_enable )  { m_batch_norm->DumpObject(os); }
        if ( m_act_enable ) { m_activation->DumpObject(os); }

    }

    void LoadObjectData(std::istream &is) override
    {
        // バージョン
        std::int64_t ver;
        bb::LoadValue(is, ver);

        BB_ASSERT(ver == 1);

        // 親クラス
        _super::LoadObjectData(is);

        // メンバ
        bb::LoadValue(is, m_memory_saving);
        bb::LoadValue(is, m_bn_enable);
        bb::LoadValue(is, m_act_enable);
        m_lut->LoadObject(is);
        if ( m_bn_enable )  { m_batch_norm->LoadObject(is); }
        if ( m_act_enable ) { m_activation->LoadObject(is); }
    }

public:
    // Serialize(旧)
    void Save(std::ostream &os) const 
    {
        bb::SaveValue(os, m_memory_saving);
        bb::SaveValue(os, m_bn_enable);
        bb::SaveValue(os, m_act_enable);

        m_lut->Save(os);
        if ( m_bn_enable )  { m_batch_norm->Save(os); }
        if ( m_act_enable ) { m_activation->Save(os); }
    }

    void Load(std::istream &is)
    {
        bb::LoadValue(is, m_memory_saving);
        bb::LoadValue(is, m_bn_enable);
        bb::LoadValue(is, m_act_enable);

        m_lut->Load(is);
        if ( m_bn_enable )  { m_batch_norm->Load(is); }
        if ( m_act_enable ) { m_activation->Load(is); }
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
        archive(cereal::make_nvp("DifferentiableLutN", *this));
        m_lut       ->Save(archive);
        m_batch_norm->Save(archive);
        m_activation->Save(archive);
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("DifferentiableLutN", *this));
        m_lut       ->Load(archive);
        m_batch_norm->Load(archive);
        m_activation->Load(archive);
    }
#endif

};


}