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

#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/Binarize.h"


namespace bb {


// Differentiable LUT (Discrite version)
template <typename BinType = float, typename RealType = float>
class BinaryDenseAffine : public Model
{
    using _super        = Model;
    using AffineType    = DenseAffine<RealType>;
    using BatchNormType = BatchNormalization<RealType>;
    using ActType       = Binarize<BinType, RealType>;

public:
    static inline std::string ModelName(void) { return "BinaryDenseAffine"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    bool                                m_bn_enable  = true;
    bool                                m_act_enable = true;
    bool                                m_memory_saving = false;

    // 2層で構成
    std::shared_ptr< AffineType >       m_affine;
    std::shared_ptr< BatchNormType >    m_batch_norm;
    std::shared_ptr< ActType >          m_activation;

public:
    struct create_t
    {
        indices_t       output_shape;
        bool            batch_norm     = true;
        bool            activation     = true;
        
        RealType        initialize_std = (RealType)0.01;
        std::string     initializer    = "";
        
        RealType        momentum       = (RealType)0.9;
        RealType        gamma          = (RealType)1.0;
        RealType        beta           = (RealType)0.0;
        bool            fix_gamma      = false;
        bool            fix_beta       = false;
        
        RealType        binary_th      = (RealType)0;
        double          binary_low     = -1.0;
        double          binary_high    = +1.0;
        RealType        hardtanh_min   = (RealType)-1;
        RealType        hardtanh_max   = (RealType)+1;
        
        std::uint64_t   seed           = 1;       //< 乱数シード
        bool            memory_saving  = true;
    };


protected:
    BinaryDenseAffine(create_t const &create)
    {
        m_bn_enable     = create.batch_norm;
        m_act_enable    = create.activation;
        m_memory_saving = create.memory_saving;

        typename AffineType::create_t affine_create;
        affine_create.output_shape   = create.output_shape;
        affine_create.initialize_std = create.initialize_std;
        affine_create.initializer    = create.initializer;
        affine_create.seed           = create.seed;
        m_affine = AffineType::Create(affine_create);

        typename BatchNormType::create_t bn_create;
        bn_create.momentum  = create.momentum;
        bn_create.gamma     = create.gamma;
        bn_create.beta      = create.beta;
        bn_create.fix_gamma = create.fix_gamma;
        bn_create.fix_beta  = create.fix_beta;
        m_batch_norm = BatchNormType::Create(bn_create);

        typename ActType::create_t act_create;
        act_create.binary_th    = create.binary_th;
        act_create.binary_low   = create.binary_low;
        act_create.binary_high  = create.binary_high;
        act_create.hardtanh_min = create.hardtanh_min;
        act_create.hardtanh_max = create.hardtanh_max;
        m_activation = ActType::Create(act_create);
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args) override
    {
        if ( args.size() == 2 && args[0] == "memory_saving" )
        {
            m_memory_saving = EvalBool(args[1]);
        }
        if ( args.size() == 2 && args[0] == "batch_norm" )
        {
            m_bn_enable = EvalBool(args[1]);
        }
        if ( args.size() == 2 && args[0] == "activation" )
        {
            m_act_enable = EvalBool(args[1]);
        }
    }


public:
    ~BinaryDenseAffine() {}

    static std::shared_ptr< BinaryDenseAffine > Create(create_t const &create)
    {
        return std::shared_ptr< BinaryDenseAffine >(new BinaryDenseAffine(create));
    }

    static std::shared_ptr< BinaryDenseAffine > Create(indices_t const &output_shape, bool batch_norm = true, std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.batch_norm   = batch_norm;
        create.seed         = seed;
        return Create(create);
    }

    static std::shared_ptr<BinaryDenseAffine> Create(void)
    {
        return Create(create_t());
    }

#ifdef BB_PYBIND11
    static std::shared_ptr< BinaryDenseAffine > CreatePy(
                    indices_t const &output_shape,
                    bool            batch_norm     = true,
                    bool            activation     = true,
        
                    RealType        initialize_std = (RealType)0.01,
                    std::string     initializer    = "",
        
                    RealType        momentum       = (RealType)0.9,
                    RealType        gamma          = (RealType)1.0,
                    RealType        beta           = (RealType)0.0,
                    bool            fix_gamma      = false,
                    bool            fix_beta       = false,
                    
                    RealType        binary_th      = (RealType)0,
                    double          binary_low     = -1.0,
                    double          binary_high    = +1.0,
                    RealType        hardtanh_min   = (RealType)-1,
                    RealType        hardtanh_max   = (RealType)+1,
                    
                    std::uint64_t   seed           = 1,
                    bool            memory_saving  = true)
    {
        create_t create;

        create.output_shape   = output_shape;
        create.batch_norm     = batch_norm;
        create.activation     = activation;
        create.initialize_std = initialize_std;
        create.initializer    = initializer;
        create.momentum       = momentum;
        create.gamma          = gamma;
        create.beta           = beta;
        create.fix_gamma      = fix_gamma;
        create.fix_beta       = fix_beta;
        create.binary_th      = binary_th;
        create.hardtanh_min   = hardtanh_min;
        create.hardtanh_max   = hardtanh_max;
        create.seed           = seed;
        create.memory_saving  = memory_saving;

        return Create(create);
    }
#endif

    Tensor       &W(void)       { return m_affine->W(); }
    Tensor const &W(void) const { return m_affine->W(); }
    Tensor       &b(void)       { return m_affine->b(); }
    Tensor const &b(void) const { return m_affine->b(); }    
    Tensor       &dW(void)       { return m_affine->dW(); }
    Tensor const &dW(void) const { return m_affine->dW(); }
    Tensor       &db(void)       { return m_affine->db(); }
    Tensor const &db(void) const { return m_affine->db(); }
    Tensor       &gamma(void)           { return m_batch_norm->gamma(); }
    Tensor const &gamma(void) const     { return m_batch_norm->gamma(); }
    Tensor       &beta(void)            { return m_batch_norm->beta(); }
    Tensor const &beta(void) const      { return m_batch_norm->beta(); }
    Tensor       &dgamma(void)          { return m_batch_norm->dgamma(); }
    Tensor const &dgamma(void) const    { return m_batch_norm->dgamma(); }
    Tensor       &dbeta(void)           { return m_batch_norm->dbeta(); }
    Tensor const &dbeta(void) const     { return m_batch_norm->dbeta(); }
    Tensor        mean(void)            { return m_batch_norm->mean(); }
    Tensor        rstd(void)            { return m_batch_norm->rstd(); }
    Tensor        running_mean(void)    { return m_batch_norm->running_mean(); }
    Tensor        running_var(void)     { return m_batch_norm->running_var(); }


    /**
     * @brief  コマンドを送る
     * @detail コマンドを送る
     */   
    void SendCommand(std::string command, std::string send_to = "all") override
    {
        _super::SendCommand(command, send_to);

        m_affine    ->SendCommand(command, send_to);
        m_batch_norm->SendCommand(command, send_to);
        m_activation->SendCommand(command, send_to);
    }

    
    std::shared_ptr< AffineType >       GetAffine(void)             { return m_affine; }
    std::shared_ptr< BatchNormType >    GetBatchNormalization(void) { return m_batch_norm; }
    std::shared_ptr< ActType >          GetActivation(void)         { return m_activation; }


    /**
     * @brief  パラメータ取得
     * @detail パラメータを取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetParameters(void) override
    {
        Variables parameters;
        parameters.PushBack(m_affine    ->GetParameters());
        parameters.PushBack(m_batch_norm->GetParameters());
        return parameters;
    }

    /**
     * @brief  勾配取得
     * @detail 勾配を取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    virtual Variables GetGradients(void) override
    {
        Variables gradients;
        gradients.PushBack(m_affine    ->GetGradients());
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
    indices_t SetInputShape(indices_t shape) override
    {
        shape = m_affine->SetInputShape(shape);
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

    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {

        x_vec = m_affine->ForwardNode(node, x_vec);
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
        if ( m_memory_saving && train && (m_bn_enable || m_act_enable) ) {
            this->PushFrameBuffer(x_buf);

            x_buf = m_affine->Forward(x_buf, train);
            m_affine->Clear();
            if ( m_bn_enable ) {
                x_buf = m_batch_norm->Forward(x_buf, train);
                m_batch_norm->ClearBuffer();
            }
            if ( m_act_enable ) {
                x_buf = m_activation->Forward(x_buf, train);
                m_activation->Clear();
            }
        }
        else {
            x_buf = m_affine->Forward(x_buf, train);
            if ( m_bn_enable ) {
                x_buf = m_batch_norm->Forward(x_buf, train);
            }
            if ( m_act_enable ) {
                x_buf = m_activation->Forward(x_buf, train);
            }        
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
            FrameBuffer x_buf = this->PopFrameBuffer();
            x_buf = m_affine->ReForward(x_buf);
            if ( m_bn_enable ) {
                x_buf = m_batch_norm->ReForward(x_buf);
            }
            if ( m_act_enable ) {
                x_buf = m_activation->ReForward(x_buf);
            }
        }

        if ( m_act_enable ) { dy_buf = m_activation->Backward(dy_buf); }
        if ( m_bn_enable )  { dy_buf = m_batch_norm->Backward(dy_buf); }
        dy_buf = m_affine->Backward(dy_buf);
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
            m_affine->PrintInfo(depth, os, columns, nest+1);
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
        m_affine->DumpObject(os);
        m_batch_norm->DumpObject(os);
        m_activation->DumpObject(os);
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
        m_affine->LoadObject(is);
        m_batch_norm->LoadObject(is);
        m_activation->LoadObject(is);
    }

public:
    // Serialize(旧)
    void Save(std::ostream &os) const 
    {
        bb::SaveValue(os, m_bn_enable);
        bb::SaveValue(os, m_act_enable);
        bb::SaveValue(os, m_memory_saving);

        m_affine->Save(os);
        m_batch_norm->Save(os);
        m_activation->Save(os);
    }

    void Load(std::istream &is)
    {
        bb::LoadValue(is, m_bn_enable);
        bb::LoadValue(is, m_act_enable);
        bb::LoadValue(is, m_memory_saving);

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
        archive(cereal::make_nvp("BinaryDenseAffine", *this));
        m_affine    ->Save(archive);
        m_batch_norm->Save(archive);
        m_activation->Save(archive);
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("BinaryDenseAffine", *this));
        m_affine    ->Load(archive);
        m_batch_norm->Load(archive);
        m_activation->Load(archive);
    }
#endif

};


}