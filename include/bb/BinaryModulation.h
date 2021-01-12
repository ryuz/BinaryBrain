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
#include "bb/ObjectReconstructor.h"


namespace bb {


template <typename BinType = float, typename RealType = float>
class BinaryModulation : public Model
{
    using _super = Model;

public:
    static inline std::string ClassName(void) { return "BinaryModulation"; }
    static inline std::string ObjectName(void){ return ClassName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const { return ClassName(); }
    std::string GetObjectName(void) const { return ObjectName(); }

protected:
    bool                                                m_binary_mode = true;
    bool                                                m_training;
    index_t                                             m_modulation_size = 1;

    typename RealToBinary<BinType, RealType>::create_t  m_training_create;
    typename RealToBinary<BinType, RealType>::create_t  m_inference_create;

    // 3層で構成
    std::shared_ptr< RealToBinary<BinType, RealType> >  m_real2bin;
    std::shared_ptr< Model >                            m_layer;
    std::shared_ptr< BinaryToReal<BinType, RealType> >  m_bin2real;

public:
    struct create_t
    {
        std::shared_ptr<Model>                      layer;
        indices_t                                   output_shape;
        index_t                                     depth_modulation_size = 1;

        index_t                                     training_modulation_size  = 1;
        std::shared_ptr< ValueGenerator<RealType> > training_value_generator;
        bool                                        training_framewise        = true;
        RealType                                    training_input_range_lo   = (RealType)0.0;
        RealType                                    training_input_range_hi   = (RealType)1.0;

        index_t                                     inference_modulation_size = 1;
        std::shared_ptr< ValueGenerator<RealType> > inference_value_generator; 
        bool                                        inference_framewise       = true;
        RealType                                    inference_input_range_lo  = (RealType)0.0;
        RealType                                    inference_input_range_hi  = (RealType)1.0;
    };

protected:
    BinaryModulation(create_t const &create)
    {
        m_training_create.depth_modulation_size = create.depth_modulation_size;
        m_training_create.frame_modulation_size = create.training_modulation_size;
        m_training_create.value_generator       = create.training_value_generator;
        m_training_create.framewise             = create.training_framewise;
        m_training_create.input_range_lo        = create.training_input_range_lo;
        m_training_create.input_range_hi        = create.training_input_range_hi;

        m_inference_create.depth_modulation_size = create.depth_modulation_size;
        m_inference_create.frame_modulation_size = create.inference_modulation_size;
        m_inference_create.value_generator       = create.inference_value_generator;
        m_inference_create.framewise             = create.inference_framewise;
        m_inference_create.input_range_lo        = create.inference_input_range_lo;
        m_inference_create.input_range_hi        = create.inference_input_range_hi;

        m_training = true;
        m_modulation_size = create.training_modulation_size;
        m_real2bin = RealToBinary<BinType, RealType>::Create(m_training_create);
        m_layer    = create.layer;
        m_bin2real = BinaryToReal<BinType, RealType>::Create(m_modulation_size, create.output_shape);
    }


    void CommandProc(std::vector<std::string> args)
    {
        // binary mode
        if ( DataType<BinType>::type != BB_TYPE_BIT ) {
            if ( args.size() == 2 && args[0] == "binary" )
            {
                m_binary_mode = EvalBool(args[1]);
            }
        }
    }

public:
    ~BinaryModulation() {}

    static std::shared_ptr<BinaryModulation> Create(create_t const & create)
    {
        return std::shared_ptr<BinaryModulation>(new BinaryModulation(create));
    }

    static std::shared_ptr<BinaryModulation> Create(std::shared_ptr<Model> layer, index_t train_modulation_size, index_t inference_modulation_size=0, index_t depth_modulation_size=1)
    {
        BB_ASSERT(train_modulation_size > 0);
        if ( inference_modulation_size <= 0 ) {
            inference_modulation_size = train_modulation_size;
        }

        create_t create;
        create.layer = layer;
        create.training_modulation_size  = train_modulation_size;
        create.inference_modulation_size = inference_modulation_size;
        create.depth_modulation_size       = depth_modulation_size;
        return Create(create);
    }

    static std::shared_ptr<BinaryModulation> Create(void)
    {
        return Create(create_t());
    }
    

#ifdef BB_PYBIND11
    static std::shared_ptr<BinaryModulation> CreatePy(
                std::shared_ptr<Model>                      layer,
                indices_t                                   output_shape,
                index_t                                     depth_modulation_size  = 1,
                
                index_t                                     training_modulation_size  = 1,
                std::shared_ptr< ValueGenerator<RealType> > training_value_generator  = nullptr,
                bool                                        training_framewise        = true,
                RealType                                    training_input_range_lo   = (RealType)0.0,
                RealType                                    training_input_range_hi   = (RealType)1.0,

                index_t                                     inference_modulation_size = 1,
                std::shared_ptr< ValueGenerator<RealType> > inference_value_generator = nullptr,
                bool                                        inference_framewise       = true,
                RealType                                    inference_input_range_lo  = (RealType)0.0,
                RealType                                    inference_input_range_hi  = (RealType)1.0
        )
    {
        BB_ASSERT(training_modulation_size > 0);
        if ( inference_modulation_size <= 0 ) {
            inference_modulation_size = training_modulation_size;
        }

        create_t create;
        create.layer                     = layer;
        create.output_shape              = output_shape;
        create.training_modulation_size  = training_modulation_size;
        create.training_value_generator  = training_value_generator;
        create.training_framewise        = training_framewise;
        create.training_input_range_lo   = training_input_range_lo;
        create.training_input_range_hi   = training_input_range_hi;
        create.inference_modulation_size = inference_modulation_size;
        create.inference_value_generator = inference_value_generator;
        create.inference_framewise       = inference_framewise;
        create.inference_input_range_lo  = inference_input_range_lo;
        create.inference_input_range_hi  = inference_input_range_hi;
        return Create(create);
    }
#endif
    

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
        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

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
    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        // bypass
        if ( !m_binary_mode ) {
            return m_layer->Forward(x_buf, train);
        }

        // change mode
        if (train && !m_training) {
            m_training = true;
            m_modulation_size = m_training_create.frame_modulation_size;

            m_real2bin->SetModulationSize(m_training_create.frame_modulation_size);
            m_real2bin->SetValueGenerator(m_training_create.value_generator);
            m_bin2real->SetFrameIntegrationSize(m_training_create.frame_modulation_size);
        }
        else if (!train && m_training) {
            m_training = false;
            m_modulation_size = m_inference_create.frame_modulation_size;

            m_real2bin->SetModulationSize(m_inference_create.frame_modulation_size);
            m_real2bin->SetValueGenerator(m_inference_create.value_generator);
            m_bin2real->SetFrameIntegrationSize(m_inference_create.frame_modulation_size);
        }

        x_buf = m_real2bin->Forward(x_buf, train);
        x_buf = m_layer->Forward(x_buf, train);
        x_buf = m_bin2real->Forward(x_buf, train);
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
        if ( !m_binary_mode ) {
            return dy_buf = m_layer->Backward(dy_buf);
        }

        dy_buf = m_bin2real->Backward(dy_buf);
        dy_buf = m_layer   ->Backward(dy_buf);
        dy_buf = m_real2bin->Backward(dy_buf);
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
            os << indent << " training  modulation size : " << m_training_create.frame_modulation_size  << std::endl;
            os << indent << " inference modulation size : " << m_inference_create.frame_modulation_size << std::endl;
        }
        else {
            os << indent << " training  modulation size : " << m_training_create.frame_modulation_size  << std::endl;
            os << indent << " inference modulation size : " << m_inference_create.frame_modulation_size << std::endl;

            // 子レイヤーの表示
            if ( m_binary_mode ) {
                m_real2bin->PrintInfo(depth, os, columns, nest+1);
                m_layer->PrintInfo(depth, os, columns, nest+1);
                m_bin2real->PrintInfo(depth, os, columns, nest+1);
            }
            else {
                m_layer->PrintInfo(depth, os, columns, nest+1);
            }
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
        bb::SaveValue(os, m_binary_mode);
        bb::SaveValue(os, m_training);
        bb::SaveValue(os, m_modulation_size);
        m_training_create.ObjectDump(os);
        m_inference_create.ObjectDump(os);

        m_real2bin->DumpObject(os);
        m_bin2real->DumpObject(os);

        bool has_layer = (bool)m_layer;
        bb::SaveValue(os, has_layer);
        if ( has_layer ) {
            m_layer->DumpObject(os);
        }
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
        bb::LoadValue(is, m_binary_mode);
        bb::LoadValue(is, m_training);
        bb::LoadValue(is, m_modulation_size);
        m_training_create.ObjectLoad(is);
        m_inference_create.ObjectLoad(is);

        m_real2bin->LoadObject(is);
        m_bin2real->LoadObject(is);

        bool has_layer;
        bb::LoadValue(is, has_layer);
        if ( has_layer ) {
            if ( m_layer ) {
                m_layer->LoadObject(is);
            }
            else {
#ifdef BB_OBJECT_RECONSTRUCTION
                m_layer = std::dynamic_pointer_cast<Model>(Object_Reconstruct(is));
#endif
            }
            BB_ASSERT(m_layer);
        }
    }


public:
    // Serialize(旧)
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

