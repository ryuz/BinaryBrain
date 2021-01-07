// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include "bb/Manager.h"
#include "bb/DataType.h"
#include "bb/Model.h"
#include "bb/FrameBuffer.h"
#include "bb/SimdSupport.h"

#ifdef BB_WITH_CUDA
#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#endif


namespace bb {


// BatchNormalization
template <typename T = float>
class BackpropagatedBatchNormalization : public Model
{
    using _super = Model;

protected:
    bool                        m_host_only = false;
    bool                        m_host_simd = true;
    
    indices_t                   m_node_shape;

    FrameBuffer                 m_x_buf;
//  FrameBuffer                 m_dx_buf;

    T                           m_gain = (T)1.00;
    T                           m_beta = (T)0.99;

public:
    struct create_t
    {
        T   gain = (T)1.00;
        T   beta = (T)0.99;
    };

protected:
    BackpropagatedBatchNormalization(create_t const &create)
    {
        m_gain = create.gain;
        m_beta = create.beta;
    }

    void CommandProc(std::vector<std::string> args)
    {
        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }

        // Host SIMDモード設定
        if (args.size() == 2 && args[0] == "host_simd")
        {
            m_host_simd = EvalBool(args[1]);
        }
    }

public:
    ~BackpropagatedBatchNormalization() {}

    static std::shared_ptr<BackpropagatedBatchNormalization> Create(create_t const &create)
    {
        return std::shared_ptr<BackpropagatedBatchNormalization>(new BackpropagatedBatchNormalization(create));
    }

    static std::shared_ptr<BackpropagatedBatchNormalization> Create(T gain = (T)1.00, T beta = (T)0.99)
    {
        create_t create;
        create.gain = gain;
        create.beta = beta;
        return Create(create);
    }

    std::string GetModelName(void) const { return "BackpropagatedBatchNormalization"; }
    
    // Serialize
    void Save(std::ostream &os) const 
    {
          SaveIndices(os, m_node_shape);
          bb::SaveValue(os, m_gain);
          bb::SaveValue(os, m_beta);
    }

    void Load(std::istream &is)
    {
         m_node_shape = LoadIndices(is);
         bb::LoadValue(is, m_gain);
         bb::LoadValue(is, m_beta);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
        archive(cereal::make_nvp("node_shape", m_node_shape));
        archive(cereal::make_nvp("gain",       m_gain));
        archive(cereal::make_nvp("beta",       m_beta));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("node_shape", m_node_shape));
        archive(cereal::make_nvp("gain",       m_gain));
        archive(cereal::make_nvp("beta",       m_beta));
     }

    void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("BackpropagatedBatchNormalization", *this));
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("BackpropagatedBatchNormalization", *this));
    }
#endif


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

        m_node_shape = shape;
        return shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_node_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_node_shape;
    }


public:
   /**
     * @brief  パラメータ取得
     * @detail パラメータを取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetParameters(void)
    {
        Variables parameters;
        return parameters;
    }

    /**
     * @brief  勾配取得
     * @detail 勾配を取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetGradients(void)
    {
        Variables gradients;
        return gradients;
    }
    

    // ノード単位でのForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {
        return x_vec;
    }


    /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    FrameBuffer Forward(FrameBuffer x_buf, bool train=true)
    {
        // backwardの為に保存
        if ( train ) {
            m_x_buf = x_buf;
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
        // 無視できるゲインになったらバイパス
        if (m_gain <= (T)1.0e-14) {
            return dy_buf;
        }
        
        FrameBuffer x_buf = m_x_buf;
        m_x_buf = FrameBuffer();

        // 出力設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), dy_buf.GetShape(), dy_buf.GetType());

        {
            auto node_size  = dy_buf.GetNodeSize();
            auto frame_size = dy_buf.GetFrameSize();

            auto x_ptr           = x_buf.LockConst<T>();
            auto dy_ptr          = dy_buf.LockConst<T>();
            auto dx_ptr          = dx_buf.Lock<T>(true);

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                T mean = 0;
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    mean += x_ptr.Get(frame, node);
                }
                mean /= frame_size;

                T var = 0;
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto d = x_ptr.Get(frame, node) - mean;
                    var += d * d;
                }
                var /= frame_size;
                T std = std::sqrt(var);

                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto x = x_ptr.Get(frame, node);
                    auto t = (x - mean) / (std + (T)10e-7);
                    t = (t * (T)0.2) + (T)0.5;

                    auto dy = dy_ptr.Get(frame, node);
                    dx_ptr.Set(frame, node, dy + (x - t) * m_gain);
                }
            }

            // ゲイン減衰
            m_gain *= m_beta;

            return dx_buf;
        }
    }
};

}
