// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

#include "bb/Manager.h"
#include "bb/DataType.h"
#include "bb/Activation.h"
#include "bb/FrameBuffer.h"
#include "bb/SimdSupport.h"

#if BB_WITH_CUDA
#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#endif


namespace bb {


// BatchNormalization
template <typename T = float>
class StochasticBatchNormalization : public Activation<T, T>
{
    using _super = Activation<T, T>;

protected:
    bool                        m_host_only = false;
    bool                        m_host_simd = true;
    
    FrameBuffer                 m_x_buf;
    FrameBuffer                 m_dx_buf;

    T                           m_gain = (T)1.0;

protected:
    StochasticBatchNormalization() {
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
    ~StochasticBatchNormalization() {}

    struct create_t
    {
        T   gain = (T)1.0;
    };

    static std::shared_ptr<StochasticBatchNormalization> Create(create_t const &create)
    {
        auto self = std::shared_ptr<StochasticBatchNormalization>(new StochasticBatchNormalization);
        self->m_gain = create.gain;
        return self;
    }

    static std::shared_ptr<StochasticBatchNormalization> Create(T gain = (T)0.1)
    {
        auto self = std::shared_ptr<StochasticBatchNormalization>(new StochasticBatchNormalization);
        self->m_gain = gain;
        return self;
    }

    std::string GetClassName(void) const { return "StochasticBatchNormalization"; }
    
    // Serialize
    void Save(std::ostream &os) const 
    {
//        SaveIndex(os, m_node_size);
//        bb::SaveValue(os, m_gain);
    }

    void Load(std::istream &is)
    {
//       m_node_size = LoadIndex(is);
//       bb::LoadValue(is, m_gain);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
//        _super::save(archive, version);
//        archive(cereal::make_nvp("node_size",    m_node_size));
//        archive(cereal::make_nvp("gain",         m_gain));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
//        _super::load(archive, version);
//        archive(cereal::make_nvp("node_size",    m_node_size));
//        archive(cereal::make_nvp("gain",         m_gamma));
     }

    void Save(cereal::JSONOutputArchive& archive) const
    {
//        archive(cereal::make_nvp("StochasticBatchNormalization", *this));
    }

    void Load(cereal::JSONInputArchive& archive)
    {
//        archive(cereal::make_nvp("StochasticBatchNormalization", *this));
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
        return shape;
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
    std::vector<T> ForwardNode(index_t node, std::vector<T> x_vec) const
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
        // forwardの為に保存
        m_x_buf = x_buf;

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
        // 出力設定
        m_dx_buf.Resize(dy_buf.GetType(), dy_buf.GetFrameSize(), dy_buf.GetShape());

        {
            auto node_size  = dy_buf.GetNodeSize();
            auto frame_size = dy_buf.GetFrameSize();

            auto x_ptr           = m_x_buf.LockConst<T>();
            auto dy_ptr          = dy_buf.LockConst<T>();
            auto dx_ptr          = m_dx_buf.Lock<T>(true);

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                T sum = 0;
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    sum += (x_ptr.Get(frame, node) - (T)0.5);
                }
                T mean = sum / frame_size;

                for (index_t frame = 0; frame < frame_size; ++frame) {
                    dx_ptr.Set(frame, node, dy_ptr.Get(frame, node) + mean * m_gain);
                }
            }

            return m_dx_buf;
        }
    }
};

}
