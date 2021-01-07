// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#ifdef BB_WITH_CEREAL
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#endif

#include "bb/Manager.h"
#include "bb/DataType.h"
#include "bb/Activation.h"
#include "bb/FrameBuffer.h"
#include "bb/SimdSupport.h"
#include "bb/StochasticBatchNormalization.h"

#ifdef BB_WITH_CUDA
#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#endif


namespace bb {


// BatchNormalization
template <typename FT = Bit, typename ST = float>
class BinaryScaling : public Activation
{
    using _super = Activation;

protected:
    bool                        m_host_only = false;
    bool                        m_host_simd = true;
    
    indices_t                   m_shape;

    FrameBuffer                 m_y_buf;

    Tensor_<ST>                 m_a;     // 0化する確率
    Tensor_<ST>                 m_b;     // 1化する確率

    std::mt19937_64             m_mt;

protected:
    BinaryScaling() {
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
    ~BinaryScaling() {}

    struct create_t
    {
        std::uint64_t   seed = 1;
    };

    static std::shared_ptr<BinaryScaling> Create(create_t const &create)
    {
        auto self = std::shared_ptr<BinaryScaling>(new BinaryScaling);
        self->m_mt.seed(create.seed);
        return self;
    }

    static std::shared_ptr<BinaryScaling> Create(std::uint64_t seed = 1)
    {
        auto self = std::shared_ptr<BinaryScaling>(new BinaryScaling);
        self->m_mt.seed(seed);
        return self;
    }

    std::string GetModelName(void) const { return "BinaryScaling"; }
    
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
        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

        m_shape = shape;

        m_a.Resize(m_shape);
        m_b.Resize(m_shape);

        return shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_shape;
    }


public:
  
    template<class T>
    void Import(std::shared_ptr<T> bn)
    {
        auto a_ptr = m_a.Lock();
        auto b_ptr = m_b.Lock();

        auto node_size  = CalcShapeSize(m_shape);

//      std::ofstream ofs("bn.txt", std::ios::app);

        for (index_t node = 0; node < node_size; ++node) {
            auto gain   = bn->GetNormalizeGain(node);
            auto offset = bn->GetNormalizeOffset(node);
            auto a      = offset / (gain + offset);
            auto b      = gain + offset;
#if 0
            ofs << node 
                << " gain="   << gain
                << " offset=" << offset
                << " a="      << a
                << " b="      << b << std::endl;
#endif
            a_ptr[node] = a;
            b_ptr[node] = b;
        }
    }

    void SetParameter(index_t node, ST gain, ST offset)
    {
        auto a_ptr = m_a.Lock();
        auto b_ptr = m_b.Lock();
        auto a = offset / (gain + offset);
        auto b = gain + offset;
//        BB_ASSERT(a >= (ST)0 && a <= (ST)1);
//        BB_ASSERT(b >= (ST)0 && b <= (ST)1);
        a_ptr[node] = offset / (gain + offset);
        b_ptr[node] = gain + offset;
    }

    // ノード単位でのForward計算
    std::vector<FT> ForwardNode(index_t node, std::vector<FT> x_vec) const
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
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        m_y_buf.Resize(x_buf.GetFrameSize(), x_buf.GetShape(), x_buf.GetType());

        {
            auto node_size  = x_buf.GetNodeSize();
            auto frame_size = x_buf.GetFrameSize();

            std::uniform_real_distribution<ST> uniform_dist((ST)0, (ST)1);

            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = m_y_buf.Lock<FT>(true);

            auto a_ptr = m_a.LockConst();
            auto b_ptr = m_b.LockConst();

            for (index_t node = 0; node < node_size; ++node) {
                ST a = a_ptr[node];
                ST b = b_ptr[node];
//              std::cout << "a=" << a << " b=" << b << std::endl;
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto x = x_ptr.Get(frame, node);
                    if ( uniform_dist(m_mt) < a ) { x = 1; }
                    if ( uniform_dist(m_mt) > b ) { x = 0; }
                    y_ptr.Set(frame, node, x);
                }
            }

            return m_y_buf;
        }
    }


    /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(0);
        return dy_buf;
    }
};

}
