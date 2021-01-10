// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Manager.h"
#include "bb/Activation.h"


namespace bb {


// Dropout
template <typename FT = float, typename BT = float>
class Dropout : public Activation
{
    using _super = Activation;

public:
    static inline std::string ModelName(void) { return "Dropout"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<FT>::Name() + "_" + DataType<BT>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }


protected:
    bool                    m_host_only = false;

    double                  m_rate = 0.5;
    std::mt19937_64         m_mt;
    Tensor_<std::int8_t>    m_mask;
    
    FrameBuffer             m_y_buf;
    FrameBuffer             m_dx_buf;


    struct create_t
    {
        double          rate = 0.5;
        std::uint64_t   seed = 1;
    };

protected:
    Dropout(create_t const &create)
    {
        m_rate = create.rate;
        m_mt.seed(create.seed);
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args) override
    {
        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }
    }

public:
    ~Dropout() {}

    static std::shared_ptr<Dropout> Create(create_t const &create)
    {
        return std::shared_ptr<Dropout>(new Dropout(create));
    }

    static std::shared_ptr<Dropout> Create(double rate=0.5, std::uint64_t seed=1)
    {
        create_t create;
        create.rate = rate;
        create.seed = seed;
        return Create(create);
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<Dropout> CreatePy(double rate=0.5, std::uint64_t seed=1)
    {
        create_t create;
        create.rate = rate;
        create.seed = seed;
        return Create(create);
    }
#endif
    
    
    // ノード単位でのForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const override
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
    inline FrameBuffer Forward(FrameBuffer x_buf, bool train = true) override
    {
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        // 戻り値のサイズ設定
        m_y_buf.ResizeLike(x_buf);

        m_mask.Resize(x_buf.GetNodeSize());

        {
            index_t frame_size = x_buf.GetFrameSize();
            index_t node_size  = x_buf.GetNodeSize();

            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = m_y_buf.Lock<FT>(true);

            if (train) {
                // generate mask
                auto mask_ptr = m_mask.Lock(true);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                for (index_t node = 0; node < node_size; ++node) {
                    mask_ptr[node] = (dist(m_mt) > m_rate) ? 0xff : 0;
                }

                #pragma omp parallel for
                for (index_t node = 0; node < node_size; ++node) {
                    if (mask_ptr[node] != 0) {
                        for (index_t frame = 0; frame < frame_size; ++frame) {
                            y_ptr.Set(frame, node, x_ptr.Get(frame, node));
                        }
                    }
                    else {
                        for (index_t frame = 0; frame < frame_size; ++frame) {
                            y_ptr.Set(frame, node, (FT)0);
                        }
                    }
                }
            }
            else {
                #pragma omp parallel for
                for (index_t node = 0; node < node_size; ++node) {
                    for (index_t frame = 0; frame < frame_size; ++frame) {
                        y_ptr.Set(frame, node, x_ptr.Get(frame, node) * (FT)(1.0 - m_rate));
                    }
                }
            }
        }

        return m_y_buf;
    }


   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    inline FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        BB_ASSERT(dy_buf.GetType() == DataType<BT>::type);

        // 戻り値のサイズ設定
        m_dx_buf.ResizeLike(dy_buf);

        {
            index_t frame_size = m_dx_buf.GetFrameSize();
            index_t node_size = m_dx_buf.GetNodeSize();
            
            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = m_dx_buf.Lock<BT>(true);
            auto mask_ptr = m_mask.LockConst();

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                if ( mask_ptr[node] != 0 ) {
                    for (index_t frame = 0; frame < frame_size; ++frame) {
                        dx_ptr.Set(frame, node, dy_ptr.Get(frame, node));
                    }
                }
                else {
                    for (index_t frame = 0; frame < frame_size; ++frame) {
                        dx_ptr.Set(frame, node, 0);
                    }
                }
            }

            return m_dx_buf;
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
        bb::SaveValue(os, m_host_only);
        bb::SaveValue(os, m_rate);
        m_mask.DumpObject(os);
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
        bb::LoadValue(is, m_host_only);
        bb::LoadValue(is, m_rate);
        m_mask.LoadObject(is);

        
        // 再構築
    }

};


}
