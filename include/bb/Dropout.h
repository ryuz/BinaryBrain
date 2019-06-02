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
class Dropout : public Activation<FT, BT>
{
protected:
    double                  m_rate = 0.5;
    std::mt19937_64         m_mt;
    Tensor_<std::int8_t>    m_mask;
    
    bool                    m_host_only = false;

    FrameBuffer             m_y_buf;
    FrameBuffer             m_dx_buf;

protected:
    Dropout() {}

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args)
    {
        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }
    }


public:
    static std::shared_ptr<Dropout> Create(double rate=0.5, std::uint64_t seed=1)
    {
        auto self = std::shared_ptr<Dropout>(new Dropout);
        self->m_rate = rate;
        self->m_mt.seed(seed);
        return self;
    }

    ~Dropout() {}

    std::string GetClassName(void) const { return "Dropout"; }
    
    
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
    inline FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
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
    inline FrameBuffer Backward(FrameBuffer dy_buf)
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
};


}
