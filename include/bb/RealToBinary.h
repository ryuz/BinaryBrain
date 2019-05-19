// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>

#include "bb/Model.h"
#include "bb/ValueGenerator.h"


namespace bb {


/**
 * @brief   バイナリ変調を行いながらバイナライズを行う
 * @details 閾値を乱数などで変調して実数をバイナリに変換する
 *          入力に対して出力は frame_mux_size 倍のフレーム数となる
 *          入力値に応じて 0と1 を確率的に発生させることを目的としている
 *          RealToBinary と組み合わせて使う想定
 * 
 * @tparam FXT  foward入力型 (x)
 * @tparam FXT  foward出力型 (y)
 * @tparam BT   backward型 (dy, dx)
 */
template <typename FXT = float, typename FYT = float, typename BT = float>
class RealToBinary : public Model
{
protected:
    bool                                    m_binary_mode = true;
    bool                                    m_host_only = false;

    indices_t                               m_node_shape;
    index_t                                 m_modulation_size;
    std::shared_ptr< ValueGenerator<FXT> >  m_value_generator;
    bool                                    m_framewise;
    FXT                                     m_input_range_lo;
    FXT                                     m_input_range_hi;
    

public:
    struct create_t
    {
        index_t                                 modulation_size = 1;        //< 変調するフレームの単位
        std::shared_ptr< ValueGenerator<FXT> >  value_generator;            //< 閾値のジェネレーター
        bool                                    framewise = false;          //< true でフレーム単位で閾値、falseでデータ単位
        FXT                                     input_range_lo = (FXT)0.0;  //< 入力データの下限値
        FXT                                     input_range_hi = (FXT)1.0;  //< 入力データの上限値
    };

protected:
    RealToBinary(create_t const &create)
    {
        m_modulation_size = create.modulation_size;
        m_value_generator = create.value_generator;
        m_framewise       = create.framewise;
        m_input_range_lo  = create.input_range_lo;
        m_input_range_hi  = create.input_range_hi;
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args)
    {
        // バイナリモード設定
        if ( args.size() == 2 && args[0] == "binary" )
        {
            m_binary_mode = EvalBool(args[1]);
        }

        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }
    }


public:
    ~RealToBinary() {}


    static std::shared_ptr<RealToBinary> Create(create_t const &create)
    {
        return std::shared_ptr<RealToBinary>(new RealToBinary(create));
    }

    static std::shared_ptr<RealToBinary> Create(
                index_t                                 modulation_size = 1,
                std::shared_ptr< ValueGenerator<FXT> >  value_generator = nullptr,
                bool                                    framewise       = false,
                FXT                                     input_range_lo  = (FXT)0.0,
                FXT                                     input_range_hi  = (FXT)1.0)
    {
        create_t create;
        create.modulation_size  = modulation_size;
        create.value_generator  = value_generator;
        create.framewise        = framewise;
        create.input_range_lo   = input_range_lo;
        create.input_range_hi   = input_range_hi;
        return Create(create);
    }

    std::string GetClassName(void) const { return "RealToBinary"; }

    void SetModulationSize(index_t modulation_size)
    {
        m_modulation_size = modulation_size;
    }

    void SetValueGenerator(std::shared_ptr< ValueGenerator<FXT> > value_generator)
    {
        m_value_generator = value_generator;
    }

    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 形状設定
        m_node_shape = shape;
        return m_node_shape;
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
    

    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        if (!m_binary_mode) {
            return x_buf;
        }

        BB_ASSERT(x_buf.GetType() == DataType<FXT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_node_shape) {
            SetInputShape(x_buf.GetShape());
        }

        // 戻り値の型を設定
        FrameBuffer y_buf(DataType<FYT>::type, x_buf.GetFrameSize() * m_modulation_size, m_node_shape);

#ifdef BB_WITH_CUDA
        if ( DataType<FXT>::type == BB_TYPE_FP32 && DataType<FYT>::type == BB_TYPE_BIT && m_modulation_size == 1 && !m_host_only
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            auto x_ptr           = x_buf.LockDeviceMemoryConst();
            auto y_ptr           = y_buf.LockDeviceMemory(true);
            
            bbcu_fp32_bit_no_modulation_RealToBinary_Forward
                (
                    (const float *)x_ptr.GetAddr(),
                    (int         *)y_ptr.GetAddr(),
                    (float        )0.5f,
                    (int          )x_buf.GetNodeSize(),
                    (int          )x_buf.GetFrameSize(),
                    (int          )(x_buf.GetFrameStride() / sizeof(float)),
                    (int          )(y_buf.GetFrameStride() / sizeof(int))
                );

            return y_buf;
        }
#endif

        {
            // 汎用
            index_t node_size        = x_buf.GetNodeSize();
            index_t input_frame_size = x_buf.GetFrameSize();

            auto x_ptr = x_buf.LockConst<FXT>();
            auto y_ptr = y_buf.Lock<FYT>();

            FXT th_step = (m_input_range_hi - m_input_range_lo) / (FXT)(m_modulation_size + 1);
            for ( index_t input_frame = 0; input_frame < input_frame_size; ++input_frame) {
                for ( index_t i = 0; i < m_modulation_size; ++i ) {
                    index_t output_frame = input_frame * m_modulation_size + i;

                    if ( m_framewise || m_value_generator == nullptr ) {
                        // frame毎に閾値変調
                        FXT th;
                        if ( m_value_generator != nullptr ) {
                            th = m_value_generator->GetValue();
                            th = std::max(th, m_input_range_lo);
                            th = std::min(th, m_input_range_hi);
                        }
                        else {
                            th = m_input_range_lo + (th_step * (FXT)(i + 1));
                        }

                        #pragma omp parallel for
                        for (index_t node = 0; node < node_size; ++node) {
                            FXT real_sig = x_ptr.Get(input_frame, node);
                            FYT bin_sig  = (real_sig > th) ? (FYT)1 : (FYT)0;
                            y_ptr.Set(output_frame, node, bin_sig);
                        }
                    }
                    else {
                        // データ毎に閾値変調
                        for (index_t node = 0; node < node_size; ++node) {
                            FXT th = m_value_generator->GetValue();
                            FXT real_sig = x_ptr.Get(input_frame, node);
                            FYT bin_sig  = (real_sig > th) ? (FYT)1 : (FYT)0;
                            y_ptr.Set(output_frame, node, bin_sig);
                        }
                    }
                }
            }

            return y_buf;
        }
    }


    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        if (!m_binary_mode || m_modulation_size == 1) {
            return dy_buf;
        }

        BB_ASSERT(dy_buf.GetType() == DataType<BT>::type);

        // 戻り値の型を設定
        FrameBuffer dx_buf(DataType<BT>::type, dy_buf.GetFrameSize() / m_modulation_size, m_node_shape);

#if 0   // 今のところ計算結果誰も使わないので一旦コメントアウト
        index_t node_size         = dy_buf.GetNodeSize();
        index_t output_frame_size = dy_buf.GetFrameSize();

        dx_buf.FillZero();

        auto dy_ptr = dy_buf.LockConst<BT>();
        auto dx_ptr = dx_buf.Lock<BT>();

        #pragma omp parallel for
        for (index_t node = 0; node < node_size; node++) {
            for (index_t output_frame = 0; output_frame < output_frame_size; ++output_frame) {
                index_t input_frame = output_frame / m_frame_unit;

                BT grad = dy_ptr.Get(output_frame, node);
                dx_ptr.Add(input_frame, node, grad);
            }
        }
#endif

        return dx_buf;
    }
};


}


// end of file
