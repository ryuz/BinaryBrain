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
 * @tparam BinType  バイナリ型 (y)
 * @tparam RealType 実数型 (x, dx, dy)
 */
template <typename BinType = float, typename RealType = float>
class RealToBinary : public Model
{
    using _super = Model;

public:
    static inline std::string ClassName(void) { return "RealToBinary"; }
    static inline std::string ObjectName(void){ return ClassName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const { return ClassName(); }
    std::string GetObjectName(void) const { return ObjectName(); }

protected:
    bool                                        m_binary_mode = true;

    indices_t                                   m_x_node_shape;
    indices_t                                   m_y_node_shape;
    index_t                                     m_point_size = 0;
    index_t                                     m_x_depth_size = 0;
    index_t                                     m_y_depth_size = 0;

    index_t                                     m_frame_modulation_size;
    index_t                                     m_depth_modulation_size;
    std::shared_ptr< ValueGenerator<RealType> > m_value_generator;
    bool                                        m_framewise;
    RealType                                    m_input_range_lo;
    RealType                                    m_input_range_hi;
    

public:
    struct create_t
    {
        index_t                                     frame_modulation_size = 1;      //< フレーム方向へ変調するサイズ
        index_t                                     depth_modulation_size = 1;      //< 深さ方向へ変調するサイズ
        std::shared_ptr< ValueGenerator<RealType> > value_generator;                //< 閾値のジェネレーター
        bool                                        framewise = false;              //< true でフレーム単位で閾値、falseでデータ単位
        RealType                                    input_range_lo = (RealType)0.0; //< 入力データの下限値
        RealType                                    input_range_hi = (RealType)1.0; //< 入力データの上限値

        void ObjectDump(std::ostream& os) const
        {
            bb::SaveValue(os, frame_modulation_size);
            bb::SaveValue(os, depth_modulation_size);
            bb::SaveValue(os, framewise);
            bb::SaveValue(os, input_range_lo);
            bb::SaveValue(os, input_range_hi);

            bool has_value_generator = (bool)value_generator;
            bb::SaveValue(os, has_value_generator);
            if ( has_value_generator ) {
                value_generator->DumpObject(os);
            }
        }

        void ObjectLoad(std::istream& is)
        {
            bb::LoadValue(is, frame_modulation_size);
            bb::LoadValue(is, depth_modulation_size);
            bb::LoadValue(is, framewise);
            bb::LoadValue(is, input_range_lo);
            bb::LoadValue(is, input_range_hi);

            bool has_value_generator;
            bb::LoadValue(is, has_value_generator);
            if ( has_value_generator ) {
                value_generator->LoadObject(is);
            }
        }
    };

protected:
    RealToBinary(create_t const &create)
    {
        m_frame_modulation_size = create.frame_modulation_size;
        m_depth_modulation_size = create.depth_modulation_size;
        m_value_generator       = create.value_generator;
        m_framewise             = create.framewise;
        m_input_range_lo        = create.input_range_lo;
        m_input_range_hi        = create.input_range_hi;
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
    }

public:
    ~RealToBinary() {}


    static std::shared_ptr<RealToBinary> Create(create_t const &create)
    {
        return std::shared_ptr<RealToBinary>(new RealToBinary(create));
    }

    static std::shared_ptr<RealToBinary> Create(
                index_t                                     frame_modulation_size = 1,
                std::shared_ptr< ValueGenerator<RealType> > value_generator = nullptr,
                bool                                        framewise       = false,
                RealType                                    input_range_lo  = (RealType)0.0,
                RealType                                    input_range_hi  = (RealType)1.0)
    {
        create_t create;
        create.frame_modulation_size = frame_modulation_size;
        create.value_generator       = value_generator;
        create.framewise             = framewise;
        create.input_range_lo        = input_range_lo;
        create.input_range_hi        = input_range_hi;
        return Create(create);
    }

    static std::shared_ptr<RealToBinary> Create(
                index_t                                     frame_modulation_size,
                index_t                                     depth_modulation_size,
                std::shared_ptr< ValueGenerator<RealType> > value_generator = nullptr,
                bool                                        framewise       = false,
                RealType                                    input_range_lo  = (RealType)0.0,
                RealType                                    input_range_hi  = (RealType)1.0)
    {
        create_t create;
        create.frame_modulation_size = frame_modulation_size;
        create.depth_modulation_size = depth_modulation_size;
        create.value_generator       = value_generator;
        create.framewise             = framewise;
        create.input_range_lo        = input_range_lo;
        create.input_range_hi        = input_range_hi;
        return Create(create);
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<RealToBinary> CreatePy(
                index_t                                     frame_modulation_size = 1,
                index_t                                     depth_modulation_size = 1,
                std::shared_ptr< ValueGenerator<RealType> > value_generator = nullptr,
                bool                                        framewise       = false,
                RealType                                    input_range_lo  = (RealType)0.0,
                RealType                                    input_range_hi  = (RealType)1.0)
    {
        create_t create;
        create.frame_modulation_size = frame_modulation_size;
        create.depth_modulation_size = depth_modulation_size;
        create.value_generator       = value_generator;
        create.framewise             = framewise;
        create.input_range_lo        = input_range_lo;
        create.input_range_hi        = input_range_hi;
        return Create(create);
    }
#endif

    void SetModulationSize(index_t modulation_size)
    {
        m_frame_modulation_size = modulation_size;
    }

    void SetValueGenerator(std::shared_ptr< ValueGenerator<RealType> > value_generator)
    {
        m_value_generator = value_generator;
    }

    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape) override
    {
        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

        // 形状設定
        BB_ASSERT(shape.size() > 0);
        m_x_node_shape = shape;
        m_x_depth_size = m_x_node_shape[0];
        m_point_size   = CalcShapeSize(m_x_node_shape) / m_x_depth_size;
        m_y_depth_size = m_x_depth_size * m_depth_modulation_size;
        m_y_node_shape = m_x_node_shape;
        m_y_node_shape[0] = m_y_depth_size;

        return m_y_node_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const override
    {
        return m_x_node_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_y_node_shape;
    }
    

    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        if ( !m_binary_mode && m_depth_modulation_size == 1 ) {
            return x_buf;
        }

        BB_ASSERT(x_buf.GetType() == DataType<RealType>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_x_node_shape) {
            (void)SetInputShape(x_buf.GetShape());
        }

        // 戻り値の型を設定
        FrameBuffer y_buf(x_buf.GetFrameSize() * m_frame_modulation_size, m_y_node_shape, DataType<BinType>::type);

#ifdef BB_WITH_CUDA
        if ( m_value_generator == nullptr
                && DataType<BinType>::type != BB_TYPE_BIT && (int)DataType<BinType>::type == (int)DataType<RealType>::type
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {

            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);
            bbcu_RealToBinary_Forward<RealType>(
                    (RealType const *)x_ptr.GetAddr(),
                    (RealType       *)y_ptr.GetAddr(),
                    (unsigned int    )m_depth_modulation_size,
                    (unsigned int    )m_frame_modulation_size,
                    (RealType        )m_input_range_lo,
                    (RealType        )m_input_range_hi,
                    (unsigned int    )m_point_size,
                    (unsigned int    )m_x_depth_size,
                    (unsigned int    )x_buf.GetFrameSize(),
                    (unsigned int    )(x_buf.GetFrameStride() / sizeof(RealType)),
                    (unsigned int    )(y_buf.GetFrameStride() / sizeof(RealType)),
                    (bool            )m_binary_mode
                );

            return y_buf;
        }

        if ( m_value_generator == nullptr
                && DataType<BinType>::type == BB_TYPE_BIT
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {

            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);
            bbcu_bit_RealToBinary_Forward<RealType>(
                    (RealType const *)x_ptr.GetAddr(),
                    (int            *)y_ptr.GetAddr(),
                    (unsigned int    )m_depth_modulation_size,
                    (unsigned int    )m_frame_modulation_size,
                    (RealType        )m_input_range_lo,
                    (RealType        )m_input_range_hi,
                    (unsigned int    )m_point_size,
                    (unsigned int    )m_x_depth_size,
                    (unsigned int    )x_buf.GetFrameSize(),
                    (unsigned int    )(x_buf.GetFrameStride() / sizeof(RealType)),
                    (unsigned int    )(y_buf.GetFrameStride() / sizeof(int))
                );

            return y_buf;
        }
#endif

        if ( m_value_generator == nullptr ) {
            auto x_ptr = x_buf.LockConst<RealType>();
            auto y_ptr = y_buf.Lock<BinType>();
            
            auto input_frame_size = x_buf.GetFrameSize();
            auto frame_step       = (RealType)1.0 / (RealType)(m_frame_modulation_size);
            auto frame_step_recip = (RealType)m_frame_modulation_size;
            auto depth_step       = (RealType)1.0 / (RealType)m_depth_modulation_size;
            auto depth_step_recip = (RealType)m_depth_modulation_size;
            auto x_offset         = m_input_range_lo;
            auto x_scale          = (RealType)1.0 / (m_input_range_hi - m_input_range_lo);

            for ( index_t input_frame = 0; input_frame < input_frame_size; ++input_frame) {
                for ( index_t f = 0; f < m_frame_modulation_size; ++f ) {
                    auto output_frame = input_frame * m_frame_modulation_size + f;
                    for ( index_t input_depth = 0; input_depth < m_x_depth_size; ++input_depth ) {
                        for ( index_t p = 0; p < m_point_size; ++p ) {
                            auto input_node = input_depth * m_point_size + p;
                            auto x = (x_ptr.Get(input_frame, input_node) - x_offset) * x_scale;

                            for ( index_t d = 0; d < m_depth_modulation_size; ++d ) {
                                auto output_depth = input_depth * m_depth_modulation_size + d;
                                auto output_node  = output_depth * m_point_size + p;
                                auto y = x;

                                // modulation for depth
                                y = (y - (RealType)(d * depth_step)) * depth_step_recip;

                                // modulation for frame
                                y = (y - (RealType)(f * frame_step)) * frame_step_recip;

                                // clamp
                                y = std::max((RealType)0.0, std::min((RealType)1.0, y));

                                if ( m_binary_mode ) {
                                    y = (y > (RealType)0.5) ? (RealType)1.0 : (RealType)0.0;
                                }
                                y_ptr.Set(output_frame, output_node, (BinType)y);
                            }
                        }
                    }
                }
            }

            return y_buf;
        }
        
        if ( m_value_generator != nullptr ) {
            auto x_ptr = x_buf.LockConst<RealType>();
            auto y_ptr = y_buf.Lock<BinType>();
            
            auto input_frame_size = x_buf.GetFrameSize();
        //  auto frame_step       = (RealType)1.0 / (RealType)m_frame_modulation_size;
        //  auto frame_step_recip = (RealType)m_frame_modulation_size;
            auto depth_step       = (RealType)1.0 / (RealType)m_depth_modulation_size;
            auto depth_step_recip = (RealType)m_depth_modulation_size;
            auto x_offset         = m_input_range_lo;
            auto x_scale          = (RealType)1.0 / (m_input_range_hi - m_input_range_lo);

            for ( index_t input_frame = 0; input_frame < input_frame_size; ++input_frame) {
                for ( index_t f = 0; f < m_frame_modulation_size; ++f ) {
                    auto output_frame = input_frame * m_frame_modulation_size + f;
                    RealType th = m_value_generator->GetValue();

                    for ( index_t input_depth = 0; input_depth < m_x_depth_size; ++input_depth ) {
                        for ( index_t p = 0; p < m_point_size; ++p ) {
                            auto input_node = input_depth * m_point_size + p;
                            auto x = (x_ptr.Get(input_frame, input_node) - x_offset) * x_scale;

                            for ( index_t d = 0; d < m_depth_modulation_size; ++d ) {
                                auto output_depth = input_depth * m_depth_modulation_size + d;
                                auto output_node  = output_depth * m_point_size + p;
                                auto y = x;

                                // modulation for depth
                                y = (y - (RealType)(d * depth_step)) * depth_step_recip;

                                // binarize for frame
                                y = (y > th) ? (RealType)1.0 : (RealType)0.0;

                                y_ptr.Set(output_frame, output_node, (BinType)y);
                            }

                            if ( !m_framewise ) {
                                th = m_value_generator->GetValue();
                            }
                        }
                    }
                }
            }

            return y_buf;
        }

        index_t node_size        = x_buf.GetNodeSize();
        index_t input_frame_size = x_buf.GetFrameSize();

        auto x_ptr = x_buf.LockConst<RealType>();
        auto y_ptr = y_buf.Lock<BinType>();

        RealType th_step = (m_input_range_hi - m_input_range_lo) / (RealType)(m_frame_modulation_size + 1);
        for ( index_t input_frame = 0; input_frame < input_frame_size; ++input_frame) {
            for ( index_t i = 0; i < m_frame_modulation_size; ++i ) {
                index_t output_frame = input_frame * m_frame_modulation_size + i;
                if ( m_framewise || m_value_generator == nullptr ) {
                    // frame毎に閾値変調
                    RealType th;
                    if ( m_value_generator != nullptr ) {
                        th = m_value_generator->GetValue();
                        th = std::max(th, m_input_range_lo);
                        th = std::min(th, m_input_range_hi);
                    }
                    else {
                        th = m_input_range_lo + (th_step * (RealType)(i + 1));
                    }

                    #pragma omp parallel for
                    for (index_t node = 0; node < node_size; ++node) {
                        RealType x = x_ptr.Get(input_frame, node);
                        BinType  y = (x > th) ? (BinType)1 : (BinType)0;
                        y_ptr.Set(output_frame, node, y);
                    }
                }
                else {
                    // データ毎に閾値変調
                    for (index_t node = 0; node < node_size; ++node) {
                        RealType th = m_value_generator->GetValue();
                        RealType x = x_ptr.Get(input_frame, node);
                        BinType  y  = (x > th) ? (BinType)1 : (BinType)0;
                        y_ptr.Set(output_frame, node, y);
                    }
                }
            }
        }

        return y_buf;
    }


    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        if ( m_depth_modulation_size == 1 && (!m_binary_mode || m_frame_modulation_size == 1) ) {
            return dy_buf;
        }

        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        // 戻り値の型を設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize() / m_frame_modulation_size, m_x_node_shape, DataType<RealType>::type);

#if 0   // 今のところ計算結果誰も使わないので一旦コメントアウト
        index_t node_size         = dy_buf.GetNodeSize();
        index_t output_frame_size = dy_buf.GetFrameSize();

        dx_buf.FillZero();

        auto dy_ptr = dy_buf.LockConst<RealType>();
        auto dx_ptr = dx_buf.Lock<RealType>();

        #pragma omp parallel for
        for (index_t node = 0; node < node_size; node++) {
            for (index_t output_frame = 0; output_frame < output_frame_size; ++output_frame) {
                index_t input_frame = output_frame / m_frame_modulation_size;

                RealType dy = dy_ptr.Get(output_frame, node);
                dx_ptr.Add(input_frame, node, dy);
            }
        }
#endif

        return dx_buf;
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
        bb::SaveValue(os, m_x_node_shape);
        bb::SaveValue(os, m_y_node_shape);
        bb::SaveValue(os, m_point_size);
        bb::SaveValue(os, m_x_depth_size);
        bb::SaveValue(os, m_y_depth_size);
        bb::SaveValue(os, m_frame_modulation_size);
        bb::SaveValue(os, m_depth_modulation_size);
        bb::SaveValue(os, m_framewise);
        bb::SaveValue(os, m_input_range_lo);
        bb::SaveValue(os, m_input_range_hi);

        bool    has_value_generator = (bool)m_value_generator;
        bb::SaveValue(os, has_value_generator);
        if ( has_value_generator ) {
            m_value_generator->DumpObject(os);
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
        bb::LoadValue(is, m_x_node_shape);
        bb::LoadValue(is, m_y_node_shape);
        bb::LoadValue(is, m_point_size);
        bb::LoadValue(is, m_x_depth_size);
        bb::LoadValue(is, m_y_depth_size);
        bb::LoadValue(is, m_frame_modulation_size);
        bb::LoadValue(is, m_depth_modulation_size);
        bb::LoadValue(is, m_framewise);
        bb::LoadValue(is, m_input_range_lo);
        bb::LoadValue(is, m_input_range_hi);

        bool    has_value_generator;
        bb::LoadValue(is, has_value_generator);
        if ( has_value_generator ) {
            m_value_generator->LoadObject(is);
        }
    }
};


}


// end of file
