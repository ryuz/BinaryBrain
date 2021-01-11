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


namespace bb {


/**
 * @brief   バイナリ変調したデータを積算して実数に戻す
 * @details バイナリ変調したデータを積算して実数に戻す
 *          出力に対して入力は frame_mux_size 倍のフレーム数を必要とする
 *          BinaryToReal と組み合わせて使う想定
 * 
 * @tparam FXT  foward入力型 (x)
 * @tparam FXT  foward出力型 (y)
 * @tparam BT   backward型 (dy, dx)
 */
template <typename BinType = float, typename RealType = float>
class BinaryToReal : public Model
{
    using _super = Model;

public:
    static inline std::string ClassName(void) { return "BinaryToReal"; }
    static inline std::string ObjectName(void){ return ClassName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const { return ClassName(); }
    std::string GetObjectName(void) const { return ObjectName(); }


protected:
    bool                m_host_only   = false;
    bool                m_binary_mode = true;

    index_t             m_frame_integration_size = 1;
    index_t             m_depth_integration_size = 0;

    indices_t           m_input_shape;
    indices_t           m_output_shape;

public:
    struct create_t
    {
        indices_t       output_shape;   
        index_t         frame_integration_size = 1;
        index_t         depth_integration_size = 0;     // 0 時は output_shape に従う
    };

protected:
    BinaryToReal(create_t const &create)
    {
        m_output_shape           = create.output_shape;
        m_frame_integration_size = create.frame_integration_size;
        m_depth_integration_size = create.depth_integration_size;
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
    ~BinaryToReal() {}

    static std::shared_ptr<BinaryToReal> Create(create_t const &create)
    {
        return std::shared_ptr<BinaryToReal>(new BinaryToReal(create));
    }

    static std::shared_ptr<BinaryToReal> Create(index_t frame_integration_size=1, indices_t output_shape = indices_t())
    {
        create_t create;
        create.output_shape    = output_shape;
        create.frame_integration_size = frame_integration_size;
        return Create(create);
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<BinaryToReal> CreatePy(index_t frame_integration_size=1, index_t depth_integration_size=0, indices_t output_shape = indices_t())
    {
        create_t create;
        create.output_shape    = output_shape;
        create.frame_integration_size = frame_integration_size;
        return Create(create);
    }
#endif
    
    void SetFrameIntegrationSize(index_t frame_integration_size)
    {
        m_frame_integration_size = frame_integration_size;
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
        m_input_shape = shape;

        if (m_output_shape.empty()) {
            m_output_shape = m_input_shape;
        }

        if (m_depth_integration_size > 0) {
            if ( m_output_shape.empty() || (CalcShapeSize(m_output_shape) != CalcShapeSize(m_input_shape) * m_depth_integration_size) ) {
                m_output_shape = m_input_shape;
                BB_ASSERT(m_output_shape[0] % m_depth_integration_size == 0);
                m_output_shape[0] /= m_depth_integration_size;
            }
        }
        
        // 整数倍の多重化のみ許容
        BB_ASSERT(CalcShapeSize(m_input_shape) >= CalcShapeSize(m_output_shape));
        BB_ASSERT(CalcShapeSize(m_input_shape) % CalcShapeSize(m_output_shape) == 0);

        return m_output_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const override
    {
        return m_input_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const override
    {
        return m_output_shape;
    }
    

    FrameBuffer Forward(FrameBuffer x_buf, bool train = true) override
    {
        if ( typeid(BinType) == typeid(RealType) && !m_binary_mode ) {
            return x_buf;
        }

        BB_ASSERT(x_buf.GetType() == DataType<BinType>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_input_shape) {
            SetInputShape(x_buf.GetShape());
        }

        // 戻り値の型を設定
        BB_ASSERT(x_buf.GetFrameSize() % m_frame_integration_size == 0);
        FrameBuffer y_buf(x_buf.GetFrameSize() / m_frame_integration_size, m_output_shape, DataType<RealType>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
            && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);

            bbcu_fp32_BinaryToReal_Forward
                (
                    (float const *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int          )(CalcShapeSize(m_input_shape) / CalcShapeSize(m_output_shape)),
                    (int          )m_frame_integration_size,
                    (int          )GetOutputNodeSize(),
                    (int          )(x_buf.GetFrameStride() / sizeof(float)),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float))
                );

            return y_buf;
        }

        if ( DataType<BinType>::type == BB_TYPE_BIT && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
            && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);

            bbcu_bit_fp32_BinaryToReal_Forward
                (
                    (int   const *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int          )(CalcShapeSize(m_input_shape) / CalcShapeSize(m_output_shape)),
                    (int          )m_frame_integration_size,
                    (int          )GetOutputNodeSize(),
                    (int          )(x_buf.GetFrameStride() / sizeof(int)),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float))
                );

            return y_buf;
        }
#endif

        {
            auto x_ptr = x_buf.LockConst<BinType>();
            auto y_ptr = y_buf.Lock<RealType>(true);

            index_t input_node_size   = GetInputNodeSize();
            index_t output_node_size  = GetOutputNodeSize();
            index_t output_frame_size = y_buf.GetFrameSize();

            index_t node_size = std::max(input_node_size, output_node_size);

            std::vector<RealType>   vec_v(output_node_size, (RealType)0);
            std::vector<int>        vec_n(output_node_size, 0);
            for (index_t frame = 0; frame < output_frame_size; ++frame) {
                std::fill(vec_v.begin(), vec_v.end(), (RealType)0);
                std::fill(vec_n.begin(), vec_n.end(), 0);
                for (index_t node = 0; node < node_size; ++node) {
                    for (index_t i = 0; i < m_frame_integration_size; ++i) {
                        RealType bin_sig = (RealType)x_ptr.Get(frame*m_frame_integration_size + i, node);
                        vec_v[node % output_node_size] += bin_sig;
                        vec_n[node % output_node_size] += 1;
                    }
                }

                for (index_t node = 0; node < output_node_size; ++node) {
                    y_ptr.Set(frame, node, vec_v[node] / (RealType)vec_n[node]);
                }
            }

            return y_buf;
        }
    }
    

    FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        if ( !m_binary_mode || (m_frame_integration_size == 1 && m_input_shape == m_output_shape) ) {
            return dy_buf;
        }
        
        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        // 戻り値の型を設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize() * m_frame_integration_size, m_input_shape, DataType<RealType>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only 
                && dy_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {

            auto dy_ptr = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr = dx_buf.LockDeviceMemory(true);

            bbcu_fp32_BinaryToReal_Backward
                (
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (int          )(CalcShapeSize(m_input_shape) / CalcShapeSize(m_output_shape)),
                    (int          )m_frame_integration_size,
                    (int          )GetOutputNodeSize(),
                    (int          )(dx_buf.GetFrameStride() / sizeof(float)),
                    (int          )dy_buf.GetFrameSize(),
                    (int          )(dy_buf.GetFrameStride() / sizeof(float))
                );

            return dx_buf;
        }
#endif

        {
            index_t input_node_size   = GetInputNodeSize();
            index_t output_node_size  = GetOutputNodeSize();
            index_t output_frame_size = dy_buf.GetFrameSize();

            auto dy_ptr = dy_buf.LockConst<RealType>();
            auto dx_ptr = dx_buf.Lock<RealType>();

            RealType  gain = (RealType)output_node_size / ((RealType)input_node_size * (RealType)m_frame_integration_size);
            for (index_t node = 0; node < input_node_size; node++) {
                for (index_t frame = 0; frame < output_frame_size; ++frame) {
                    for (index_t i = 0; i < m_frame_integration_size; i++) {
                        auto grad = dy_ptr.Get(frame, node % output_node_size);
                        grad *= gain;
                        dx_ptr.Set(frame*m_frame_integration_size + i, node, grad);
                    }
                }
            }

            return dx_buf;
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
        bb::SaveValue(os, m_binary_mode);
        bb::SaveValue(os, m_frame_integration_size);
        bb::SaveValue(os, m_depth_integration_size);
        bb::SaveValue(os, m_input_shape);
        bb::SaveValue(os, m_output_shape);
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
        bb::LoadValue(is, m_binary_mode);
        bb::LoadValue(is, m_frame_integration_size);
        bb::LoadValue(is, m_depth_integration_size);
        bb::LoadValue(is, m_input_shape);
        bb::LoadValue(is, m_output_shape);
    }

};

}