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
protected:
    bool                m_binary_mode = true;
    bool                m_host_only   = false;

    index_t             m_modulation_size;

    indices_t           m_input_shape;
    indices_t           m_output_shape;

public:
    struct create_t
    {
        indices_t       output_shape;   
        index_t         modulation_size = 1;
    };

protected:
    BinaryToReal(create_t const &create)
    {
        m_output_shape    = create.output_shape;
        m_modulation_size = create.modulation_size;
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

    static std::shared_ptr<BinaryToReal> Create(index_t modulation_size=1, indices_t output_shape = indices_t())
    {
        create_t create;
        create.output_shape    = output_shape;
        create.modulation_size = modulation_size;
        return Create(create);
    }

    std::string GetClassName(void) const { return "BinaryToReal"; }

    
    void SetModulationSize(index_t modulation_size)
    {
        m_modulation_size = modulation_size;
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
        m_input_shape = shape;

        if (m_output_shape.empty()) {
            m_output_shape = m_input_shape;
        }

        // 整数倍の多重化のみ許容
        BB_ASSERT(GetShapeSize(m_input_shape) >= GetShapeSize(m_output_shape));
        BB_ASSERT(GetShapeSize(m_input_shape) % GetShapeSize(m_output_shape) == 0);

        return m_output_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_input_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_output_shape;
    }
    

    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
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
        BB_ASSERT(x_buf.GetFrameSize() % m_modulation_size == 0);
        FrameBuffer y_buf(x_buf.GetFrameSize() / m_modulation_size, m_output_shape, DataType<RealType>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
            && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);

            bbcu_fp32_BinaryToReal_Forward
                (
                    (float const *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int          )(GetShapeSize(m_input_shape) / GetShapeSize(m_output_shape)),
                    (int          )m_modulation_size,
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
                    (int          )(GetShapeSize(m_input_shape) / GetShapeSize(m_output_shape)),
                    (int          )m_modulation_size,
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
                    for (index_t i = 0; i < m_modulation_size; ++i) {
                        RealType bin_sig = (RealType)x_ptr.Get(frame*m_modulation_size + i, node);
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
    

    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        if ( !m_binary_mode || (m_modulation_size == 1 && m_input_shape == m_output_shape) ) {
            return dy_buf;
        }
        
        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        // 戻り値の型を設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize() * m_modulation_size, m_input_shape, DataType<RealType>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only 
                && dy_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {

            auto dy_ptr = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr = dx_buf.LockDeviceMemory(true);

            bbcu_fp32_BinaryToReal_Backward
                (
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (int          )(GetShapeSize(m_input_shape) / GetShapeSize(m_output_shape)),
                    (int          )m_modulation_size,
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

            RealType  gain = (RealType)output_node_size / ((RealType)input_node_size * (RealType)m_modulation_size);
            for (index_t node = 0; node < input_node_size; node++) {
                for (index_t frame = 0; frame < output_frame_size; ++frame) {
                    for (index_t i = 0; i < m_modulation_size; i++) {
                        auto grad = dy_ptr.Get(frame, node % output_node_size);
                        grad *= gain;
                        dx_ptr.Set(frame*m_modulation_size + i, node, grad);
                    }
                }
            }

            return dx_buf;
        }
    }
};

}