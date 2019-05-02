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
template <typename FXT = float, typename FYT = float, typename BT = float>
class BinaryToReal : public Model
{
protected:
    bool                m_host_only = false;

    index_t             m_frame_unit;

    indices_t           m_input_shape;
    indices_t           m_output_shape;

    FrameBuffer         m_y_buf;
    FrameBuffer         m_dx_buf;


protected:
    BinaryToReal() {}

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
    ~BinaryToReal() {}

    struct create_t
    {
        indices_t       output_shape;   
        index_t         frame_unit = 1;
    };

    static std::shared_ptr<BinaryToReal> Create(create_t const &create)
    {
        auto self = std::shared_ptr<BinaryToReal>(new BinaryToReal);

        self->m_output_shape = create.output_shape;
        self->m_frame_unit   = create.frame_unit;

        return self;
    }

    static std::shared_ptr<BinaryToReal> Create(indices_t output_shape, index_t frame_unit=1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.frame_unit   = frame_unit;
        return Create(create);
    }

    std::string GetClassName(void) const { return "BinaryToReal"; }

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
        BB_ASSERT(x_buf.GetType() == DataType<FXT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_input_shape) {
            SetInputShape(x_buf.GetShape());
        }

        // 戻り値の型を設定
        BB_ASSERT(x_buf.GetFrameSize() % m_frame_unit == 0);
        m_y_buf.Resize(DataType<FYT>::type, x_buf.GetFrameSize() / m_frame_unit, m_output_shape);

#ifdef BB_WITH_CUDA
        if ( DataType<FXT>::type == BB_TYPE_FP32 && !m_host_only && DataType<FYT>::type == BB_TYPE_FP32
            && x_buf.IsDeviceAvailable() && m_y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = m_y_buf.LockDeviceMemory(true);

            bbcu_fp32_BinaryToReal_Forward
                (
                    (float const *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int          )(GetShapeSize(m_input_shape) / GetShapeSize(m_output_shape)),
                    (int          )m_frame_unit,
                    (int          )GetOutputNodeSize(),
                    (int          )(x_buf.GetFrameStride() / sizeof(float)),
                    (int          )m_y_buf.GetFrameSize(),
                    (int          )(m_y_buf.GetFrameStride() / sizeof(float))
                );

            return m_y_buf;
        }
#endif

        {
            auto x_ptr = x_buf.LockConst<FXT>();
            auto y_ptr = m_y_buf.Lock<FYT>(true);

            index_t input_node_size   = GetInputNodeSize();
            index_t output_node_size  = GetOutputNodeSize();
            index_t output_frame_size = m_y_buf.GetFrameSize();

            index_t node_size = std::max(input_node_size, output_node_size);

            std::vector<FYT>    vec_v(output_node_size, (FYT)0);
            std::vector<int>    vec_n(output_node_size, 0);
            for (index_t frame = 0; frame < output_frame_size; ++frame) {
                std::fill(vec_v.begin(), vec_v.end(), (FYT)0);
                std::fill(vec_n.begin(), vec_n.end(), 0);
                for (index_t node = 0; node < node_size; ++node) {
                    for (index_t i = 0; i < m_frame_unit; ++i) {
                        FYT bin_sig = (FYT)x_ptr.Get(frame*m_frame_unit + i, node);
                        vec_v[node % output_node_size] += bin_sig;
                        vec_n[node % output_node_size] += 1;
                    }
                }

                for (index_t node = 0; node < output_node_size; ++node) {
                    y_ptr.Set(frame, node, (FYT)vec_v[node] / vec_n[node]);
                }
            }

            return m_y_buf;
        }
    }
    

    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(dy_buf.GetType() == DataType<BT>::type);

        // 戻り値の型を設定
        m_dx_buf.Resize(DataType<BT>::type, dy_buf.GetFrameSize() * m_frame_unit, m_input_shape);

#ifdef BB_WITH_CUDA
        if ( DataType<BT>::type == BB_TYPE_FP32 && !m_host_only 
                && dy_buf.IsDeviceAvailable() && m_dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {

            auto dy_ptr = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr = m_dx_buf.LockDeviceMemory(true);

            bbcu_fp32_BinaryToReal_Backward
                (
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (int          )(GetShapeSize(m_input_shape) / GetShapeSize(m_output_shape)),
                    (int          )m_frame_unit,
                    (int          )GetOutputNodeSize(),
                    (int          )(m_dx_buf.GetFrameStride() / sizeof(float)),
                    (int          )dy_buf.GetFrameSize(),
                    (int          )(dy_buf.GetFrameStride() / sizeof(float))
                );

            return m_dx_buf;
        }
#endif

        {
            index_t input_node_size   = GetInputNodeSize();
            index_t output_node_size  = GetOutputNodeSize();
            index_t output_frame_size = dy_buf.GetFrameSize();

            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = m_dx_buf.Lock<BT>();

            BT  gain = (BT)output_node_size / ((BT)input_node_size * (BT)m_frame_unit);
            for (index_t node = 0; node < input_node_size; node++) {
                for (index_t frame = 0; frame < output_frame_size; ++frame) {
                    for (index_t i = 0; i < m_frame_unit; i++) {
                        auto grad = dy_ptr.Get(frame, node % output_node_size);
                        grad *= gain;
                        dx_ptr.Set(frame*m_frame_unit + i, node, grad);
                    }
                }
            }

            return m_dx_buf;
        }
    }
};

}