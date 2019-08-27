// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
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
 * @tparam FT   foward入力型 (x, y)
 * @tparam BT   backward型 (dy, dx)
 */
template <typename FT = float, typename BT = float>
class Reduce : public Model
{
protected:
    bool                m_host_only = false;

    indices_t           m_input_shape;
    indices_t           m_output_shape;

protected:
    Reduce() {}

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
    ~Reduce() {}

    struct create_t
    {
        indices_t       output_shape;   
    };

    static std::shared_ptr<Reduce> Create(create_t const &create)
    {
        auto self = std::shared_ptr<Reduce>(new Reduce);

        self->m_output_shape   = create.output_shape;

        return self;
    }

    static std::shared_ptr<Reduce> Create(index_t output_node)
    {
        return Create(indices_t({output_node}));
    }

    static std::shared_ptr<Reduce> Create(indices_t output_shape)
    {
        create_t create;
        create.output_shape = output_shape;
        return Create(create);
    }

    static std::shared_ptr<Reduce> CreateEx(indices_t output_shape)
    {
        create_t create;
        create.output_shape = output_shape;
        return Create(create);
    }


    std::string GetClassName(void) const { return "Reduce"; }

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

        // 整数倍の縮退のみ許容
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
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_input_shape) {
            SetInputShape(x_buf.GetShape());
        }

        // 戻り値の型を設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), m_output_shape, DataType<FT>::type);

#if 0 // #ifdef BB_WITH_CUDA
        if ( DataType<FT>::type == BB_TYPE_FP32 && !m_host_only && DataType<FT>::type == BB_TYPE_FP32
            && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);

            bbcu_fp32_BinaryToReal_Forward
                (
                    (float const *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int          )(GetShapeSize(m_input_shape) / GetShapeSize(m_output_shape)),
                    (int          )m_frame_mux_size,
                    (int          )GetOutputNodeSize(),
                    (int          )(x.GetFrameStride() / sizeof(float)),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float))
                );

            return y_buf;
        }
#endif

        {
            // 汎用版
            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = y_buf.Lock<FT>(true);

            index_t input_node_size   = GetInputNodeSize();
            index_t output_node_size  = GetOutputNodeSize();
            index_t frame_size        = y_buf.GetFrameSize();

            index_t mux_size          = input_node_size / output_node_size;

//          index_t node_size = std::max(input_node_size, output_node_size);

            #pragma omp parallel for
            for (index_t output_node = 0; output_node < output_node_size; ++output_node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    FT sum = 0;
                    for (index_t i = 0; i < mux_size; ++i) {
                        sum += x_ptr.Get(frame, output_node_size * i + output_node);
                    }
                    y_ptr.Set(frame, output_node, sum / (FT)mux_size);
                }
            }

            return y_buf;
        }
    }

    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(dy_buf.GetType() == DataType<BT>::type);

        // 戻り値の型を設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), m_input_shape, DataType<BT>::type);

#if 0 // #ifdef BB_WITH_CUDA
        if ( DataType<BT>::type == BB_TYPE_FP32 && !m_host_only 
                && dy_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {

            auto dy_ptr = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr = dx_buf.LockDeviceMemory(true);

            bbcu_fp32_BinaryToReal_Backward
                (
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (int          )(GetShapeSize(m_input_shape) / GetShapeSize(m_output_shape)),
                    (int          )m_frame_mux_size,
                    (int          )GetOutputNodeSize(),
                    (int          )(dx_buf.GetFrameStride() / sizeof(float)),
                    (int          )dy.GetFrameSize(),
                    (int          )(dy.GetFrameStride() / sizeof(float))
                );

            return dx_buf;
        }
#endif

        {
            // 汎用版
            index_t input_node_size   = GetInputNodeSize();
            index_t output_node_size  = GetOutputNodeSize();
            index_t frame_size        = dy_buf.GetFrameSize();

            index_t mux_size          = input_node_size / output_node_size;

            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = dx_buf.Lock<BT>();

            #pragma omp parallel for
            for (index_t output_node = 0; output_node < output_node_size; ++output_node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    BT dy = dy_ptr.Get(frame, output_node);
                    BT dx = dy / (BT)mux_size;
                    for (index_t i = 0; i < mux_size; i++) {
                        dx_ptr.Set(frame, output_node_size * i + output_node, dx);
                    }
                }
            }

            return dx_buf;
        }
    }
};

}


// end of file
