// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                 Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                 https://github.com/ryuz
//                                 ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>
#include <random>

#include "bb/Model.h"


namespace bb {


template <typename FT = float, typename BT = float>
class UpSampling : public Model
{
protected:
    bool            m_host_only = false;

    indices_t       m_input_shape;
    
    index_t         m_filter_h_size;
    index_t         m_filter_w_size;
    bool            m_fill;

public:
    struct create_t
    {
        index_t     filter_h_size = 1;  //< 高さの倍率
        index_t     filter_w_size = 1;  //< 幅の倍率
        bool        fill = true;        //< 全画素埋めるかどうか
    };

protected:
    /**
     * @brief  コンストラクタ
     * @detail コンストラクタ
     * @param  create 生成情報
     */
    UpSampling(create_t const &create)
    {
        m_filter_h_size = create.filter_h_size;
        m_filter_w_size = create.filter_w_size;
        m_fill          = create.fill;
    }

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
    ~UpSampling() {}


    static std::shared_ptr<UpSampling> Create(create_t const & create)
    {
        return std::shared_ptr<UpSampling>(new UpSampling(create));
    }

    static std::shared_ptr<UpSampling> Create(index_t filter_h_size, index_t filter_w_size, bool fill=true)
    {
        create_t create;
        create.filter_h_size = filter_h_size;
        create.filter_w_size = filter_w_size;
        create.fill   = fill;
        return Create(create);
    }
    
    std::string GetClassName(void) const { return "UpSampling"; }

public:

    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        BB_ASSERT(shape.size() == 3);

        m_input_shape  = shape;
        index_t w_size = m_input_shape[0] * m_filter_w_size;
        index_t h_size = m_input_shape[1] * m_filter_h_size;
        index_t c_size = m_input_shape[2];
        return indices_t({w_size, h_size, c_size});
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
        index_t w_size = m_input_shape[0] * m_filter_w_size;
        index_t h_size = m_input_shape[1] * m_filter_h_size;
        index_t c_size = m_input_shape[2];
        return indices_t({w_size, h_size, c_size});
    }


    FrameBuffer Forward(FrameBuffer x_buf, bool train=true)
    {
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        FrameBuffer y_buf(x_buf.GetFrameSize(), GetOutputShape(), DataType<FT>::type);
        
#ifdef BB_WITH_CUDA
        if ( !m_host_only && DataType<FT>::type == BB_TYPE_FP32 && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // FP32 CUDA
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);
            bbcu_fp32_UpSampling_Forward
                (
                    (float const *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int          )m_input_shape[0],
                    (int          )m_input_shape[1],
                    (int          )m_input_shape[2],
                    (int          )m_filter_w_size,
                    (int          )m_filter_h_size,
                    (int          )(m_fill ? 1 : 0),
                    (int          )x_buf.GetFrameSize(),
                    (int          )(x_buf.GetFrameStride() / sizeof(int))
                );

            return y_buf;
        }
#endif

#ifdef BB_WITH_CUDA
        if ( !m_host_only && DataType<FT>::type == BB_TYPE_BIT && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // Bit CUDA
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);

            bbcu_bit_UpSampling_Forward
                (
                    (int const *)x_ptr.GetAddr(),
                    (int       *)y_ptr.GetAddr(),
                    (int        )m_input_shape[0],
                    (int        )m_input_shape[1],
                    (int        )m_input_shape[2],
                    (int        )m_filter_w_size,
                    (int        )m_filter_h_size,
                    (int        )(m_fill ? 1 : 0),
                    (int        )x_buf.GetFrameSize(),
                    (int        )(x_buf.GetFrameStride() / sizeof(int))
                );

            return y_buf;
        }
#endif

        {
            // 汎用版
            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = y_buf.Lock<FT>(true);

            index_t frame_size    = x_buf.GetFrameSize();
            index_t c_size        = m_input_shape[2];
            index_t input_h_size  = m_input_shape[1];
            index_t input_w_size  = m_input_shape[0];
            index_t output_h_size = input_h_size * m_filter_h_size;
            index_t output_w_size = input_w_size * m_filter_w_size;

            #pragma omp parallel for
            for (index_t c = 0; c < c_size; ++c) {
                for (index_t iy = 0; iy < input_h_size; ++iy) {
                    for (index_t ix = 0; ix < input_w_size; ++ix) {
                        index_t input_node = (c * input_h_size + iy) * input_w_size + ix;
                        for ( index_t frame = 0; frame < frame_size; ++frame ) {
                            auto x_val = x_ptr.Get(frame, input_node);
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t oy = iy * m_filter_h_size + fy;
                                for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                    index_t ox = ix * m_filter_w_size + fx;
                                    index_t output_node = (c * output_h_size + oy) * output_w_size + ox;
                                    if ( m_fill ) {
                                        y_ptr.Set(frame, output_node, x_val);
                                    }
                                    else {
                                        if ( fx == (m_filter_w_size / 2) && fy == (m_filter_h_size / 2) ) {
                                            y_ptr.Set(frame, output_node, x_val);
                                        }
                                        else {
                                            y_ptr.Set(frame, output_node, 0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return y_buf;
        }
    }
    
    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(dy_buf.GetType() == DataType<BT>::type);

        FrameBuffer dx_buf(dy_buf.GetFrameSize(), GetInputShape(), DataType<BT>::type);

#ifdef BB_WITH_CUDA
        if ( !m_host_only && DataType<BT>::type == BB_TYPE_FP32 && dy_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() )
        {
            auto dy_ptr = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr = dx_buf.LockDeviceMemory(true);

            bbcu_fp32_UpSampling_Backward
                (
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (int          )m_input_shape[0],
                    (int          )m_input_shape[1],
                    (int          )m_input_shape[2],
                    (int          )m_filter_w_size,
                    (int          )m_filter_h_size,
                    (int          )(m_fill ? 1 : 0),
                    (int          )dy_buf.GetFrameSize(),
                    (int          )(dy_buf.GetFrameStride() / sizeof(float))
                );

            return dx_buf;
        }
#endif

        {
            // 汎用版
            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = dx_buf.Lock<BT>(true);
            
            index_t frame_size    = dy_buf.GetFrameSize();
            index_t c_size        = m_input_shape[2];
            index_t input_h_size  = m_input_shape[1];
            index_t input_w_size  = m_input_shape[0];
            index_t output_h_size = input_h_size * m_filter_h_size;
            index_t output_w_size = input_w_size * m_filter_w_size;

            #pragma omp parallel for
            for (index_t c = 0; c < c_size; ++c) {
                for (index_t iy = 0; iy < input_h_size; ++iy) {
                    for (index_t ix = 0; ix < input_w_size; ++ix) {
                        index_t input_node = (c * input_h_size + iy) * input_w_size + ix;
                        for ( index_t frame = 0; frame < frame_size; ++frame ) {
                            BT dx_val = 0;
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t oy = iy * m_filter_h_size + fy;
                                for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                    index_t ox = ix * m_filter_w_size + fx;
                                    index_t output_node = (c * output_h_size + oy) * output_w_size + ox;
                                    if ( m_fill ) {
                                        dx_val += dy_ptr.Get(frame, output_node);
                                    }
                                    else {
                                        if ( fx == (m_filter_w_size / 2) && fy == (m_filter_h_size / 2) ) {
                                            dx_val += dy_ptr.Get(frame, output_node);
                                        }
                                    }
                                }
                            }
                            dx_ptr.Set(frame, input_node, dx_val);
                        }
                    }
                }
            }

            return dx_buf;
        }
    }
};


}