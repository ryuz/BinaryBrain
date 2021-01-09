// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <vector>
#include <random>

#include "bb/MaxPooling.h"


namespace bb {

// MaxPoolingクラス
template <typename FT = float, typename BT = float>
class StochasticMaxPooling : public MaxPooling<FT, BT>
{
protected:
    bool                m_host_only = false;

    index_t             m_filter_h_size;
    index_t             m_filter_w_size;

    index_t             m_input_w_size;
    index_t             m_input_h_size;
    index_t             m_input_c_size;
    index_t             m_output_w_size;
    index_t             m_output_h_size;
    index_t             m_output_c_size;

    indices_t           m_input_shape;
    indices_t           m_output_shape;

    FrameBuffer         m_x_buf;

protected:
    StochasticMaxPooling(index_t filter_h_size, index_t filter_w_size)
    {
        m_filter_h_size = filter_h_size;
        m_filter_w_size = filter_w_size;
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
    ~StochasticMaxPooling() {}
    
    static std::shared_ptr<StochasticMaxPooling> Create(index_t filter_h_size, index_t filter_w_size)
    {
        auto self = std::shared_ptr<StochasticMaxPooling>(new StochasticMaxPooling(filter_h_size, filter_w_size));
        return self;
    }
    

    std::string GetModelName(void) const { return "StochasticMaxPooling"; }

    index_t GetFilterHeight(void) const override  { return m_filter_h_size; }
    index_t GetFilterWidth(void) const override  { return m_filter_w_size; }

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
        
        BB_ASSERT(shape.size() == 3);

        m_input_c_size = shape[0];
        m_input_h_size = shape[1];
        m_input_w_size = shape[2];
        m_output_w_size = (m_input_w_size + m_filter_w_size - 1) / m_filter_w_size;
        m_output_h_size = (m_input_h_size + m_filter_h_size - 1) / m_filter_h_size;
        m_output_c_size = m_input_c_size;

        m_input_shape  = shape;
        m_output_shape = indices_t({m_output_c_size, m_output_h_size, m_output_w_size});

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
    
protected:

    /*
    inline void* GetInputPtr(NeuralNetBuffer<T>& buf, int c, int y, int x)
    {
        return buf.Lock((c*m_input_h_size + y)*m_input_w_size + x);
    }

    inline void* GetOutputPtr(NeuralNetBuffer<T>& buf, int c, int y, int x)
    {
        return buf.Lock((c*m_output_h_size + y)*m_output_w_size + x);
    }
    */

    inline index_t GetInputNode(index_t c, index_t y, index_t x)
    {
        return (c * m_input_h_size + y) * m_input_w_size + x;
    }

    inline index_t GetOutputNode(index_t c, index_t y, index_t x)
    {
        return (c * m_output_h_size + y) * m_output_w_size + x;
    }

public:
    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_input_shape) {
            SetInputShape(x_buf.GetShape());
        }

        // backwardの為に保存
        if ( train ) {
            m_x_buf = x_buf;
        }

        // 出力を設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), m_output_shape, DataType<FT>::type);
        
#ifdef BB_WITH_CUDA
        // CUDA版
        if ( DataType<FT>::type == BB_TYPE_FP32 && !m_host_only
                && m_filter_h_size == 2 && m_filter_w_size == 2
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory(true);
            bbcu_fp32_StochasticMaxPooling2x2_Forward
                (
                    (float const *)ptr_x.GetAddr(),
                    (float*       )ptr_y.GetAddr(),
                    (int          )m_input_w_size,
                    (int          )m_input_h_size,
                    (int          )m_output_w_size,
                    (int          )m_output_h_size,
                    (int          )m_output_c_size,
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float))
                );

            return y_buf;
        }
#endif
        

        // 汎用版実装
        {
            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = y_buf.Lock<FT>(true);

            auto frame_size = x_buf.GetFrameSize();

            #pragma omp parallel for
            for (index_t c = 0; c < m_input_c_size; ++c) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        for (index_t frame = 0; frame < frame_size; ++frame) {
                            // OR演算を実施(反転してANDを取って、出力反転)
                            BT out_sig = (BT)1.0;
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y*m_filter_h_size + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x*m_filter_w_size + fx;
                                        if ( ix < m_input_w_size ) {
                                            FT in_sig = x_ptr.Get(frame, {c, iy, ix});
                                            out_sig *= ((BT)1.0 - in_sig);
                                        }
                                    }
                                }
                            }
                            y_ptr.Set(frame, {c, y, x}, (FT)((BT)1.0 - out_sig));
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
        BB_ASSERT(dy_buf.GetShape().size() == 3);

        FrameBuffer dx_buf(dy_buf.GetFrameSize(), m_input_shape, DataType<BT>::type);

        FrameBuffer x_buf = m_x_buf;
        m_x_buf = FrameBuffer();


#ifdef BB_WITH_CUDA
        if ( DataType<BT>::type == BB_TYPE_FP32 && DataType<FT>::type == BB_TYPE_FP32 && !m_host_only
                && m_filter_h_size == 2 && m_filter_w_size == 2
                && x_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x  = x_buf.LockDeviceMemoryConst();
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = dx_buf.LockDeviceMemory(true);
            bbcu_fp32_StochasticMaxPooling2x2_Backward
                (
                    (float const *)ptr_x.GetAddr(),
                    (float const *)ptr_dy.GetAddr(),
                    (float       *)ptr_dx.GetAddr(),
                    (int          )m_input_w_size,
                    (int          )m_input_h_size,
                    (int          )m_output_w_size,
                    (int          )m_output_h_size,
                    (int          )m_output_c_size,
                    (int          )dy_buf.GetFrameSize(),
                    (int          )(dy_buf.GetFrameStride() / sizeof(float))
                );

            return dx_buf;
        }
#endif


        // 汎用版実装
        if (  m_filter_h_size == 2 && m_filter_w_size == 2 ) {
            auto x_ptr  = x_buf.LockConst<FT>();
            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = dx_buf.Lock<BT>(true);

            auto frame_size = x_buf.GetFrameSize();

            #pragma omp parallel for
            for (index_t c = 0; c < m_input_c_size; ++c) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        for (index_t frame = 0; frame < frame_size; ++frame) {
                            BT in_sig[2][2] = {{0, 0}, {0, 0}};
                            for (index_t fy = 0; fy < 2; ++fy) {
                                index_t iy = y*2 + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < 2; ++fx) {
                                        index_t ix = x*2 + fx;
                                        if ( ix < m_input_w_size ) {
                                            in_sig[fy][fx] = (BT)x_ptr.Get(frame, c, iy, ix);
                                        }
                                    }
                                }
                            }
                            BT out_grad = dy_ptr.Get(frame, c, y, x);
                            BT in_grad[2][2];
                            in_grad[0][0] = (BT)(out_grad * (1.0 - in_sig[0][1]) * (1.0 - in_sig[1][0]) * (1.0 - in_sig[1][1]));
                            in_grad[0][1] = (BT)(out_grad * (1.0 - in_sig[0][0]) * (1.0 - in_sig[1][0]) * (1.0 - in_sig[1][1]));
                            in_grad[1][0] = (BT)(out_grad * (1.0 - in_sig[0][0]) * (1.0 - in_sig[0][1]) * (1.0 - in_sig[1][1]));
                            in_grad[1][1] = (BT)(out_grad * (1.0 - in_sig[0][0]) * (1.0 - in_sig[0][1]) * (1.0 - in_sig[1][0]));
                            for (index_t fy = 0; fy < 2; ++fy) {
                                index_t iy = y*2 + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < 2; ++fx) {
                                        index_t ix = x*2 + fx;
                                        if ( ix < m_input_w_size ) {
                                            dx_ptr.Set(frame, c, iy, ix, in_grad[fy][fx]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return dx_buf;
        }

        // 汎用版実装(未着手：下記は MaxPooling のまま)
        BB_ASSERT(0);
#if 0
        {
            auto x_ptr  = x_buf.LockConst<FT>();
            auto y_ptr  = m_y_buf.LockConst<FT>();
            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = dx_buf.Lock<BT>(true);

            auto frame_size = m_x_buf.GetFrameSize();

            #pragma omp parallel for
            for (index_t c = 0; c < m_input_c_size; ++c) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        for (index_t frame = 0; frame < frame_size; ++frame) {
                            FT out_sig = y_ptr.Get(frame, c, y, x);
                            BT grad    = dy_ptr.Get(frame, c, y, x);
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y*m_filter_h_size + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x*m_filter_w_size + fx;
                                        if ( ix < m_input_w_size ) {
                                            FT in_sig  = x_ptr.Get(frame, c, iy, ix);
                                            dx_ptr.Set(frame, c, iy, ix, (in_sig == out_sig) ? grad : (BT)0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return dx_buf;
        }
#endif
    }
};


}