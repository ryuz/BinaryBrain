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

#include "bb/Filter2d.h"


namespace bb {

// MaxPoolingクラス
template <typename FT = float, typename BT = float>
class StochasticMaxPooling2x2 : public Filter2d<FT, BT>
{
protected:
    bool                m_host_only;

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
    StochasticMaxPooling2x2() {}

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
    ~StochasticMaxPooling2x2() {}
    
    static std::shared_ptr<StochasticMaxPooling2x2> Create()
    {
        auto self = std::shared_ptr<StochasticMaxPooling2x2>(new StochasticMaxPooling2x2);
        return self;
    }
    

    std::string GetClassName(void) const { return "StochasticMaxPooling2x2"; }

    index_t GetFilterHeight(void) { return 2; }
    index_t GetFilterWidth(void)  { return 2; }

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
        BB_ASSERT(shape.size() == 3);

        m_input_w_size = shape[0];
        m_input_h_size = shape[1];
        m_input_c_size = shape[2];
        m_output_w_size = (m_input_w_size + 2 - 1) / 2;
        m_output_h_size = (m_input_h_size + 2 - 1) / 2;
        m_output_c_size = m_input_c_size;

        m_input_shape  = shape;
        m_output_shape = indices_t({m_output_w_size, m_output_h_size, m_output_c_size});

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
        if ( DataType<FT>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
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
     
#if 0
        if ( DataType<FT>::type == BB_TYPE_BIT ) {
            // バイナリ用実装
            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = y_buf.Lock<FT>(true);

            index_t  m256_frame_size = (int)y.GetFrameStride() / 32;

            #pragma omp parallel for
            for (index_t c = 0; c < m_input_c_size; ++c) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        __m256i *y_addr = (__m256i *)y_ptr.GetAddr(GetOutputNode(c, y, x));

                        for (index_t frame = 0; frame < m256_frame_size; ++frame) {
                            __m256i max_val = _mm256_set1_epi8(0);
                            for (index_t fy = 0; fy < 2; ++fy) {
                                index_t iy = y*2 + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < 2; ++fx) {
                                        index_t ix = x*2 + fx;
                                        if ( ix < m_input_w_size ) {
                                            __m256i const *x_addr = (__m256i const *)x_ptr.GetAddr(GetInputNode(c, iy, ix));
                                            __m256i in_sig = _mm256_load_si256(&x_addr[frame]);
                                            max_val = _mm256_or_si256(max_val, in_sig);
                                        }
                                    }
                                }
                            }
                            _mm256_store_si256(&y_addr[frame], max_val);
                        }
                    }
                }
            }

            return y_buf;
        }
#endif

#if 0
        // float用実装
        if ( DataType<FT>::type == BB_TYPE_FP32 ) {
            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = y_buf.Lock<FT>(true);

            index_t  m256_frame_size = (int)y_buf.GetFrameStride() / sizeof(float);

            #pragma omp parallel for
            for (index_t c = 0; c < m_input_c_size; ++c) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        float *y_addr = (float *)y_ptr.GetAddr(GetOutputNode(c, y, x));

                        for (index_t frame = 0; frame < m256_frame_size; frame += 8) {
                            __m256  max_val = _mm256_set1_ps(-1.0e7f);  // 前段に活性化入れるから0がminだよね？
                            for (index_t fy = 0; fy < 2; ++fy) {
                                index_t iy = y*2 + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < 2; ++fx) {
                                        index_t ix = x*2 + fx;
                                        if ( ix < m_input_w_size ) {
                                            float const *x_addr = (float const *)x_ptr.GetAddr(GetInputNode(c, iy, ix));
                                            __m256 in_sig = _mm256_load_ps(&x_addr[frame]);
                                            max_val = _mm256_max_ps(max_val, in_sig);
                                        }
                                    }
                                }
                            }
                            _mm256_store_ps(&y_addr[frame], max_val);
                        }
                    }
                }
            }

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
                            FT out_sig = (FT)1.0;
                            for (index_t fy = 0; fy < 2; ++fy) {
                                index_t iy = y*2 + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < 2; ++fx) {
                                        index_t ix = x*2 + fx;
                                        if ( ix < m_input_w_size ) {
                                            FT in_sig = x_ptr.Get(frame, {ix, iy, c});
                                            out_sig *= ((FT)1.0 - in_sig);
                                        }
                                    }
                                }
                            }
                            y_ptr.Set(frame, {x, y, c}, ((FT)1.0 - out_sig));
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

#if 0
        if ( DataType<BT>::type == BB_TYPE_FP32 && DataType<FT>::type == BB_TYPE_FP32 ) {
            // float用実装
            index_t  m256_frame_size = dx_buf.GetFrameStride() / sizeof(float);

            auto x_ptr  = x_buf.LockConst<FT>();
            auto y_ptr  = y_buf.LockConst<FT>();
            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = dx_buf.Lock<BT>(true);

    #pragma omp parallel for
            for (index_t n = 0; n < m_input_c_size; ++n) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        float const * y_addr  = (float const *)y_ptr.GetAddr(GetOutputNode(n, y, x));
                        float const * dy_addr = (float const *)dy_ptr.GetAddr(GetOutputNode(n, y, x));

                        for (index_t frame = 0; frame < m256_frame_size; frame += 8) {
                            __m256 out_sig  = _mm256_load_ps(&y_addr[frame]);
                            __m256 out_grad = _mm256_load_ps(&dy_addr[frame]);
                            for (index_t fy = 0; fy < 2; ++fy) {
                                index_t iy = y*2 + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < 2; ++fx) {
                                        index_t ix = x*2 + fx;
                                        if ( ix < m_input_w_size ) {
                                            float const *x_addr  = (float const *)x_ptr.GetAddr(GetInputNode(n, iy, ix));
                                            float       *dx_addr = (float       *)dx_ptr.GetAddr(GetInputNode(n, iy, ix));
                                            __m256 in_sig  = _mm256_load_ps(&x_addr[frame]);
                                            __m256 mask    = _mm256_cmp_ps(in_sig, out_sig, _CMP_EQ_OQ);
                                            __m256 in_grad = _mm256_and_ps(mask, out_grad);
                                            _mm256_store_ps(&dx_addr[frame], in_grad);
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

        // 汎用版実装
        {
            auto x_ptr  = x_buf.LockConst<FT>();
            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = dx_buf.Lock<BT>(true);

            auto frame_size = x_buf.GetFrameSize();

            #pragma omp parallel for
            for (index_t c = 0; c < m_input_c_size; ++c) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        for (index_t frame = 0; frame < frame_size; ++frame) {
                            FT in_sig[2][2] = {{0, 0}, {0, 0}};
                            for (index_t fy = 0; fy < 2; ++fy) {
                                index_t iy = y*2 + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < 2; ++fx) {
                                        index_t ix = x*2 + fx;
                                        if ( ix < m_input_w_size ) {
                                            in_sig[fy][fx] = x_ptr.Get(frame, c, iy, ix);
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
    }
};


}