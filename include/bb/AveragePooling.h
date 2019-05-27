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

// AveragePoolingクラス
template <typename FT = float, typename BT = float>
class AveragePooling : public Filter2d<FT, BT>
{
protected:
    bool                m_host_only;

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

    FrameBuffer         m_x;
    FrameBuffer         m_y;
    FrameBuffer         m_dx;

protected:
    AveragePooling() {}

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
    ~AveragePooling() {}
    
    static std::shared_ptr<MaxPooling> Create(index_t filter_h_size, index_t filter_w_size)
    {
        auto self = std::shared_ptr<AveragePooling>(new AveragePooling);

        self->m_filter_h_size = filter_h_size;
        self->m_filter_w_size = filter_w_size;

        return self;
    }
    

    std::string GetClassName(void) const { return "AveragePooling"; }

    index_t GetFilterHeight(void) { return m_filter_h_size; }
    index_t GetFilterWidth(void)  { return m_filter_w_size; }

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
        m_output_w_size = (m_input_w_size + m_filter_w_size - 1) / m_filter_w_size;
        m_output_h_size = (m_input_h_size + m_filter_h_size - 1) / m_filter_h_size;
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
    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        BB_ASSERT(x.GetType() == DataType<FT>::type);

        // backwardの為に保存
        m_x = x;

        // SetInputShpaeされていなければ初回に設定
        if (m_x.GetShape() != m_input_shape) {
            SetInputShape(m_x.GetShape());
        }

        // 出力を設定
        m_y.Resize(DataType<FT>::type, m_x.GetFrameSize(), m_output_shape);
        

#if 0 // #ifdef BB_WITH_CUDA
        // CUDA版
        if ( DataType<FT>::type == BB_TYPE_FP32 && !m_host_only && m_x.IsDeviceAvailable() && m_y.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto ptr_x = x.LockDeviceMemoryConst();
            auto ptr_y = m_y.LockDeviceMemory(true);
            bbcu_fp32_MaxPooling_Forward
                (
                    (float const *)ptr_x.GetAddr(),
                    (float*       )ptr_y.GetAddr(),
                    (int          )m_filter_h_size,
                    (int          )m_filter_w_size,
                    (int          )m_input_w_size,
                    (int          )m_input_h_size,
                    (int          )m_output_w_size,
                    (int          )m_output_h_size,
                    (int          )m_output_c_size,
                    (int          )m_y.GetFrameSize(),
                    (int          )(m_y.GetFrameStride() / sizeof(float))
                );

            return m_y;
        }
#endif

#if 0
        if ( DataType<FT>::type == BB_TYPE_BIT ) {
            // バイナリ用実装
            auto x_ptr = m_x.LockConst<FT>();
            auto y_ptr = m_y.Lock<FT>(true);

            index_t  m256_frame_size = (int)m_y.GetFrameStride() / 32;

            #pragma omp parallel for
            for (index_t c = 0; c < m_input_c_size; ++c) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        __m256i *y_addr = (__m256i *)y_ptr.GetAddr(GetOutputNode(c, y, x));

                        for (index_t frame = 0; frame < m256_frame_size; ++frame) {
                            __m256i max_val = _mm256_set1_epi8(0);
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y*m_filter_h_size + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x*m_filter_w_size + fx;
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

            return m_y;
        }
#endif

        // float用実装
        if ( DataType<FT>::type == BB_TYPE_FP32 ) {
            auto x_ptr = m_x.LockConst<FT>();
            auto y_ptr = m_y.Lock<FT>(true);

            index_t  m256_frame_size = (int)m_y.GetFrameStride() / sizeof(float);

            __m256   frac = _mm256_set1_ps(1.0f / (m_filter_h_size * m_filter_w_size));

            #pragma omp parallel for
            for (index_t c = 0; c < m_input_c_size; ++c) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        float *y_addr = (float *)y_ptr.GetAddr(GetOutputNode(c, y, x));

                        for (index_t frame = 0; frame < m256_frame_size; frame += 8) {
                            __m256  sum_val = _mm256_set1_ps(0.0f);  // 前段に活性化入れるから0がminだよね？
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y*m_filter_h_size + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x*m_filter_w_size + fx;
                                        if ( ix < m_input_w_size ) {
                                            float const *x_addr = (float const *)x_ptr.GetAddr(GetInputNode(c, iy, ix));
                                            __m256 in_sig = _mm256_load_ps(&x_addr[frame]);
                                            sum_val = _mm256_add_ps(sum_val, in_sig);
                                        }
                                    }
                                }
                            }

                            __m256 ave_val = _mm256_mul_ps(sum_val, frac);

                            _mm256_store_ps(&y_addr[frame], ave_val);
                        }
                    }
                }
            }

            return m_y;
        }

        // 汎用版実装
        {
            auto x_ptr = m_x.LockConst<FT>();
            auto y_ptr = m_y.Lock<FT>(true);

            auto frame_size = m_x.GetFrameSize();

            #pragma omp parallel for
            for (index_t c = 0; c < m_input_c_size; ++c) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        for (index_t frame = 0; frame < frame_size; ++frame) {
                            FT sum_val = 0;
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y*m_filter_h_size + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x*m_filter_w_size + fx;
                                        if ( ix < m_input_w_size ) {
                                            FT in_sig = x_ptr.Get(frame, {ix, iy, c});
                                            sum_val += in_sig;
                                        }
                                    }
                                }
                            }
                            y_ptr.Set(frame, {x, y, c}, sum_val / (m_filter_h_size * m_filter_w_size));
                        }
                    }
                }
            }

            return m_y;
        }
    }
    
    FrameBuffer Backward(FrameBuffer dy)
    {
        BB_ASSERT(dy.GetType() == DataType<BT>::type);

        m_dx.Resize(DataType<BT>::type, dy.GetFrameSize(), m_input_shape);

#if 0 // #ifdef BB_WITH_CUDA
        if ( DataType<BT>::type == BB_TYPE_FP32 && DataType<FT>::type == BB_TYPE_FP32 && !m_host_only 
                && m_x.IsDeviceAvailable() && m_y.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x  = m_x.LockDeviceMemoryConst();
            auto ptr_y  = m_y.LockDeviceMemoryConst();
            auto ptr_dy = dy.LockDeviceMemoryConst();
            auto ptr_dx = m_dx.LockDeviceMemory(true);
            bbcu_fp32_MaxPooling_Backward
                (
                    (float const *)ptr_x.GetAddr(),
                    (float const *)ptr_y.GetAddr(),
                    (float const *)ptr_dy.GetAddr(),
                    (float*       )ptr_dx.GetAddr(),
                    (int          )m_filter_h_size,
                    (int          )m_filter_w_size,
                    (int          )m_input_w_size,
                    (int          )m_input_h_size,
                    (int          )m_output_w_size,
                    (int          )m_output_h_size,
                    (int          )m_output_c_size,
                    (int          )m_y.GetFrameSize(),
                    (int          )(m_y.GetFrameStride() / sizeof(float))
                );

            return m_dx;
        }
#endif

        if ( DataType<BT>::type == BB_TYPE_FP32 && DataType<FT>::type == BB_TYPE_FP32 ) {
            // float用実装
            index_t  m256_frame_size = m_dx.GetFrameStride() / sizeof(float);

            m_dx.FillZero();

            auto x_ptr  = m_x.LockConst<FT>();
            auto y_ptr  = m_y.LockConst<FT>();
            auto dy_ptr = dy.LockConst<BT>();
            auto dx_ptr = m_dx.Lock<BT>(true);

            __m256   frac = _mm256_set1_ps(1.0f / (m_filter_h_size * m_filter_w_size));

    #pragma omp parallel for
            for (index_t n = 0; n < m_input_c_size; ++n) {
                for (index_t y = 0; y < m_output_h_size; ++y) {
                    for (index_t x = 0; x < m_output_w_size; ++x) {
                        float const * y_addr  = (float const *)y_ptr.GetAddr(GetOutputNode(n, y, x));
                        float const * dy_addr = (float const *)dy_ptr.GetAddr(GetOutputNode(n, y, x));

                        for (index_t frame = 0; frame < m256_frame_size; frame += 8) {
                            __m256 out_sig  = _mm256_load_ps(&y_addr[frame]);
                            __m256 out_grad = _mm256_load_ps(&dy_addr[frame]);
                            __m256 in_grad  = _mm256_mul_ps(frac, out_grad);
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y*m_filter_h_size + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x*m_filter_w_size + fx;
                                        if ( ix < m_input_w_size ) {
                                            float       *dx_addr = (float *)dx_ptr.GetAddr(GetInputNode(n, iy, ix));
                                            __m256 dx = _mm256_load_ps(&dx_addr[frame]);
                                            __m256 dx = _mm256_add_ps(dx, in_grad);
                                            _mm256_store_ps(&dx_addr[frame], dx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return m_dx;
        }

        // 汎用版実装
        {
            m_dx.FillZero();

            auto x_ptr  = m_x.LockConst<FT>();
            auto y_ptr  = m_y.LockConst<FT>();
            auto dy_ptr = dy.LockConst<BT>();
            auto dx_ptr = m_dx.Lock<BT>(true);

            auto frame_size = m_x.GetFrameSize();

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
                                            dx_ptr.Add(frame, c, iy, ix, grad / (m_filter_h_size * m_filter_w_size));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return m_dx;
        }
    }
};


}