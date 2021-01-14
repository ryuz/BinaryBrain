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
class MaxPooling : public Filter2d
{
    using _super = Filter2d;

public:
    static inline std::string ModelName(void) { return "MaxPooling"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<FT>::Name() + "_" + DataType<BT>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }


protected:
    bool                m_host_only = false;

    index_t             m_filter_h_size;
    index_t             m_filter_w_size;

    index_t             m_input_c_size;
    index_t             m_input_h_size;
    index_t             m_input_w_size;
    index_t             m_output_c_size;
    index_t             m_output_h_size;
    index_t             m_output_w_size;

    indices_t           m_input_shape;
    indices_t           m_output_shape;

    FrameBuffer         m_x_buf;
    FrameBuffer         m_y_buf;

public:
    struct create_t
    {
        index_t filter_h_size = 1;
        index_t filter_w_size = 1;
    };


protected:
    // コンストラクタ
    MaxPooling() {}

    MaxPooling(create_t const &create)
    {
        m_filter_h_size = create.filter_h_size;
        m_filter_w_size = create.filter_w_size;
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
    ~MaxPooling() {}

    static std::shared_ptr<MaxPooling> Create(create_t const &create)
    {
         return std::shared_ptr<MaxPooling>(new MaxPooling(create));
    }
    
    static std::shared_ptr<MaxPooling> Create(index_t filter_h_size=1, index_t filter_w_size=1)
    {
        create_t create;
        create.filter_h_size = filter_h_size;
        create.filter_w_size = filter_w_size;
        return Create(create);
    }

#ifdef BB_PYBIND11
    // 全パラメータを引数としたオーバーロード無しの生成関数(主にpython用)
    static std::shared_ptr<MaxPooling> CreatePy(index_t filter_h_size, index_t filter_w_size)
    {
        create_t create;
        create.filter_h_size = filter_h_size;
        create.filter_w_size = filter_w_size;
        return Create(create);
    }
#endif


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
    indices_t SetInputShape(indices_t shape) override
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
    FrameBuffer Forward(FrameBuffer x_buf, bool train = true) override
    {
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_input_shape) {
            SetInputShape(x_buf.GetShape());
        }

        // 出力を設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), m_output_shape, DataType<FT>::type);
        
        // backwardの為に保存
        if ( train ) {
            m_x_buf = x_buf;
            m_y_buf = y_buf;
        }

#ifdef BB_WITH_CUDA
        // FP32 CUDA版
        if ( DataType<FT>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory(true);
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
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float))
                );

            return y_buf;
        }
        
        // Bit CUDA版
        if ( DataType<FT>::type == BB_TYPE_BIT && !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory(true);
            bbcu_bit_MaxPooling_Forward
                (
                    (int const  *)ptr_x.GetAddr(),
                    (int        *)ptr_y.GetAddr(),
                    (int         )m_filter_h_size,
                    (int         )m_filter_w_size,
                    (int         )m_input_w_size,
                    (int         )m_input_h_size,
                    (int         )m_output_w_size,
                    (int         )m_output_h_size,
                    (int         )m_output_c_size,
                    (int         )y_buf.GetFrameSize(),
                    (int         )(y_buf.GetFrameStride() / sizeof(int))
                );

            return y_buf;
        }
#endif
     
        if ( DataType<FT>::type == BB_TYPE_BIT ) {
            // バイナリ用実装
            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = y_buf.Lock<FT>(true);

            index_t  m256_frame_size = (int)y_buf.GetFrameStride() / 32;

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

            return y_buf;
        }

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
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y*m_filter_h_size + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x*m_filter_w_size + fx;
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
                            FT max_val = 0;
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y*m_filter_h_size + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x*m_filter_w_size + fx;
                                        if ( ix < m_input_w_size ) {
                                            FT in_sig = x_ptr.Get(frame, {c, iy, ix});
                                            max_val = (max_val > in_sig) ? max_val : in_sig;
                                        }
                                    }
                                }
                            }
                            y_ptr.Set(frame, {c, y, x}, max_val);
                        }
                    }
                }
            }

            return y_buf;
        }
    }
    
    FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        BB_ASSERT(dy_buf.GetType() == DataType<BT>::type);

        FrameBuffer dx_buf(dy_buf.GetFrameSize(), m_input_shape, DataType<BT>::type);

        FrameBuffer x_buf = m_x_buf;
        FrameBuffer y_buf = m_y_buf;
        m_x_buf = FrameBuffer();
        m_y_buf = FrameBuffer();

        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);
        BB_ASSERT(y_buf.GetType() == DataType<FT>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<BT>::type == BB_TYPE_FP32 && DataType<FT>::type == BB_TYPE_FP32 && !m_host_only 
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x  = x_buf.LockDeviceMemoryConst();
            auto ptr_y  = y_buf.LockDeviceMemoryConst();
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = dx_buf.LockDeviceMemory(true);
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
                    (int          )dy_buf.GetFrameSize(),
                    (int          )(dy_buf.GetFrameStride() / sizeof(float))
                );

            return dx_buf;
        }
#endif

#ifdef BB_WITH_CUDA
        if ( DataType<FT>::type == BB_TYPE_BIT && DataType<BT>::type == BB_TYPE_FP32 && !m_host_only 
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x  = x_buf.LockDeviceMemoryConst();
            auto ptr_y  = y_buf.LockDeviceMemoryConst();
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = dx_buf.LockDeviceMemory(true);
            bbcu_bit_fp32_MaxPooling_Backward
                (
                    (int   const *)ptr_x.GetAddr(),
                    (int   const *)ptr_y.GetAddr(),
                    (float const *)ptr_dy.GetAddr(),
                    (float*       )ptr_dx.GetAddr(),
                    (int          )m_filter_h_size,
                    (int          )m_filter_w_size,
                    (int          )m_input_w_size,
                    (int          )m_input_h_size,
                    (int          )m_output_w_size,
                    (int          )m_output_h_size,
                    (int          )m_output_c_size,
                    (int          )dy_buf.GetFrameSize(),
                    (int          )(x_buf.GetFrameStride() / sizeof(int)),
                    (int          )(dy_buf.GetFrameStride() / sizeof(float))
                );

            return dx_buf;
        }
#endif

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
                            for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y*m_filter_h_size + fy;
                                if ( iy < m_input_h_size ) {
                                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x*m_filter_w_size + fx;
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

        // 汎用版実装
        {
            auto x_ptr  = x_buf.LockConst<FT>();
            auto y_ptr  = y_buf.LockConst<FT>();
            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = dx_buf.Lock<BT>(true);

            auto frame_size = x_buf.GetFrameSize();

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
        bb::SaveValue(os, m_filter_h_size);
        bb::SaveValue(os, m_filter_w_size);
        bb::SaveValue(os, m_input_c_size);
        bb::SaveValue(os, m_input_h_size);
        bb::SaveValue(os, m_input_w_size);
        bb::SaveValue(os, m_output_c_size);
        bb::SaveValue(os, m_output_h_size);
        bb::SaveValue(os, m_output_w_size);
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
        bb::LoadValue(is, m_filter_h_size);
        bb::LoadValue(is, m_filter_w_size);
        bb::LoadValue(is, m_input_c_size);
        bb::LoadValue(is, m_input_h_size);
        bb::LoadValue(is, m_input_w_size);
        bb::LoadValue(is, m_output_c_size);
        bb::LoadValue(is, m_output_h_size);
        bb::LoadValue(is, m_output_w_size);
        
        // 再構築
        m_input_shape  = indices_t({m_input_c_size, m_input_h_size, m_input_w_size});
        m_output_shape = indices_t({m_output_c_size, m_output_h_size, m_output_w_size});
    }
};


}