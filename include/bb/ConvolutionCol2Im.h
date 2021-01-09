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
class ConvolutionCol2Im : public Model
{
    using _super = Model;

public:
    static inline std::string ModelName(void) { return "ConvolutionCol2Im"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<FT>::Name() + "_" + DataType<BT>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    bool            m_host_only = false;

    indices_t       m_input_shape;

    index_t         m_c_size = 1;
    index_t         m_h_size = 1;
    index_t         m_w_size = 1;

public:
    struct create_t
    {
        index_t h_size = 1;
        index_t w_size = 1;
    };
    
protected:
    ConvolutionCol2Im(create_t const & create)
    {
        m_h_size = create.h_size;
        m_w_size = create.w_size;
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
    ~ConvolutionCol2Im() {}

    static std::shared_ptr<ConvolutionCol2Im> Create(create_t const &create)
    {
        return std::shared_ptr<ConvolutionCol2Im>(new ConvolutionCol2Im(create));
    }

    static std::shared_ptr<ConvolutionCol2Im> Create(index_t h_size=1, index_t w_size=1)
    {
        create_t create;
        create.h_size = h_size;
        create.w_size = w_size;
        return Create(create);
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<ConvolutionCol2Im> CreatePy(index_t h_size, index_t w_size)
    {
        create_t create;
        create.h_size = h_size;
        create.w_size = w_size;
        return Create(create);
    }
#endif

    void SetOutputSize(index_t h_size, index_t w_size)
    {
        m_h_size = h_size;
        m_w_size = w_size;
    }

    index_t GetChannel(void) const { return m_c_size; }
    index_t GetHeight(void)  const { return m_h_size; }
    index_t GetWidth(void)   const { return m_w_size; }
    

public:

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

 //      BB_ASSERT(shape.size() == 1);
        m_input_shape  = shape;
        m_c_size = CalcShapeSize(shape);
        return indices_t({m_c_size, m_h_size, m_w_size});
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
        return indices_t({m_c_size, m_h_size, m_w_size});
    }


    FrameBuffer Forward(FrameBuffer x_buf, bool train=true) override
    {
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        index_t input_frame_size  = x_buf.GetFrameSize();
        BB_ASSERT(input_frame_size % (m_h_size * m_w_size) == 0);
        index_t output_frame_size = input_frame_size / (m_h_size * m_w_size);

        FrameBuffer y_buf(output_frame_size, indices_t({m_c_size, m_h_size, m_w_size}), DataType<FT>::type);


#ifdef BB_WITH_CUDA
        if ( !m_host_only && DataType<FT>::type == BB_TYPE_FP32 && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // FP32 CUDA
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);

            bbcu_fp32_Col2Im_Forward
                (
                    (float const *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int          )m_w_size,
                    (int          )m_h_size,
                    (int          )m_c_size,
                    (int          )(x_buf.GetFrameStride() / sizeof(float)),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float))
                );

            return y_buf;
        }
#endif

#ifdef BB_WITH_CUDA
        if ( !m_host_only && DataType<FT>::type == BB_TYPE_BIT && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // Bit CUDA
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);

            bbcu_bit_Col2Im_Forward
                (
                    (int const *)x_ptr.GetAddr(),
                    (int       *)y_ptr.GetAddr(),
                    (int        )m_w_size,
                    (int        )m_h_size,
                    (int        )m_c_size,
                    (int        )(x_buf.GetFrameStride() / sizeof(int)),
                    (int        )y_buf.GetFrameSize(),
                    (int        )(y_buf.GetFrameStride() / sizeof(int))
                );

            return y_buf;
        }
#endif

        {
            // 汎用版
            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = y_buf.Lock<FT>(true);

            auto hw_size = m_h_size * m_w_size;

            for (index_t c = 0; c < m_c_size; ++c) {
                #pragma omp parallel for
                for (index_t xy = 0; xy < hw_size; ++xy) {
                    for ( index_t output_frame = 0; output_frame < output_frame_size; ++output_frame ) {
                        index_t output_node = c * hw_size + xy;
                        index_t input_frame = output_frame * hw_size + xy;
                        index_t input_node  = c;
                        y_ptr.Set(output_frame, output_node, x_ptr.Get(input_frame, input_node));
                    }
                }
            }
            return y_buf;
        }

        {
            // 汎用版(旧)
            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = y_buf.Lock<FT>(true);
            index_t input_frame = 0;
            for (index_t output_frame = 0; output_frame < output_frame_size; ++output_frame) {
                for (index_t y = 0; y < m_h_size; ++y) {
                    for (index_t x = 0; x < m_w_size; ++x) {
                        #pragma omp parallel for
                        for (index_t c = 0; c < m_c_size; ++c) {
                            index_t input_node = c;
                            index_t output_node = (c*m_h_size + y)*m_w_size + x;
                            y_ptr.Set(output_frame, output_node, x_ptr.Get(input_frame, input_node));
                        }
                        ++input_frame;
                    }
                }
            }
            return y_buf;
        }
    }
    
    FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        BB_ASSERT(dy_buf.GetType() == DataType<BT>::type);

        index_t output_frame_size = dy_buf.GetFrameSize();
        index_t input_frame_size  = output_frame_size *(m_h_size * m_w_size);

        FrameBuffer dx_buf(input_frame_size, {m_c_size}, DataType<BT>::type);

#ifdef BB_WITH_CUDA
        if ( !m_host_only && DataType<BT>::type == BB_TYPE_FP32 && dy_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() )
        {
            auto dy_ptr = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr = dx_buf.LockDeviceMemory(true);

            bbcu_fp32_Col2Im_Backward
                (
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (int          )m_w_size,
                    (int          )m_h_size,
                    (int          )m_c_size,
                    (int          )(dx_buf.GetFrameStride() / sizeof(float)),
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
            
            auto hw_size = m_h_size * m_w_size;

            for (index_t c = 0; c < m_c_size; ++c) {
                #pragma omp parallel for
                for (index_t xy = 0; xy < hw_size; ++xy) {
                    for (index_t output_frame = 0; output_frame < output_frame_size; ++output_frame) {
                        index_t output_node = c * hw_size + xy;
                        index_t input_frame = output_frame * hw_size + xy;
                        index_t input_node  = c;

                        dx_ptr.Set(input_frame, input_node, dy_ptr.Get(output_frame, output_node));
                    }
                }
            }

            return dx_buf;
        }

        {
            // 汎用版
            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = dx_buf.Lock<BT>(true);

            index_t input_frame = 0;
            for (index_t output_frame = 0; output_frame < output_frame_size; ++output_frame) {
                for (index_t y = 0; y < m_h_size; ++y) {
                    for (index_t x = 0; x < m_w_size; ++x) {
                        #pragma omp parallel for
                        for (index_t c = 0; c < m_c_size; ++c) {
                            index_t output_node = (c*m_h_size + y)*m_w_size + x;
                            index_t input_node = c;
                            dx_ptr.Set(input_frame, input_node, dy_ptr.Get(output_frame, output_node));
                        }
                        ++input_frame;
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
        bb::SaveValue(os, m_h_size);
        bb::SaveValue(os, m_w_size);
        bb::SaveValue(os, m_input_shape);
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
        bb::LoadValue(is, m_h_size);
        bb::LoadValue(is, m_w_size);
        bb::LoadValue(is, m_input_shape);
        
        // 再構築
        m_c_size = CalcShapeSize(m_input_shape);
    }
};


}