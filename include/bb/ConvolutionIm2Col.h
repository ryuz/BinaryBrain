// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <fstream>
#include <vector>
#include <random>

#include "bb/Manager.h"
#include "bb/Model.h"
#include "bb/FrameBuffer.h"


namespace bb {


template <typename FT = float, typename BT = float>
class ConvolutionIm2Col : public Model
{
protected:
    bool            m_host_only = false;
    
    indices_t       m_input_shape;
    indices_t       m_output_shape;
    index_t         m_input_frame_size;
    index_t         m_output_frame_size;
    index_t         m_input_h_size;
    index_t         m_input_w_size;
    index_t         m_input_c_size;
    index_t         m_filter_h_size;
    index_t         m_filter_w_size;
    index_t         m_x_stride = 1;
    index_t         m_y_stride = 1;
    index_t         m_y_offset = 0;
    index_t         m_x_offset = 0;
    index_t         m_output_h_size;
    index_t         m_output_w_size;
    std::string     m_padding;

    // メモリの確保/開放を繰り返さないように演算後も確保
    FrameBuffer     m_y_buf;
    FrameBuffer     m_dx_buf;

protected:
    ConvolutionIm2Col() {}

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
    ~ConvolutionIm2Col() {}

    struct create_t
    {
        index_t         filter_h_size = 3;
        index_t         filter_w_size = 3;
        index_t         stride_x      = 1;
        index_t         stride_y      = 1;
        std::string     padding       = "valid";
    };

    static std::shared_ptr<ConvolutionIm2Col> Create(create_t const & create)
    {
        auto self = std::shared_ptr<ConvolutionIm2Col>(new ConvolutionIm2Col);
        self->m_filter_h_size = create.filter_h_size;
        self->m_filter_w_size = create.filter_w_size;
        self->m_stride_x      = create.stride_x;
        self->m_stride_y      = create.stride_y;
        self->m_padding       = create.padding;
        return self;
    }

    static std::shared_ptr<ConvolutionIm2Col> Create(size_t filter_h_size, size_t filter_w_size, size_t y_stride=1, size_t x_stride=1, std::string padding="valid")
    {
        auto self = std::shared_ptr<ConvolutionIm2Col>(new ConvolutionIm2Col);
        self->m_filter_h_size = filter_h_size;
        self->m_filter_w_size = filter_w_size;
        self->m_y_stride      = y_stride;
        self->m_x_stride      = x_stride;
        self->m_padding       = padding;
        return self;
    }

    std::string GetClassName(void) const { return "ConvolutionIm2Col"; }


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
        BB_ASSERT(m_input_shape.size() == 3);

        m_input_w_size = m_input_shape[0];
        m_input_h_size = m_input_shape[1];
        m_input_c_size = m_input_shape[2];

        // 出力サイズ計算
        if ( m_padding == "valid" ) {
            m_output_h_size = ((m_input_h_size - m_filter_h_size + 1) + (m_x_stride - 1)) / m_x_stride;
            m_output_w_size = ((m_input_w_size - m_filter_w_size + 1) + (m_x_stride - 1)) / m_y_stride;
            m_y_offset = 0;
            m_x_offset = 0;
        }
        else if ( m_padding == "same" ) {
            m_output_h_size = (m_input_h_size + (m_x_stride - 1)) / m_y_stride;
            m_output_w_size = (m_input_w_size + (m_x_stride - 1)) / m_x_stride;
            m_y_offset = (m_filter_w_size - 1) / 2;
            m_x_offset = (m_filter_w_size - 1) / 2;
        }
        else {
            BB_ASSERT(0);
        }

        m_output_shape.resize(3);
        m_output_shape[0] = m_filter_w_size;
        m_output_shape[1] = m_filter_h_size;
        m_output_shape[2] = m_input_c_size;

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
    inline index_t GetInputNode(index_t c, index_t y, index_t x)
    {
        return (c * m_input_h_size + y)*m_input_w_size + x;
    }

    inline index_t GetOutputNode(index_t c, index_t y, index_t x)
    {
        return (c*m_filter_h_size + y)*m_filter_w_size + x;
    }

public:

    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        // SetInputShpaeされていなければ初回に設定
        if ( x_buf.GetShape() != m_input_shape ) {
            SetInputShape(x_buf.GetShape());
        }

        // 出力Frameサイズ計算
        m_input_frame_size = x_buf.GetFrameSize();
        m_output_frame_size = m_input_frame_size * m_output_h_size * m_output_w_size;

        // 出力形状設定
        m_y_buf.Resize(x_buf.GetType(), m_output_frame_size, m_output_shape);
        
#ifdef BB_WITH_CUDA
        if ( m_padding == "valid" && m_y_stride == 1 &&  m_x_stride == 1 && 
            DataType<FT>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && m_y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            // FP32 CUDA
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = m_y_buf.LockDeviceMemory();
            bbcu_fp32_Im2Col_Forward(
                (float const *)ptr_x.GetAddr(),
                (float       *)ptr_y.GetAddr(),
                (int          )m_input_frame_size,
                (int          )x_buf.GetFrameStride() / sizeof(float),
                (int          )m_input_w_size,
                (int          )m_input_h_size,
                (int          )m_input_c_size,
                (int          )m_y_buf.GetFrameStride() / sizeof(float),
                (int          )m_filter_w_size,
                (int          )m_filter_h_size);
            return m_y_buf;
        }
#endif

#ifdef BB_WITH_CUDA
        if (  m_padding == "valid" && m_y_stride == 1 &&  m_x_stride == 1 && 
            DataType<FT>::type == BB_TYPE_BIT && !m_host_only && x_buf.IsDeviceAvailable() && m_y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            // bit CUDA
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = m_y_buf.LockDeviceMemory();
            bbcu_bit_Im2Col_Forward(
                (int const *)ptr_x.GetAddr(),
                (int       *)ptr_y.GetAddr(),
                (int        )m_input_frame_size,
                (int        )x_buf.GetFrameStride() / sizeof(int),
                (int        )m_input_w_size,
                (int        )m_input_h_size,
                (int        )m_input_c_size,
                (int        )m_y_buf.GetFrameStride() / sizeof(int),
                (int        )m_filter_w_size,
                (int        )m_filter_h_size);
            return m_y_buf;
        }
#endif

        {
            // 汎用版
            index_t const output_frame_size = m_y_buf.GetFrameSize();
            index_t const output_size       = m_output_w_size * m_output_h_size;

            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = m_y_buf.Lock<FT>(true);

            for (index_t c = 0; c < m_input_c_size; ++c ) {
//              #pragma omp parallel for
                for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
//                  #pragma omp parallel for
                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                        for ( index_t output_frame = 0; output_frame < output_frame_size; ++output_frame ) {
                            index_t input_frame = output_frame / output_size;
                            index_t f           = output_frame % output_size;
                            index_t iy = (f / m_output_w_size) * m_y_stride - m_y_offset + fy;
                            index_t ix = (f % m_output_w_size) * m_x_stride - m_x_offset + fx;

                            FT in_sig = 0;
                            if ( iy >= 0 && iy < m_input_h_size && ix >= 0 && ix < m_input_w_size ) {
                                index_t input_node  = (c * m_input_h_size  + iy) * m_input_w_size  + ix;
                                in_sig = x_ptr.Get(input_frame, input_node);
                            }

                            index_t output_node = (c * m_filter_h_size + fy) * m_filter_w_size + fx;    
                            y_ptr.Set(output_frame, output_node, in_sig);
                        }
                    }
                }
            }

            return m_y_buf;
        }
    }


    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(dy_buf.GetType() == DataType<BT>::type);
        
        // 出力設定
        m_dx_buf.Resize(DataType<BT>::type, m_input_frame_size, m_input_shape);

#ifdef BB_WITH_CUDA
        if ( m_padding == "valid" && m_y_stride == 1 &&  m_x_stride == 1 && 
            DataType<BT>::type == BB_TYPE_FP32 && !m_host_only && dy_buf.IsDeviceAvailable() && m_dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable())
        {
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = m_dx_buf.LockDeviceMemory();
            bbcu_fp32_Im2Col_Backward(
                (float const *)ptr_dy.GetAddr(),
                (float       *)ptr_dx.GetAddr(),
                (int          )m_input_frame_size,
                (int          )(m_dx_buf.GetFrameStride() / sizeof(float)),
                (int          )m_input_w_size,
                (int          )m_input_h_size,
                (int          )m_input_c_size,
                (int          )(dy_buf.GetFrameStride() / sizeof(float)),
                (int          )m_filter_w_size,
                (int          )m_filter_h_size);
            return m_dx_buf;
        }
#endif

#if 0
        if ( 0 ) {
            // 汎用版
            m_dx_buf.FillZero();

            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = m_dx_buf.Lock<BT>();

            for (index_t c = 0; c < m_input_c_size; ++c) {
                #pragma omp parallel for
                for (index_t y = 0; y < m_input_h_size; ++y ) {
                    #pragma omp parallel for
                    for (index_t x = 0; x < m_input_w_size; ++x ) {
                        index_t input_node = (c * m_input_h_size + y) * m_input_w_size + x;
                        for ( index_t input_frame = 0; input_frame < m_input_frame_size; ++input_frame ) {
                            BT dx = dx_ptr.Get(input_frame, input_node);
                            float dy = 0;
                            for (int fy = 0; fy < m_filter_h_size; ++fy) {
                                index_t iy = y - fy;
                                if ( iy >= 0 && iy < (m_input_h_size - m_filter_h_size + 1)) {
                                    for (int fx = 0; fx < m_filter_w_size; ++fx) {
                                        index_t ix = x - fx;
                                        if (ix >= 0 && ix < (m_input_w_size - m_filter_w_size + 1)) {
                                            index_t output_frame = (input_frame * m_output_h_size + iy) * m_output_w_size + ix;
                                            index_t output_node  = (c * m_filter_h_size + fy) * m_filter_w_size + fx;
                                            dy += dy_ptr.Get(output_frame, output_node);
                                        }
                                    }
                                }
                            }
                            dx_ptr.Set(input_frame, input_node, dx + dy);
                        }
                    }
                }
            }

            return m_dx_buf;
        }
#endif

        {
            // stride版
            m_dx_buf.FillZero();

            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = m_dx_buf.Lock<BT>();

            for (index_t c = 0; c < m_input_c_size; ++c) {
//                #pragma omp parallel for
                for (index_t y = 0; y < m_input_h_size; ++y ) {
//                    #pragma omp parallel for
                    for (index_t x = 0; x < m_input_w_size; ++x ) {
                        index_t input_node = (c * m_input_h_size + y) * m_input_w_size + x;
                        index_t x_align = x % m_x_stride;
                        index_t y_align = y % m_y_stride;
                        for ( index_t input_frame = 0; input_frame < m_input_frame_size; ++input_frame ) {
                            BT dx = dx_ptr.Get(input_frame, input_node);
                            float dy = 0;
                            for (index_t fy = y_align; fy < m_filter_h_size; fy += m_y_stride ) {
                                index_t iy = y - fy;
                                if ( iy >= 0 && iy < (m_input_h_size - m_filter_h_size + 1)) {
                                    for (index_t fx = x_align; fx < m_filter_w_size; fx += m_x_stride) {
                                        index_t ix = x - fx;
                                        if (ix >= 0 && ix < (m_input_w_size - m_filter_w_size + 1)) {
                                            index_t output_frame = (input_frame * m_output_h_size + (iy/m_y_stride)) * m_output_w_size + (ix/m_x_stride);
                                            index_t output_node  = (c * m_filter_h_size + fy) * m_filter_w_size + fx;
                                            dy += dy_ptr.Get(output_frame, output_node);
                                        }
                                    }
                                }
                            }
                            dx_ptr.Set(input_frame, input_node, dx + dy);
                        }
                    }
                }
            }

            return m_dx_buf;
        }
    }
};


}