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
#include "bb/Filter2d.h"


namespace bb {


template <typename FT = float, typename BT = float>
class ConvolutionIm2Col : public Model
{
    using _super = Model;

public:
    static inline std::string ModelName(void) { return "ConvolutionIm2Col"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<FT>::Name() + "_" + DataType<BT>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    bool            m_host_only = false;
    
    indices_t       m_input_shape;
    indices_t       m_output_shape;
    index_t         m_input_frame_size = 1;
    index_t         m_output_frame_size = 1;
    index_t         m_input_c_size = 1;
    index_t         m_input_h_size = 1;
    index_t         m_input_w_size = 1;
    index_t         m_filter_h_size = 1;
    index_t         m_filter_w_size = 1;
    index_t         m_y_stride = 1;
    index_t         m_x_stride = 1;
    index_t         m_y_offset = 0;
    index_t         m_x_offset = 0;
    index_t         m_output_h_size = 1;
    index_t         m_output_w_size = 1;
    std::string     m_padding      = "valid";
    int             m_border_mode  = BB_BORDER_REFLECT_101;
    FT              m_border_value = (FT)0;

public:
    struct create_t
    {
        index_t         filter_h_size = 1;
        index_t         filter_w_size = 1;
        index_t         x_stride      = 1;
        index_t         y_stride      = 1;
        std::string     padding       = "valid";
        std::string     border_mode   = "reflect_101";
        FT              border_value  = (FT)0;
    };

protected:
    ConvolutionIm2Col(create_t const & create)
    {
        m_filter_h_size = create.filter_h_size;
        m_filter_w_size = create.filter_w_size;
        m_x_stride      = create.x_stride;
        m_y_stride      = create.y_stride;
        m_padding       = create.padding;
        m_border_mode   = BorderConv(create.border_mode);
        m_border_value  = create.border_value;
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
    ~ConvolutionIm2Col() {}

    static std::shared_ptr<ConvolutionIm2Col> Create(create_t const &create)
    {
        return std::shared_ptr<ConvolutionIm2Col>(new ConvolutionIm2Col(create));
    }

    static std::shared_ptr<ConvolutionIm2Col> Create(index_t filter_h_size, index_t filter_w_size, index_t y_stride=1, index_t x_stride=1,
                                                std::string padding="valid", std::string  border_mode = "reflect_101")
    {
        create_t create;
        create.filter_h_size = filter_h_size;
        create.filter_w_size = filter_w_size;
        create.y_stride      = y_stride;
        create.x_stride      = x_stride;
        create.padding       = padding;
        create.border_mode   = border_mode;
        return Create(create);
    }

    static std::shared_ptr<ConvolutionIm2Col> Create(void)
    {
        return Create(create_t());
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<ConvolutionIm2Col> CreatePy(index_t filter_h_size, index_t filter_w_size, index_t y_stride=1, index_t x_stride=1,
                                                std::string padding="valid", std::string  border_mode = "reflect_101")
    {
        create_t create;
        create.filter_h_size = filter_h_size;
        create.filter_w_size = filter_w_size;
        create.y_stride      = y_stride;
        create.x_stride      = x_stride;
        create.padding       = padding;
        create.border_mode   = border_mode;
        return Create(create);
    }
#endif


    index_t     GetFilterSizeH(void) const { return m_filter_h_size; }
    index_t     GetFilterSizeW(void) const { return m_filter_w_size; }
    index_t     GetStrideX(void) const     { return m_x_stride; }
    index_t     GetStrideY(void) const     { return m_y_stride; }

    std::string GetPadding(void) const     { return m_padding; }
//  int         GetBorderMode(void) const  { return m_border_mode; }*/
    std::string GetBorderMode(void) const
    {
        switch ( m_border_mode ) {
        case BB_BORDER_CONSTANT:     return "constant";
        case BB_BORDER_REFLECT:      return "reflect";
        case BB_BORDER_REFLECT_101:  return "reflect_101";
        case BB_BORDER_REPLICATE:    return "replicate";
        case BB_BORDER_WRAP:         return "wrap";
        }
        BB_DEBUG_ASSERT(0);
        return "";
    }
    FT          GetBorderValue(void) const { return m_border_value; }



    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

        // 形状設定
        m_input_shape = shape;
        BB_ASSERT(m_input_shape.size() == 3);

        m_input_c_size = m_input_shape[0];
        m_input_h_size = m_input_shape[1];
        m_input_w_size = m_input_shape[2];

        // 出力サイズ計算
        if ( m_padding == "valid" ) {
            m_output_h_size = ((m_input_h_size - m_filter_h_size + 1) + (m_y_stride - 1)) / m_y_stride;
            m_output_w_size = ((m_input_w_size - m_filter_w_size + 1) + (m_x_stride - 1)) / m_x_stride;
            m_y_offset = 0;
            m_x_offset = 0;
        }
        else if ( m_padding == "same" ) {
            m_output_h_size = (m_input_h_size + (m_y_stride - 1)) / m_y_stride;
            m_output_w_size = (m_input_w_size + (m_x_stride - 1)) / m_x_stride;
            m_y_offset = (m_filter_h_size - 1) / 2;
            m_x_offset = (m_filter_w_size - 1) / 2;
        }
        else {
            BB_ASSERT(0);
        }

        m_output_shape.resize(3);
        m_output_shape[0] = m_input_c_size;
        m_output_shape[1] = m_filter_h_size;
        m_output_shape[2] = m_filter_w_size;

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


    inline int BorderConv(std::string const& mode)
    {
        if ( mode == "constant" )    { return BB_BORDER_CONSTANT; }
        if ( mode == "reflect" )     { return BB_BORDER_REFLECT  ; }
        if ( mode == "reflect_101" ) { return BB_BORDER_REFLECT_101; }
        if ( mode == "replicate" )   { return BB_BORDER_REPLICATE; }
        if ( mode == "wrap" )        { return BB_BORDER_WRAP; }
        BB_DEBUG_ASSERT(0);
        return BB_BORDER_CONSTANT;
    }

    inline bool Border(int border_mode, index_t &x, index_t &y, index_t w, index_t h)
    {
        switch ( border_mode ) {
        case BB_BORDER_REFLECT:
            if ( x < 0  ) { x = -x - 1; }
            if ( y < 0  ) { y = -y - 1; }
            if ( x >= w ) { x = (w - 1) - (x - w); }
            if ( y >= h ) { y = (h - 1) - (y - h); }
            return true;
    
        case BB_BORDER_REFLECT_101:
            if ( x < 0  ) { x = -x; }
            if ( y < 0  ) { y = -y; }
            if ( x >= w ) { x = (w - 2) - (x - w); }
            if ( y >= h ) { y = (h - 2) - (y - h); }
            return true;

        case BB_BORDER_REPLICATE:
            if ( x < 0  ) { x = 0; }
            if ( y < 0  ) { y = 0; }
            if ( x >= w ) { x = w - 1; }
            if ( y >= h ) { y = h - 1; }
            return true;

        case BB_BORDER_WRAP:
            if ( x < 0  ) { x += w; }
            if ( y < 0  ) { y += h; }
            if ( x >= w ) { x -= w; }
            if ( y >= h ) { y -= h; }
            return true;

        default:
            return false;
        }
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
        FrameBuffer y_buf(m_output_frame_size, m_output_shape, x_buf.GetType());
        

#ifdef BB_WITH_CUDA
        if ( DataType<FT>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            // FP32 CUDA
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory();
            bbcu_fp32_Im2Col_Forward(
                    (float const *)ptr_x.GetAddr(),
                    (float       *)ptr_y.GetAddr(),
                    (int          )m_x_stride,
                    (int          )m_y_stride,
                    (int          )m_x_offset,
                    (int          )m_y_offset,
                    (int          )m_input_frame_size,
                    (int          )x_buf.GetFrameStride() / sizeof(float),
                    (int          )m_input_w_size,
                    (int          )m_input_h_size,
                    (int          )m_input_c_size,
                    (int          )m_output_w_size,
                    (int          )m_output_h_size,
                    (int          )y_buf.GetFrameStride() / sizeof(float),
                    (int          )m_filter_w_size,
                    (int          )m_filter_h_size,
                    (int          )m_border_mode,
                    (float        )m_border_value
                );
            return y_buf;
        }
#endif

#ifdef BB_WITH_CUDA
        if ( m_filter_w_size * m_filter_h_size <= 1024 / 32
            && DataType<FT>::type == BB_TYPE_BIT && !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            // bit CUDA
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory();
            bbcu_bit_Im2Col_Forward(
                    (int const *)ptr_x.GetAddr(),
                    (int       *)ptr_y.GetAddr(),
                    (int        )m_x_stride,
                    (int        )m_y_stride,
                    (int        )m_x_offset,
                    (int        )m_y_offset,
                    (int        )m_input_frame_size,
                    (int        )x_buf.GetFrameStride() / sizeof(int),
                    (int        )m_input_w_size,
                    (int        )m_input_h_size,
                    (int        )m_input_c_size,
                    (int        )m_output_w_size,
                    (int        )m_output_h_size,
                    (int        )y_buf.GetFrameStride() / sizeof(int),
                    (int        )m_filter_w_size,
                    (int        )m_filter_h_size,
                    (int        )m_border_mode
                );
            return y_buf;
        }
#endif

        {
            // 汎用版
            index_t const output_frame_size = y_buf.GetFrameSize();
            index_t const output_size       = m_output_w_size * m_output_h_size;

            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = y_buf.Lock<FT>(true);

            for (index_t c = 0; c < m_input_c_size; ++c ) {
                #pragma omp parallel for
                for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
                    #pragma omp parallel for
                    for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
                        for ( index_t output_frame = 0; output_frame < output_frame_size; ++output_frame ) {
                            index_t input_frame = output_frame / output_size;
                            index_t f           = output_frame % output_size;
                            index_t iy = (f / m_output_w_size) * m_y_stride - m_y_offset + fy;
                            index_t ix = (f % m_output_w_size) * m_x_stride - m_x_offset + fx;

                            FT in_sig = m_border_value;
                            if ( iy >= 0 && iy < m_input_h_size && ix >= 0 && ix < m_input_w_size ) {
                                index_t input_node  = (c * m_input_h_size  + iy) * m_input_w_size  + ix;
                                in_sig = x_ptr.Get(input_frame, input_node);
                            }
                            else {
                              if ( Border(m_border_mode, ix, iy, m_input_w_size, m_input_h_size) ) {
                                    index_t input_node = (c * m_input_h_size  + iy) * m_input_w_size  + ix;
                                    in_sig = x_ptr.Get(input_frame, input_node);
                                }
                            }

                            index_t output_node = (c * m_filter_h_size + fy) * m_filter_w_size + fx;    
                            y_ptr.Set(output_frame, output_node, in_sig);
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
        
        // 出力設定
        FrameBuffer dx_buf(m_input_frame_size, m_input_shape, DataType<BT>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<BT>::type == BB_TYPE_FP32 && !m_host_only && dy_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = dx_buf.LockDeviceMemory();
            bbcu_fp32_Im2Col_Backward(
                (float const *)ptr_dy.GetAddr(),
                (float       *)ptr_dx.GetAddr(),
                (int          )m_x_stride,
                (int          )m_y_stride,
                (int          )m_x_offset,
                (int          )m_y_offset,
                (int          )m_input_frame_size,
                (int          )(dx_buf.GetFrameStride() / sizeof(float)),
                (int          )m_input_w_size,
                (int          )m_input_h_size,
                (int          )m_input_c_size,
                (int          )m_output_w_size,
                (int          )m_output_h_size,
                (int          )(dy_buf.GetFrameStride() / sizeof(float)),
                (int          )m_filter_w_size,
                (int          )m_filter_h_size);
            return dx_buf;
        }
#endif

        {
            // stride版
            dx_buf.FillZero();

            auto dy_ptr = dy_buf.LockConst<BT>();
            auto dx_ptr = dx_buf.Lock<BT>();

            index_t iy_limit = (m_output_h_size - 1) * m_y_stride;
            index_t ix_limit = (m_output_w_size - 1) * m_x_stride;

            for (index_t c = 0; c < m_input_c_size; ++c) {
                #pragma omp parallel for
                for (index_t y = 0; y < m_input_h_size; ++y ) {
                    #pragma omp parallel for
                    for (index_t x = 0; x < m_input_w_size; ++x ) {
                        index_t input_node = (c * m_input_h_size + y) * m_input_w_size + x;
                        index_t x_align = x % m_x_stride;
                        index_t y_align = y % m_y_stride;
                        for ( index_t input_frame = 0; input_frame < m_input_frame_size; ++input_frame ) {
                            BT dx = 0; // dx_ptr.Get(input_frame, input_node);
                            float dy = 0;
                            for (index_t fy = y_align; fy < m_filter_h_size; fy += m_y_stride ) {
                                index_t iy = y - fy + m_y_offset;
                                if ( iy >= 0 && iy <= iy_limit ) {
                                    for (index_t fx = x_align; fx < m_filter_w_size; fx += m_x_stride) {
                                        index_t ix = x - fx + m_x_offset;
                                        if ( ix >= 0 && ix <= ix_limit ) {
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
        bb::SaveValue(os, m_input_frame_size);
        bb::SaveValue(os, m_output_frame_size);
        bb::SaveValue(os, m_input_c_size);
        bb::SaveValue(os, m_input_h_size);
        bb::SaveValue(os, m_input_w_size);
        bb::SaveValue(os, m_filter_h_size);
        bb::SaveValue(os, m_filter_w_size);
        bb::SaveValue(os, m_y_stride);
        bb::SaveValue(os, m_x_stride);
        bb::SaveValue(os, m_padding);
        bb::SaveValue(os, m_border_mode);
        bb::SaveValue(os, m_border_value);
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
        bb::LoadValue(is, m_input_frame_size);
        bb::LoadValue(is, m_output_frame_size);
        bb::LoadValue(is, m_input_c_size);
        bb::LoadValue(is, m_input_h_size);
        bb::LoadValue(is, m_input_w_size);
        bb::LoadValue(is, m_filter_h_size);
        bb::LoadValue(is, m_filter_w_size);
        bb::LoadValue(is, m_y_stride);
        bb::LoadValue(is, m_x_stride);
        bb::LoadValue(is, m_padding);
        bb::LoadValue(is, m_border_mode);
        bb::LoadValue(is, m_border_value);
        
        // 再構築
        m_input_shape  = bb::indices_t({m_input_c_size, m_input_h_size, m_input_w_size});
        m_output_shape = bb::indices_t({m_input_c_size, m_filter_h_size, m_filter_w_size});
        if ( m_padding == "valid" ) {
            m_output_h_size = ((m_input_h_size - m_filter_h_size + 1) + (m_y_stride - 1)) / m_y_stride;
            m_output_w_size = ((m_input_w_size - m_filter_w_size + 1) + (m_x_stride - 1)) / m_x_stride;
            m_y_offset = 0;
            m_x_offset = 0;
        }
        else if ( m_padding == "same" ) {
            m_output_h_size = (m_input_h_size + (m_y_stride - 1)) / m_y_stride;
            m_output_w_size = (m_input_w_size + (m_x_stride - 1)) / m_x_stride;
            m_y_offset = (m_filter_h_size - 1) / 2;
            m_x_offset = (m_filter_w_size - 1) / 2;
        }
        else {
            BB_ASSERT(0);
        }
    }

};


}