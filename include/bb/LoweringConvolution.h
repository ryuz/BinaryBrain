// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>

#include "bb/Filter2d.h"
#include "bb/ConvolutionIm2Col.h"
#include "bb/ConvolutionCol2Im.h"


namespace bb {


// 入力数制限Affine Binary Connect版
template <typename FT = float, typename BT = float>
class LoweringConvolution : public Filter2d<FT, BT>
{
    using _super = Filter2d<FT, BT>;

protected:
    index_t     m_filter_h_size = 1;
    index_t     m_filter_w_size = 1;
    index_t     m_x_stride = 1;
    index_t     m_y_stride = 1;
    index_t     m_output_w_size = 1;
    index_t     m_output_h_size = 1;
    std::string m_padding = "valid";

    using Im2Col = ConvolutionIm2Col<FT, BT>;
    using Col2Im = ConvolutionCol2Im<FT, BT>;

    // 3層で構成
    std::shared_ptr< Im2Col >    m_im2col;
    std::shared_ptr< Model  >    m_layer;
    std::shared_ptr< Col2Im >    m_col2im;

public:
    struct create_t
    {
        std::shared_ptr<Model>  layer;
        index_t                 filter_h_size = 1;
        index_t                 filter_w_size = 1;
        index_t                 x_stride      = 1;
        index_t                 y_stride      = 1;
        std::string             padding       = "valid";
        int                     border_mode   = BB_BORDER_REFLECT_101;
        FT                      border_value  = (FT)0;
    };
    
protected:
    LoweringConvolution(create_t const & create)
    {
        m_filter_w_size = create.filter_w_size;
        m_filter_h_size = create.filter_h_size;
        m_x_stride      = create.x_stride;
        m_y_stride      = create.y_stride;
        m_padding       = create.padding;
       
        typename ConvolutionIm2Col<FT, BT>::create_t im2col_create;
        im2col_create.filter_h_size = create.filter_h_size;
        im2col_create.filter_w_size = create.filter_w_size;
        im2col_create.x_stride      = create.x_stride;
        im2col_create.y_stride      = create.y_stride;
        im2col_create.padding       = create.padding;
        im2col_create.border_mode   = create.border_mode;
        im2col_create.border_value  = create.border_value;
        m_im2col = ConvolutionIm2Col<FT, BT>::Create(im2col_create);
        m_layer  = create.layer;
        // col2im の形状は入力形状確定時に決まる
    }

public:
    ~LoweringConvolution() {}

    static std::shared_ptr<LoweringConvolution> Create(create_t const & create)
    {
        return std::shared_ptr<LoweringConvolution>(new LoweringConvolution(create));
    }

    static std::shared_ptr<LoweringConvolution> Create(std::shared_ptr<Model> layer,
        index_t filter_h_size, index_t filter_w_size, index_t y_stride=1, index_t x_stride=1, std::string padding="valid")
    {
        create_t create;
        create.layer         = layer;
        create.filter_h_size = filter_h_size;
        create.filter_w_size = filter_w_size;
        create.y_stride      = y_stride;
        create.x_stride      = x_stride;
        create.padding       = padding;
        return Create(create);
    }

    static std::shared_ptr<LoweringConvolution> CreateEx(
            std::shared_ptr<Model>  layer,
            index_t                 filter_h_size,
            index_t                 filter_w_size,
            index_t                 y_stride      = 1,
            index_t                 x_stride      = 1,
            std::string             padding       = "valid",
            int                     border_mode   = BB_BORDER_REFLECT_101,
            FT                      border_value  = (FT)0
        )
    {
        create_t create;
        create.layer         = layer;
        create.filter_h_size = filter_h_size;
        create.filter_w_size = filter_w_size;
        create.y_stride      = y_stride;
        create.x_stride      = x_stride;
        create.padding       = padding;
        create.border_mode   = border_mode;
        create.border_value  = border_value;
        return Create(create);
    }

    std::string GetClassName(void) const { return "LoweringConvolution"; }

    
    std::shared_ptr< Model > GetLayer(void)
    {
        return m_layer;
    }


    index_t GetFilterHeight(void) { return m_filter_h_size; }
    index_t GetFilterWidth(void)  { return m_filter_w_size; }


    /**
     * @brief  コマンドを送る
     * @detail コマンドを送る
     */   
    void SendCommand(std::string command, std::string send_to = "all")
    {
        m_im2col->SendCommand(command, send_to);
        m_layer->SendCommand(command, send_to);
        m_col2im->SendCommand(command, send_to);
    }
    
    /**
     * @brief  パラメータ取得
     * @detail パラメータを取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetParameters(void)
    {
        Variables parameters;
        parameters.PushBack(m_im2col->GetParameters());
        parameters.PushBack(m_layer->GetParameters());
        parameters.PushBack(m_col2im->GetParameters());
        return parameters;
    }

    /**
     * @brief  勾配取得
     * @detail 勾配を取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    virtual Variables GetGradients(void)
    {
        Variables gradients;
        gradients.PushBack(m_im2col->GetGradients());
        gradients.PushBack(m_layer->GetGradients());
        gradients.PushBack(m_col2im->GetGradients());
        return gradients;
    }  

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

        index_t input_w_size = shape[0];
        index_t input_h_size = shape[1];
//      index_t input_c_size = shape[2];

        // 出力サイズ計算
        if ( m_padding == "valid" ) {
            m_output_h_size = ((input_h_size - m_filter_h_size + 1) + (m_y_stride - 1)) / m_y_stride;
            m_output_w_size = ((input_w_size - m_filter_w_size + 1) + (m_x_stride - 1)) / m_x_stride;
        }
        else if ( m_padding == "same" ) {
            m_output_h_size = (input_h_size + (m_y_stride - 1)) / m_y_stride;
            m_output_w_size = (input_w_size + (m_x_stride - 1)) / m_x_stride;
        }
        else {
            BB_ASSERT(0);
        }

        m_col2im = ConvolutionCol2Im<FT, BT>::Create(m_output_h_size, m_output_w_size);

        shape = m_im2col->SetInputShape(shape);
        shape = m_layer->SetInputShape(shape);
        shape = m_col2im->SetInputShape(shape);

        return shape;
    }


    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_im2col->GetInputShape();
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_col2im->GetOutputShape();
    }
    

   /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        x_buf = m_im2col->Forward(x_buf, train);
        x_buf = m_layer->Forward(x_buf, train);
        x_buf = m_col2im->Forward(x_buf, train);
        return x_buf;
    }

   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        dy_buf = m_col2im->Backward(dy_buf);
        dy_buf = m_layer->Backward(dy_buf);
        dy_buf = m_im2col->Backward(dy_buf);
        return dy_buf; 
    }
    
protected:
    /**
     * @brief  モデルの情報を表示
     * @detail モデルの情報を表示する
     * @param  os     出力ストリーム
     * @param  indent インデント文字列
     */
    void PrintInfoText(std::ostream& os, std::string indent, int columns, int nest, int depth)
    {
        _super::PrintInfoText(os, indent, columns, nest, depth);

        // 子レイヤーの表示
        if ( depth == 0 || (nest+1) < depth ) {
            m_im2col->PrintInfo(depth, os, columns, nest+1);
            m_layer->PrintInfo(depth, os, columns, nest+1);
            m_col2im->PrintInfo(depth, os, columns, nest+1);
        }
    }

public:
    // Serialize
    void Save(std::ostream &os) const 
    {
        SaveValue(os, m_filter_h_size);
        SaveValue(os, m_filter_w_size);

        m_im2col->Save(os);
        m_layer->Save(os);
        m_col2im->Save(os);
    }

    void Load(std::istream &is)
    {
        LoadValue(is, m_filter_h_size);
        LoadValue(is, m_filter_w_size);

        m_im2col->Load(is);
        m_layer->Load(is);
        m_col2im->Load(is);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
        archive(cereal::make_nvp("filter_h_size", m_filter_h_size));
        archive(cereal::make_nvp("filter_w_size", m_filter_w_size));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("filter_h_size", m_filter_h_size));
        archive(cereal::make_nvp("filter_w_size", m_filter_w_size));
    }

    void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("LoweringConvolution", *this));
        m_im2col->Save(archive);
        m_layer->Save(archive);
        m_col2im->Save(archive);
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("LoweringConvolution", *this));
        m_im2col->Load(archive);
        m_layer->Load(archive);
        m_col2im->Load(archive);
    }
#endif
};


}