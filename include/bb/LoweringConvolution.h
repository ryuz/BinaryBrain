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
    using super = Filter2d<FT, BT>;

protected:
    index_t m_filter_h_size = 1;
    index_t m_filter_w_size = 1;

    // 3層で構成
	std::shared_ptr< ConvolutionIm2Col<FT, BT> >	m_im2col;
	std::shared_ptr< Model                     >    m_layer;
	std::shared_ptr< ConvolutionCol2Im<FT, BT> >	m_col2im;
	
protected:
	LoweringConvolution() {}

public:
	~LoweringConvolution() {}

    struct create_t
    {
        std::shared_ptr<Model>  layer;
        index_t                 filter_h_size = 1;
        index_t                 filter_w_size = 1;
    };

    static std::shared_ptr<LoweringConvolution> Create(create_t const & create)
	{
        auto self = std::shared_ptr<LoweringConvolution>(new LoweringConvolution);
        
        self->m_filter_w_size = create.filter_w_size;
        self->m_filter_h_size = create.filter_h_size;

  		self->m_im2col = ConvolutionIm2Col<FT, BT>::Create(self->m_filter_h_size, self->m_filter_w_size);
        self->m_layer  = create.layer;
        // col2im の形状は入力形状確定時に決まる

        return self;
	}

    static std::shared_ptr<LoweringConvolution> Create(std::shared_ptr<Model> layer, index_t filter_h_size, index_t filter_w_size)
	{
        auto self = std::shared_ptr<LoweringConvolution>(new LoweringConvolution);
        
        self->m_filter_w_size = filter_w_size;
        self->m_filter_h_size = filter_h_size;

  		self->m_im2col = ConvolutionIm2Col<FT, BT>::Create(filter_h_size, filter_w_size);
        self->m_layer  = layer;
        // col2im の形状は入力形状確定時に決まる

        return self;
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
        index_t input_c_size = shape[2];
		index_t output_w_size = input_w_size - m_filter_w_size + 1;
		index_t output_h_size = input_h_size - m_filter_h_size + 1;

		m_col2im = ConvolutionCol2Im<FT, BT>::Create(output_h_size, output_w_size);

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
    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
	    x = m_im2col->Forward(x, train);
	    x = m_layer->Forward(x, train);
	    x = m_col2im->Forward(x, train);
        return x;
    }

   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    FrameBuffer Backward(FrameBuffer dy)
    {
	    dy = m_col2im->Backward(dy);
	    dy = m_layer->Backward(dy);
	    dy = m_im2col->Backward(dy);
        return dy; 
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
        // これ以上ネストしないなら自クラス概要
        if ( depth > 0 && (nest+1) >= depth ) {
            Model::PrintInfoText(os, indent, columns, nest, depth);
        }
        else {
            // 子レイヤーの表示
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
        super::save(archive, version);
        archive(cereal::make_nvp("filter_h_size", m_filter_h_size));
        archive(cereal::make_nvp("filter_w_size", m_filter_w_size));
    }

	template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
	{
        super::load(archive, version);
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