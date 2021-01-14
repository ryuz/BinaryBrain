// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2020 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Manager.h"
#include "bb/Model.h"


namespace bb {


// Reshape
class Reshape : public Model
{
    using _super = Model;

protected:
    indices_t   m_input_shape;
    indices_t   m_output_shape;

public:
    struct create_t
    {
        indices_t   output_shape;
    };
    
protected:
    Reshape(create_t const &create) {
        m_output_shape = create.output_shape;
    }

public:
    static std::shared_ptr<Reshape> Create(create_t const &create)
    {
        return std::shared_ptr<Reshape>(new Reshape(create));
    }

    static std::shared_ptr<Reshape> Create(indices_t const &output_shape)
    {
        create_t create;
        create.output_shape = output_shape;
        return Create(create);
    }

    ~Reshape() {}

    std::string GetModelName(void) const { return "Reshape"; }

    
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
        m_input_shape = shape;
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


    /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    inline FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        BB_ASSERT(x_buf.GetNodeSize() == CalcShapeSize(m_output_shape));

        m_input_shape = x_buf.GetShape();
        x_buf.Reshape(m_output_shape);

        return x_buf;
    }


   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    inline FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(dy_buf.GetNodeSize() == CalcShapeSize(m_input_shape));

        dy_buf.Reshape(m_input_shape);

        return dy_buf;
    }
};


}

// end of file