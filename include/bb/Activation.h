// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Manager.h"
#include "bb/Model.h"


namespace bb {


// Activation
class Activation : public Model
{
protected:
    indices_t   m_shape;    //< 入出力の形状 

public:
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
        m_shape = shape;
        return m_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_shape;
    }
};


};

