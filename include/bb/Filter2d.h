// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Model.h"


namespace bb {


// border_mode
#define BB_BORDER_CONSTANT      0
#define BB_BORDER_REFLECT       1
#define BB_BORDER_REFLECT_101   2
#define BB_BORDER_REPLICATE     3
#define BB_BORDER_WRAP          4


// 二次元フィルタの基本クラス
class Filter2d : public Model
{
protected:
    /**
     * @brief  モデルの情報を表示
     * @detail モデルの情報を表示する
     * @param  os     出力ストリーム
     * @param  indent インデント文字列
     */
    void PrintInfoText(std::ostream& os, std::string indent, int columns, int nest, int depth) const override
    {
        os << indent << " filter size : (" << GetFilterHeight() << ", " << GetFilterWidth() << ")" << std::endl;
        Model::PrintInfoText(os, indent, columns, nest, depth);
    }

public:
    virtual index_t GetFilterHeight(void) const = 0;
    virtual index_t GetFilterWidth(void) const = 0;

    virtual std::shared_ptr< Model > GetSubLayer(void) const
    {
        return nullptr;
    }

    index_t GetInputChannels(void) const
    {
        auto shape = this->GetInputShape();
        BB_ASSERT(shape.size() == 3);
        return shape[0];
    }

    index_t GetInputHeight(void) const
    {
        auto shape = this->GetInputShape();
        BB_ASSERT(shape.size() == 3);
        return shape[1];
    }

    index_t GetInputWidth(void) const
    {
        auto shape = this->GetInputShape();
        BB_ASSERT(shape.size() == 3);
        return shape[2];
    }

    index_t GetOutputChannels(void) const
    {
        auto shape = this->GetOutputShape();
        BB_ASSERT(shape.size() == 3);
        return shape[0];
    }

    index_t GetOutputHeight(void) const
    {
        auto shape = this->GetOutputShape();
        BB_ASSERT(shape.size() == 3);
        return shape[1];
    }

    index_t GetOutputWidth(void) const
    {
        auto shape = this->GetOutputShape();
        BB_ASSERT(shape.size() == 3);
        return shape[2];
    }
};


}