// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>


#include "bb/Object.h"
#include "bb/FrameBuffer.h"


namespace bb {


class LossFunction : public Object
{
public:
    virtual std::string GetLossFunctionName(void) const = 0;

    /**
     * @brief  積算していた損失をクリア
     * @detail 積算していた損失をクリアする
     */
    virtual void Clear(void) = 0;

   /**
     * @brief  損失取得
     * @detail 損失取得
     *         損失の取得にGPUからのメモリコピーが発生する可能性があるので
     *         CalculateLoss とは別メソッドにする
     * @return 積算していた損失を返す
     */
    virtual double GetLoss(void) const = 0;

    /**
     * @brief  損失計算
     * @detail 損失を計算する
     * @param  y    結果の入力
     * @param  t    期待値
     * @return backwardする誤差勾配を返す
     */
    virtual FrameBuffer CalculateLoss(FrameBuffer y_buf, FrameBuffer t_buf, index_t mini_batch_size) = 0;
};


}

