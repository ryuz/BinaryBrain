// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <algorithm>
#include <random>

#include "bb/Model.h"
#include "bb/ValueGenerator.h"


namespace bb {



/**
 * @brief   バイナリ変調された確率値の再無相関化を狙ったシャッフル層
 * @details バイナリ変調したユニット内でのシャッフルを行う
 * 
 * @tparam FT  foward入力型 (x, y)
 * @tparam BT  backward型 (dy, dx)
 */
template <typename FT = Bit, typename BT = float>
class ShuffleModulation : public Model
{
protected:
    FrameBuffer         m_y_buf;

    indices_t           m_node_shape;
    index_t             m_shuffle_size = 1;
    index_t             m_lowering_size = 1;
    std::mt19937_64     m_mt;


public:
    struct create_t
    {
        index_t         shuffle_size  = 1;      //< シャッフルする単位
        index_t         lowering_size = 1;
        std::uint64_t   seed          = 1;
    };

protected:
    ShuffleModulation(create_t const &create)
    {
        m_shuffle_size = create.shuffle_size;
        m_lowering_size = create.lowering_size;
        m_mt.seed(create.seed);
    }

public:
    ~ShuffleModulation() {}


    static std::shared_ptr<ShuffleModulation> Create(create_t const &create)
    {
        return std::shared_ptr<ShuffleModulation>(new ShuffleModulation(create));
    }

    static std::shared_ptr<ShuffleModulation> Create(
                index_t         shuffle_size = 1,
                index_t         lowering_size = 1,
                std::uint64_t   seed       = 1)
    {
        create_t create;
        create.shuffle_size  = shuffle_size;
        create.lowering_size = lowering_size;
        create.seed          = seed;
        return Create(create);
    }

    std::string GetClassName(void) const { return "ShuffleModulation"; }


    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 形状設定
        m_node_shape = shape;
        return m_node_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_node_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_node_shape;
    }
    

    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_node_shape) {
            SetInputShape(x_buf.GetShape());
        }

        // 戻り値の型を設定
        m_y_buf.Resize(DataType<FT>::type, x_buf.GetFrameSize(), m_node_shape);

        index_t node_size  = x_buf.GetNodeSize();
        index_t frame_size = x_buf.GetFrameSize();

        BB_ASSERT(frame_size % (m_shuffle_size * m_lowering_size) == 0);

        auto x_ptr = x_buf.LockConst<FT>();
        auto y_ptr = m_y_buf.Lock<FT>();

        std::vector<int> table(m_shuffle_size);
        for ( int i = 0; i < (int)m_shuffle_size; ++i ) {
            table[i] = i;
        }

        for ( index_t node = 0; node < node_size; ++node) {
            for ( index_t frame = 0; frame < frame_size; frame += (m_shuffle_size * m_lowering_size)) {
                for ( index_t i = 0; i < m_lowering_size; ++i ) {
                    std::shuffle(table.begin(), table.end(), m_mt);
                    for ( index_t j = 0; j < m_shuffle_size; ++j ) {
                        auto x = x_ptr.Get(frame + (table[j] * m_lowering_size) + i, node);
                        y_ptr.Set(frame + (j * m_lowering_size) + i, node, x);
                    }
                }
            }
        }

        return m_y_buf;
    }

    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(0);   // backwardは未定義
        return dy_buf;
    }
};


}


// end of file
