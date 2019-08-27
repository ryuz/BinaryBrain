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
    bool                    m_host_only = false;

    FrameBuffer             m_y_buf;

    Tensor_<std::int32_t>   m_table;

    indices_t               m_node_shape;
    index_t                 m_shuffle_size = 1;
    index_t                 m_lowering_size = 1;
    std::mt19937_64         m_mt;


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

        auto node_size = GetShapeSize(shape);
        m_table.Resize(node_size, m_shuffle_size);

        std::vector<int> table(m_shuffle_size);
        for ( int i = 0; i < (int)m_shuffle_size; ++i ) {
            table[i] = i;
        }

        auto table_ptr = m_table.Lock(true);
        for ( index_t node = 0; node < node_size; ++node) {
            std::shuffle(table.begin(), table.end(), m_mt);
            for ( index_t i = 0; i < m_shuffle_size; ++i ) {
                table_ptr(node, i) = table[i];
            }
        }
        
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
        BB_ASSERT(x_buf.GetFrameSize() % (m_shuffle_size * m_lowering_size) == 0);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_node_shape) {
            SetInputShape(x_buf.GetShape());
        }

        // 戻り値の型を設定
        m_y_buf.Resize(x_buf.GetFrameSize(), m_node_shape, DataType<FT>::type);

#ifdef BB_WITH_CUDA
        if ( false && DataType<FT>::type == BB_TYPE_BIT && !m_host_only
                && x_buf.IsDeviceAvailable() && m_y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            // GPU版
            auto x_ptr     = x_buf.LockDeviceMemoryConst();
            auto y_ptr     = m_y_buf.LockDeviceMemory(true);
            auto table_ptr = m_table.LockDeviceMemoryConst();

            bbcu_bit_ShuffleModulation_Forward
                (
                    (int const *)x_ptr.GetAddr(),
                    (int       *)y_ptr.GetAddr(),
                    (int const *)table_ptr.GetAddr(),
                    (int        )m_shuffle_size,
                    (int        )m_lowering_size,
                    (int        )x_buf.GetNodeSize(),
                    (int        )x_buf.GetFrameSize(),
                    (int        )(x_buf.GetFrameStride() / sizeof(int))
                );

            return m_y_buf;
        }
#endif

         if ( DataType<FT>::type == BB_TYPE_BIT ) {
            // Bit版
            int node_size     = (int)x_buf.GetNodeSize();
            int frame_size    = (int)x_buf.GetFrameSize();
            int frame_stride  = (int)(x_buf.GetFrameStride() / sizeof(int));
            int shuffle_size  = (int)m_shuffle_size;
            int lowering_size = (int)m_lowering_size;

            auto x_ptr     = x_buf.LockMemoryConst();
            auto y_ptr     = m_y_buf.LockMemory();
            auto table_ptr = m_table.LockMemoryConst();

            auto x_addr     = (int const *)x_ptr.GetAddr();
            auto y_addr     = (int       *)y_ptr.GetAddr();    
            auto table_addr = (int const *)table_ptr.GetAddr();

            for ( int node = 0; node < node_size; ++node) {
                for ( int f = 0; f < frame_size/32; ++f ) {
                    int y = 0;
                    for ( int bit = 0; bit < 32; ++bit ) {
                        int frame = f*32 + bit;
                        if ( frame < frame_size ) {
                            int i = frame / (lowering_size * shuffle_size);
                            int j = frame / lowering_size % shuffle_size;
                            int k = frame % lowering_size;

                            int input_frame  = i * (lowering_size * shuffle_size) + table_addr[node * shuffle_size + j] * lowering_size + k;
                            int output_frame = i * (lowering_size * shuffle_size) +                                  j  * lowering_size + k;
                            int x = ((x_addr[node*frame_stride + (input_frame / 32)] >> (input_frame % 32)) & 1);
                            y |= (x << bit);
                        }
                    }
                    y_addr[node*frame_stride + f] = y;
                }
            }
            return m_y_buf;
        }


        {
            // 汎用版
            index_t node_size  = x_buf.GetNodeSize();
            index_t frame_size = x_buf.GetFrameSize();

            auto x_ptr = x_buf.LockConst<FT>();
            auto y_ptr = m_y_buf.Lock<FT>();

//          std::vector<int> table(m_shuffle_size);
//          for ( int i = 0; i < (int)m_shuffle_size; ++i ) {
//              table[i] = i;
//          }

            auto table_ptr = m_table.Lock(true);
            for ( index_t node = 0; node < node_size; ++node) {
                for ( index_t frame = 0; frame < frame_size; frame += (m_shuffle_size * m_lowering_size)) {
                    for ( index_t i = 0; i < m_shuffle_size; ++i ) {
    //                  std::shuffle(table.begin(), table.end(), m_mt);
                        for ( index_t j = 0; j < m_lowering_size; ++j ) {
    //                      auto x = x_ptr.Get(frame + (table[i] * m_lowering_size) + j, node);
                            auto x = x_ptr.Get(frame + (table_ptr(node, i) * m_lowering_size) + j, node);
                            y_ptr.Set(frame + (i * m_lowering_size) + j, node, x);
                        }
                    }
                }
            }

            return m_y_buf;
        }
    }

    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(0);   // backwardは未定義
        return dy_buf;
    }
};


}


// end of file
