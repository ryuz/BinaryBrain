// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                 Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                 https://github.com/ryuz
//                                 ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <set>
#include <algorithm>

#include "bb/SparseLayer.h"
#include "bb/Utility.h"


namespace bb {


// 入力接続固定のレイヤーモデル
template <int N = 6>
class SparseN : public SparseLayer
{
    using _super = SparseLayer;

private:
    indices_t               m_input_shape;
    indices_t               m_output_shape;

    Tensor_<std::int32_t>   m_input_index;
    Tensor_<std::int32_t>   m_reverse_index;
    bool                    m_reverse_index_dirty = true;


protected:
    // Serialize
    void Save(std::ostream &os) const 
    {
        SaveIndices(os, m_input_shape);
        SaveIndices(os, m_output_shape);
        m_input_index.Save(os);
    }

    void Load(std::istream &is)
    {
        m_input_shape  = LoadIndices(is);
        m_output_shape = LoadIndices(is);
        m_input_index.Load(is);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
        archive(cereal::make_nvp("input_shape",  m_input_shape));
        archive(cereal::make_nvp("output_shape", m_output_shape));
        archive(cereal::make_nvp("input_index",  m_input_index));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("input_shape",  m_input_shape));
        archive(cereal::make_nvp("output_shape", m_output_shape));
        archive(cereal::make_nvp("input_index",  m_input_index));
    }

    /*
    void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("SparseN", *this));
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("SparseN", *this));
    }
    */
#endif


public:
    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    void SetOutputShape(indices_t output_shape)
    {
        m_output_shape = output_shape;
        m_input_index.Resize(GetShapeSize(m_output_shape), N);
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
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t input_shape)
    {
        m_input_shape = input_shape;
        m_input_index.Resize(GetOutputNodeSize(), N);
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



    index_t GetNodeInputSize(index_t node) const
    {
        return N;
    }

    void SetNodeInput(index_t node, index_t input_index, index_t input_node)
    {
        auto ptr = lock_InputIndex();
        ptr(node, input_index) = (std::int32_t)input_node;
    }

    index_t GetNodeInput(index_t node, index_t input_index) const
    {
        auto ptr = lock_InputIndex_const();
        return (index_t)ptr(node, input_index);
    }
    
protected:
    auto lock_InputIndex(void)
    {
        m_reverse_index_dirty = true;
        return m_input_index.Lock();
    }

    auto lock_InputIndex_const(void) const
    {
        return m_input_index.LockConst();
    }

    auto lockDeviceMem_InputIndex_const(void)
    {
        return m_input_index.LockDeviceMemoryConst();
    }


    auto lock_ReverseIndex_const(void) const
    {
        BuildReverseIndexTable();
        return m_reverse_index.LockConst();
    }

    auto lockDeviceMem_ReverseIndex_const(void)
    {
        BuildReverseIndexTable();
        return m_reverse_index.LockDeviceMemoryConst();
    }

    index_t GetReverseIndexStride(void)
    {
        return m_reverse_index.GetShape()[0];
    }
         
    void BuildReverseIndexTable(void)
    {
        if ( !m_reverse_index_dirty ) {
            return;
        }
        m_reverse_index_dirty = true;

        auto input_node_size  = GetInputNodeSize();
        auto output_node_size = GetOutputNodeSize();

        auto input_index_ptr = m_input_index.LockConst();
        std::vector<index_t> n(input_node_size, 0);
        for ( index_t node = 0; node < output_node_size; ++node ) {
            for ( index_t input = 0; input < N; ++input ) {
                n[input_index_ptr(node, input)]++;
            }
        }

        index_t max_n = 0;
        for ( index_t node = 0; node < input_node_size; ++node ) {
            max_n = std::max(max_n, n[node]);
        }

        m_reverse_index.Resize(indices_t({max_n+1, input_node_size}));
        m_reverse_index = 0;

        auto reverse_index_ptr = m_reverse_index.Lock();
        for ( index_t node = 0; node < output_node_size; ++node ) {
            for ( index_t input = 0; input < N; ++input ) {
                std::int32_t idx = input_index_ptr(node, input);
                auto cnt = reverse_index_ptr(idx, 0) + 1;
                reverse_index_ptr(idx, 0)   = cnt;
                reverse_index_ptr(idx, cnt) = (std::int32_t)(node*N + input);
            }
        }
    }
};


}

