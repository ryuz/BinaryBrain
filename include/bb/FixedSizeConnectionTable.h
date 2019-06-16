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

#include "bb/ConnectionTable.h"
#include "bb/Tensor.h"
#include "bb/Utility.h"


namespace bb {


// 入力接続固定のレイヤーモデル
template <int N = 6, typename IndexType = std::int32_t>
class FixedSizeConnectionTable : public ConnectionTable
{
    using _super = ConnectionTable;

protected:
    Tensor_<IndexType>   m_input_table;
    Tensor_<IndexType>   m_reverse_table;
    bool                 m_reverse_table_dirty = true;

public:
    // Serialize
    void Save(std::ostream &os) const 
    {
        _super::Save(os);
        m_input_table.Save(os);
    }

    void Load(std::istream &is)
    {
        _super::Load(is);
        m_input_table.Load(is);
    }

#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
        archive(cereal::make_nvp("input_table",  m_input_table));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("input_table",  m_input_table));
    }
#endif


public:
    // set shape
    void SetShape(indices_t input_shape, indices_t output_shape)
    {
        _super::SetShape(input_shape, output_shape);
        m_input_table.Resize(this->GetOutputNodeSize(), N);
    }

    index_t GetInputConnectionSize(index_t output_node) const
    {
        return N;
    }

    index_t GetInputConnection(index_t output_node, index_t connection_index) const
    {
        auto ptr = LockConst_InputTable();
        return (index_t)ptr(output_node, connection_index);
    }

    void SetInputConnection(index_t output_node, index_t connection_index, index_t input_node)
    {
        auto ptr = Lock_InputTable();
        ptr(output_node, connection_index) = (IndexType)input_node;
    }
    

    // Lock
    auto Lock_InputTable(void)
    {
        m_reverse_table_dirty = true;
        return m_input_table.Lock();
    }

    auto LockConst_InputTable(void) const
    {
        return m_input_table.LockConst();
    }

    auto LockDeviceMemConst_InputTable(void)
    {
        return m_input_table.LockDeviceMemoryConst();
    }


    auto LockConst_ReverseTable(void)
    {
        BuildReverseTable();
        return m_reverse_table.LockConst();
    }

    auto LockDeviceMemConst_ReverseTable(void)
    {
        BuildReverseTable();
        return m_reverse_table.LockDeviceMemoryConst();
    }

    index_t GetReverseTableStride(void)
    {
        return m_reverse_table.GetShape()[0];
    }
    

protected:
    void BuildReverseTable(void)
    {
        if ( !m_reverse_table_dirty ) {
            return;
        }
        m_reverse_table_dirty = true;

        auto input_node_size  = this->GetInputNodeSize();
        auto output_node_size = this->GetOutputNodeSize();

        auto input_table_ptr = m_input_table.LockConst();
        std::vector<index_t> n(input_node_size, 0);
        for ( index_t node = 0; node < output_node_size; ++node ) {
            for ( index_t input = 0; input < N; ++input ) {
                n[input_table_ptr(node, input)]++;
            }
        }

        index_t max_n = 0;
        for ( index_t node = 0; node < input_node_size; ++node ) {
            max_n = std::max(max_n, n[node]);
        }

        m_reverse_table.Resize(indices_t({max_n+1, input_node_size}));
        m_reverse_table = 0;

        auto reverse_table_ptr = m_reverse_table.Lock();
        for ( index_t node = 0; node < output_node_size; ++node ) {
            for ( index_t input = 0; input < N; ++input ) {
                std::int32_t idx = input_table_ptr(node, input);
                auto cnt = reverse_table_ptr(idx, 0) + 1;
                reverse_table_ptr(idx, 0)   = cnt;
                reverse_table_ptr(idx, cnt) = (std::int32_t)(node*N + input);
            }
        }
    }
};


}

