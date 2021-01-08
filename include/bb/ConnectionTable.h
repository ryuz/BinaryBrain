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

#include "bb/Object.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"


#if BB_WITH_CEREAL
#include "cereal/types/array.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/json.hpp"
#endif


namespace bb {

// 接続テーブル
class ConnectionTable : public Object
{
    using _super = Object;

public:
    // Table management
    virtual index_t GetInputConnectionSize(index_t output_node) const = 0;
    virtual index_t GetInputConnection(index_t output_node, index_t connection_index) const = 0;
    virtual void    SetInputConnection(index_t output_node, index_t connection_index, index_t input_node) = 0;

protected:
    indices_t   m_input_shape;
    indices_t   m_output_shape;


protected:
    void DumpObjectData(std::ostream &os) const
    {
        // バージョン
        std::int64_t ver = 1;
        bb::SaveValue(os, ver);

        // 親クラス
        _super::DumpObjectData(os);

        // メンバ
        bb::SaveIndices(os, m_input_shape);
        bb::SaveIndices(os, m_output_shape);
    }

    void LoadObjectData(std::istream &is)
    {
        // バージョン
        std::int64_t ver;
        bb::LoadValue(is, ver);

        BB_ASSERT(ver == 1);

        // 親クラス
        _super::LoadObjectData(is);

        // メンバ
        m_input_shape  = bb::LoadIndices(is);
        m_output_shape = bb::LoadIndices(is);
    }

public:
    // Serialize(旧)
    void Save(std::ostream &os) const 
    {
        SaveIndices(os, m_input_shape);
        SaveIndices(os, m_output_shape);
    }

    void Load(std::istream &is)
    {
        m_input_shape  = LoadIndices(is);
        m_output_shape = LoadIndices(is);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("input_shape",  m_input_shape));
        archive(cereal::make_nvp("output_shape", m_output_shape));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("input_shape",  m_input_shape));
        archive(cereal::make_nvp("output_shape", m_output_shape));
    }
#endif


    // Flatten
    index_t GetInputConnectionSize(indices_t output_node) const
    {
        return GetInputConnectionSize(ConvertIndicesToIndex(output_node, m_output_shape));
    }

    index_t GetInputConnection(indices_t output_node, index_t connection_index) const
    {
        return GetInputConnection(ConvertIndicesToIndex(output_node, m_output_shape), connection_index);
    }

    void SetInputTable(indices_t output_node, index_t connection_index, index_t input_node) 
    {
        SetInputConnection(ConvertIndicesToIndex(output_node, m_output_shape), connection_index, input_node);
    }

    void SetInputTable(index_t output_node, index_t connection_index, indices_t input_node) 
    {
        SetInputConnection(output_node, connection_index, ConvertIndicesToIndex(input_node, m_input_shape));
    }

    void SetInputConnection(indices_t output_node, index_t connection_index, indices_t input_node) 
    {
        SetInputConnection(ConvertIndicesToIndex(output_node, m_output_shape), connection_index, ConvertIndicesToIndex(input_node, m_input_shape));
    }

public:
    // set shape
    virtual void SetShape(indices_t input_shape, indices_t output_shape)
    {
        m_input_shape  = input_shape;
        m_output_shape = output_shape;
    }
    
    // accessor
    indices_t GetInputShape(void)  const { return m_input_shape;  }
    indices_t GetOutputShape(void) const { return m_output_shape; }
    index_t   GetInputNodeSize(void)  const { return CalcShapeSize(GetInputShape());  }
    index_t   GetOutputNodeSize(void) const { return CalcShapeSize(GetOutputShape());  }

    // Initialize connection
    void InitializeConnection(std::uint64_t seed, std::string connection = "")
    {
        auto input_shape  = this->GetInputShape();
        auto output_shape = this->GetOutputShape();

        auto input_node_size  = CalcShapeSize(input_shape);
        auto output_node_size = CalcShapeSize(output_shape);

        auto argv = SplitString(connection);

        if (argv.size() > 0 && argv[0] == "pointwise") {
            BB_ASSERT(input_shape.size() == 3);
            BB_ASSERT(output_shape.size() == 3);
            BB_ASSERT(input_shape[1] == output_shape[1]);
            BB_ASSERT(input_shape[2] == output_shape[2]);
            std::mt19937_64 mt(seed);
            for (index_t y = 0; y < output_shape[1]; ++y) {
                for (index_t x = 0; x < output_shape[2]; ++x) {
                    // shuffle index
                    ShuffleSet<index_t> ss(input_shape[0], mt());
                    for (index_t c = 0; c < output_shape[0]; ++c) {
                        // random connection
                        index_t  input_size = GetInputConnectionSize({c, y, x});
                        auto random_set = ss.GetRandomSet(input_size);
                        for (index_t i = 0; i < input_size; ++i) {
                            SetInputConnection({c, y, x}, i, {random_set[i], y, x});
                        }
                    }
                }
            }
            return;
        }

        if (argv.size() > 0 && argv[0] == "depthwise") {
            BB_ASSERT(input_shape.size() == 3);
            BB_ASSERT(output_shape.size() == 3);
            BB_ASSERT(input_shape[0] == output_shape[0]);
            std::mt19937_64 mt(seed);
            for (index_t c = 0; c < output_shape[0]; ++c) {
                // shuffle index
                ShuffleSet<index_t> ss(input_shape[1] * input_shape[2], mt());
                for (index_t y = 0; y < output_shape[1]; ++y) {
                    for (index_t x = 0; x < output_shape[2]; ++x) {
                        // random connection
                        index_t  input_size = GetInputConnectionSize({x, y, x});
                        auto random_set = ss.GetRandomSet(input_size);
                        for (index_t i = 0; i < input_size; ++i) {
                            index_t iy = random_set[i] / input_shape[2];
                            index_t ix = random_set[i] % input_shape[2];

                            index_t output_node = ConvertIndicesToIndex({c, y, x}, output_shape);
                            index_t input_node  = ConvertIndicesToIndex({c, iy, ix}, input_shape);

                            BB_ASSERT(output_node >= 0 && output_node < output_node_size);
                            BB_ASSERT(input_node  >= 0 && input_node  < input_node_size);
                            SetInputConnection(output_node, i, input_node);
                        }
                    }
                }
            }
            return;
        }

        if ( argv.size() > 0 && argv[0] == "gauss" ) {
            // ガウス分布で結線
            int n = (int)input_shape.size();
            std::vector<double> step(n);
            std::vector<double> sigma(n);
            for (int i = 0; i < n; ++i) {
                step[i]  = (double)(input_shape[i] - 1) / (double)(output_shape[i] - 1);
                sigma[i] = (double)input_shape[i] / (double)output_shape[i];
            }

            std::mt19937_64                     mt(seed);
            std::normal_distribution<double>    norm_dist(0.0, 1.0);
            indices_t           output_index(n, 0);
            do {
                // 入力の参照基準位置算出
                std::vector<double> input_offset(n);
                for (int i = 0; i < n; ++i) {
                    input_offset[i] = output_index[i] * step[i];
                }

                auto output_node = ConvertIndicesToIndex(output_index, output_shape);
                auto m = GetInputConnectionSize(output_node);
                std::set<index_t>   s;
                std::vector<double> input_position(n);
                for ( int i = 0; i < m; ++i ) {
                    for ( ; ; ) {
                        for ( int j = 0; j < n; ++j ) {
                            input_position[j] = input_offset[j] + norm_dist(mt) * sigma[j];
                        }
                        auto input_index = RegurerlizeIndices(input_position, input_shape);
                        auto input_node  = ConvertIndicesToIndex(input_index, input_shape);
                        if ( s.count(input_node) == 0 ){
                            SetInputConnection(output_node, i, input_node);
                            s.insert(input_node);
                            break;
                        }
                    }
                }
            } while ( CalcNextIndices(output_index, output_shape) );
            return;
        }

        if ( argv.size() > 0 && argv[0] == "serial" ) {
            // 連番結線
            index_t input_node = 0;
            for ( index_t output_node = 0; output_node < output_node_size; ++output_node ) {
                index_t m = GetInputConnectionSize(output_node);
                for ( index_t i = 0; i < m; ++i ) {
                    SetInputConnection(output_node, i, input_node % input_node_size);
                    ++input_node;
                }
            }
            return;
        }

        if ( argv.size() == 0 || argv[0] == "random" ) {
            // ランダム結線
            ShuffleSet<index_t> ss(input_node_size, seed);    // 接続先をシャッフル
            for (index_t node = 0; node < output_node_size; ++node) {
                // 入力をランダム接続
                index_t  input_size = GetInputConnectionSize(node);
                auto random_set = ss.GetRandomSet(input_size);
                for (index_t i = 0; i < input_size; ++i) {
                    SetInputConnection(node, i, random_set[i]);
                }
            }
            return;
        }

        std::cout << "unknown connection rule : \"" << argv[0] <<  "\"" << std::endl;
        BB_ASSERT(0);
    }
};


}

