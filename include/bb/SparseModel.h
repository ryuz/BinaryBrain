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

#include "bb/Model.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"


namespace bb {


// 入力接続数に制限のあるネット
class SparseModel : public Model
{
public:
    //ノードの 疎結合の管理
    virtual index_t GetNodeConnectionSize(index_t output_node) const = 0;
    virtual void    SetNodeConnectionIndex(index_t output_node, index_t connection, index_t input_node) = 0;
    virtual index_t GetNodeConnectionIndex(index_t output_node, index_t connection) const = 0;
    
    index_t GetConnectionSize(indices_t output_indices) const
    {
        return GetNodeConnectionSize(ConvertIndicesToIndex(output_indices, this->GetOutputShape()));
    }

    void SetConnectionIndex(indices_t output_indices, index_t connection, index_t input_indices)
    {
        return SetNodeConnectionIndex(ConvertIndicesToIndex(output_indices, this->GetOutputShape()), connection, input_indices);
    }

    void SetConnectionIndices(indices_t output_indices, index_t connection, indices_t input_indices)
    {
        return SetNodeConnectionIndex(ConvertIndicesToIndex(output_indices, this->GetOutputShape()), connection, ConvertIndicesToIndex(input_indices, this->GetInputShape()));
    }

    index_t GetConnectionIndex(indices_t output_indices, index_t connection) const
    {
        return GetNodeConnectionIndex(ConvertIndicesToIndex(output_indices, this->GetOutputShape()), connection);
    }

    indices_t GetConnectionIndices(indices_t output_indices, index_t connection) const
    {
        index_t input_node = GetNodeConnectionIndex(ConvertIndicesToIndex(output_indices, this->GetOutputShape()), connection);
        return ConvertIndexToIndices(input_node, this->GetInputShape());
    }
    

    // LUTに見立て場合の値取得
    virtual bool GetLutTable(index_t node, int bitpos) const
    {
        BB_DEBUG_ASSERT(node   >= 0 && node <= this->GetOutputNodeSize());
        BB_DEBUG_ASSERT(bitpos >= 0 && bitpos <= this->GetNodeConnectionSize(node));

        auto connection_size = this->GetNodeConnectionSize(node);

        // 係数をバイナリ化
        std::vector<double> vec(connection_size);
        for (int bit = 0; bit < connection_size; ++bit) {
            vec[bit] = (bitpos & (1 << bit)) ? 1.0 : 0.0;
        }
        auto v = this->ForwardNode(node, vec);
        return (v[0] >= 0.5);
    }

    virtual int GetLutTableSize(index_t node) const
    {
        return (int)(1 << this->GetNodeConnectionSize(node));
    }


protected:

    void InitializeNodeInput(std::uint64_t seed, std::string connection = "")
    {
        auto input_shape  = this->GetInputShape();
        auto output_shape = this->GetOutputShape();

        auto input_node_size  = CalcShapeSize(input_shape);
        auto output_node_size = CalcShapeSize(output_shape);

        auto argv = SplitString(connection);

        if (argv.size() > 0 && argv[0] == "pointwise") {
            BB_ASSERT(input_shape.size() == 3);
            BB_ASSERT(output_shape.size() == 3);
            BB_ASSERT(input_shape[0] == output_shape[0]);
            BB_ASSERT(input_shape[1] == output_shape[1]);
            std::mt19937_64 mt(seed);
            for (index_t y = 0; y < output_shape[1]; ++y) {
                for (index_t x = 0; x < output_shape[0]; ++x) {
                    // 接続先をシャッフル
                    ShuffleSet<index_t> ss(input_shape[2], mt());
                    for (index_t c = 0; c < output_shape[2]; ++c) {
                        // 入力をランダム接続
                        index_t  connection_size = GetConnectionSize({x, y, c});
                        auto random_set = ss.GetRandomSet(connection_size);
                        for (index_t i = 0; i < connection_size; ++i) {
                            SetConnectionIndices({x, y, c}, i, {x, y, random_set[i]});
                        }
                    }
                }
            }
            return;
        }

        if (argv.size() > 0 && argv[0] == "depthwise") {
            BB_ASSERT(input_shape.size() == 3);
            BB_ASSERT(output_shape.size() == 3);
            BB_ASSERT(input_shape[2] == output_shape[2]);
            std::mt19937_64 mt(seed);
            for (index_t c = 0; c < output_shape[2]; ++c) {
                // 接続先をシャッフル
                ShuffleSet<index_t> ss(input_shape[0] * input_shape[1], mt());
                for (index_t y = 0; y < output_shape[1]; ++y) {
                    for (index_t x = 0; x < output_shape[0]; ++x) {
                        // 入力をランダム接続
                        index_t  connection_size = GetConnectionSize({x, y, c});
                        auto random_set = ss.GetRandomSet(connection_size);
                        for (index_t i = 0; i < connection_size; ++i) {
                            index_t iy = random_set[i] / input_shape[0];
                            index_t ix = random_set[i] % input_shape[0];

                            index_t output_node = ConvertIndicesToIndex({x, y, c}, output_shape);
                            index_t input_node  = ConvertIndicesToIndex({ix, iy, c}, input_shape);

                            BB_ASSERT(output_node >= 0 && output_node < output_node_size);
                            BB_ASSERT(input_node  >= 0 && input_node  < input_node_size);
                            SetNodeConnectionIndex(output_node, i, input_node);
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
                auto m = GetNodeConnectionSize(output_node);
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
                            SetNodeConnectionIndex(output_node, i, input_node);
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
                index_t m = GetNodeConnectionSize(output_node);
                for ( index_t i = 0; i < m; ++i ) {
                    SetNodeConnectionIndex(output_node, i, input_node % input_node_size);
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
                index_t  connection_size = GetNodeConnectionSize(node);
                auto random_set = ss.GetRandomSet(connection_size);
                for (index_t i = 0; i < connection_size; ++i) {
                    SetNodeConnectionIndex(node, i, random_set[i]);
                }
            }
            return;
        }

        std::cout << "unknown connection rule : \"" << argv[0] <<  "\"" << std::endl;
        BB_ASSERT(0);
    }
};


}

