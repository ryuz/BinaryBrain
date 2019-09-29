// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <cmath>
#include <array>
#include <vector>

#include "bb/SparseLayer.h"
#include "bb/StochasticLutN.h"

namespace bb {


// LUT方式基底クラス
template <typename FT = Bit, typename BT = float>
class LutLayer : public SparseLayer
{
public:
    // LUT操作の定義
    virtual int   GetLutTableSize(index_t node) const = 0;
    virtual void  SetLutTable(index_t node, int bitpos, bool value) = 0;
    virtual bool  GetLutTable(index_t node, int bitpos) const = 0;

    /*
    virtual bool  GetLutInput(index_t frame, index_t node, int bitpos) const = 0;
    virtual int   GetLutInputIndex(index_t frame, index_t node) const
    {
        int index = 0;
        int lut_table_size = GetLutTableSize(node);
        for (int bitpos = 0; bitpos < lut_table_size; ++bitpos) {
            index |= (GetLutInput(frame, node, bitpos) ? (1 << bitpos) : 0);
        }
        return index;
    }
    */

protected:
    void InitializeLutTable(std::uint64_t seed)
    {
        std::mt19937_64                     mt(seed);
        std::uniform_int_distribution<int>  rand(0, 1);
        
        index_t node_size = GetShapeSize(this->GetOutputShape());

        // LUTテーブルをランダムに初期化
        for ( index_t node = 0; node < node_size; ++node) {
            int lut_table_size = GetLutTableSize(node);
            for (int i = 0; i < lut_table_size; i++) {
                this->SetLutTable(node, i, rand(mt) != 0);
            }
        }
    }
    
public:
    // 形状が同一のSparceLayerをテーブル化して取り込む
    void ImportLayer(std::shared_ptr< SparseLayer > src)
    {
        BB_ASSERT(GetShapeSize(src->GetInputShape())  == GetShapeSize(this->GetInputShape()));
        BB_ASSERT(GetShapeSize(src->GetOutputShape()) == GetShapeSize(this->GetOutputShape()));
        
        auto node_size  = GetShapeSize(this->GetOutputShape());

        for (index_t node = 0; node < node_size; ++node) {
            auto connection_size = this->GetNodeConnectionSize(node);
            auto table_size = this->GetLutTableSize(node);
            
            BB_ASSERT(src->GetNodeConnectionSize(node) == connection_size);
            
            // 入力をコピー
            for (int input_index = 0; input_index < connection_size; ++input_index) {
                this->SetNodeConnectionIndex(node, input_index, src->GetNodeConnectionIndex(node, input_index));
            }

            // 係数をバイナリ化
            std::vector<double> vec(connection_size);
            for (int index = 0; index < table_size; ++index) {
                for (int bit = 0; bit < connection_size; ++bit) {
                    vec[bit] = (index & (1 << bit)) ? 1.0 : 0.0;
                }
                auto v = src->ForwardNode(node, vec);
                this->SetLutTable(node, index, (v[0] >= 0.5));
            }
        }
    }

    // 形状が同一のSparceLayerをテーブル化して取り込む
    template <class T>
    void Import(std::shared_ptr<T> src)
    {
        BB_ASSERT(GetShapeSize(src->GetInputShape())  == GetShapeSize(this->GetInputShape()));
        BB_ASSERT(GetShapeSize(src->GetOutputShape()) == GetShapeSize(this->GetOutputShape()));
        
        auto node_size  = GetShapeSize(this->GetOutputShape());

        auto input_index_ptr = src->lock_InputIndex_const();
        auto W_ptr           = src->lock_W_const();

        for (index_t node = 0; node < node_size; ++node) {
            auto input_size = this->GetNodeInputSize(node);
            auto table_size = this->GetLutTableSize(node);
            
//            BB_ASSERT(input_size == N);
//            BB_ASSERT(table_size == (1 << N));
            
            // 入力をコピー
            for (int input_index = 0; input_index < input_size; ++input_index) {
                this->SetNodeInput(node, input_index, input_index_ptr(node, input_index));
            }

            // 係数をコピー
            for (int index = 0; index < table_size; ++index) {
                this->SetLutTable(node, index, (W_ptr(node, index) >= (T)0.5));
            }
        }
    }
};


}

// end of file
