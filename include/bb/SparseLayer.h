// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                 Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                 https://github.com/ryuz
//                                 ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include "bb/Layer.h"
#include "bb/ShuffleSet.h"


namespace bb {


// 入力接続数に制限のあるネット
template <typename FT = float, typename BT = float>
class SparseLayer : public Layer<FT, BT>
{
public:
	//ノードの 疎結合の管理
	virtual index_t GetNodeInputSize(index_t node) const = 0;
	virtual void    SetNodeInput(index_t node, index_t input_index, index_t input_node) = 0;
	virtual index_t GetNodeInput(index_t node, index_t input_index) const = 0;
	
protected:
	void InitializeNodeInput(std::uint64_t seed)
	{
		std::mt19937_64                     mt(seed);
		std::uniform_int_distribution<int>	distribution(0, 1);
		
        auto input_shape  = this->GetInputShape();
        auto output_shape = this->GetOutputShape();

        auto input_node_size  = GetShapeSize(input_shape);
        auto output_node_size = GetShapeSize(output_shape);

        if ( input_shape.size() == 3 && input_shape[2] > 3) {
            index_t c = input_shape[2];
            index_t h = input_shape[1];
            index_t w = input_shape[0];
            indices_t offset_shape({w, h, 3});

    		ShuffleSet<index_t>	ss(3*h*w, seed);
            indices_t idx({0, 0, 0});
            for (index_t node = 0; node < output_node_size; ++node) {
    			index_t  input_size = GetNodeInputSize(node);

    			auto random_set = ss.GetRandomSet(input_size);
			    for (index_t i = 0; i < input_size; ++i) {
                    indices_t offset_idx = GetShapeIndices(random_set[i], offset_shape);
                    indices_t input_idx(3);
                    input_idx[2] = (idx[2] + offset_idx[2]) % c;
                    input_idx[1] = (idx[1] + offset_idx[1]) % h;
                    input_idx[0] = (idx[0] + offset_idx[0]) % w;
				    SetNodeInput(node, i, GetShapeIndex(input_idx, input_shape));
    			}

                GetNextIndices(idx, input_shape);
            }
            return;
        }

		// 接続先をシャッフル
		ShuffleSet<index_t>	ss(input_node_size, seed);
		
		for (index_t node = 0; node < output_node_size; ++node) {
			// 入力をランダム接続
			index_t  input_size = GetNodeInputSize(node);
			auto random_set = ss.GetRandomSet(input_size);
			for (index_t i = 0; i < input_size; ++i) {
				SetNodeInput(node, i, random_set[i]);
			}
		}
	}
};


}

