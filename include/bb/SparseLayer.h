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
	void InitializeNodeInput(index_t output_node_size, index_t input_node_size, std::uint64_t seed)
	{
		std::mt19937_64                     mt(seed);
		std::uniform_int_distribution<int>	distribution(0, 1);
		
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

