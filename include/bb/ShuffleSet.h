// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <vector>
#include <list>
#include <random>
#include <algorithm>


namespace bb {


// なるべく重複しないようにランダムにインデックスをシャッフルする
// トランプのカードを配るイメージで、手持ちが無くなれば再充填することで、
// 特定の値がずっと出なかったり、同じものが出続けることを防止する

// シャッフルクラス
template <typename INDEX>
class ShuffleSet
{
protected:
    std::mt19937_64     m_mt;
    std::list<INDEX>    m_heap;
    std::list<INDEX>    m_reserve;

public:
    ShuffleSet()
    {
    }

    ShuffleSet(INDEX size, std::uint64_t seed = 1)
    {
        Setup(size, seed);
    }

    void Setup(INDEX size, std::uint64_t seed = 1)
    {
        // 初期化
        m_mt.seed(seed);
        m_heap.clear();
        m_reserve.clear();

        // シャッフル
        std::vector<INDEX> heap(size);
        for (INDEX i = 0; i < size; i++) {
            heap[i] = i;
        }
        std::shuffle(heap.begin(), heap.end(), m_mt);

        // 設定
        m_heap.assign(heap.begin(), heap.end());
    }
    
    std::vector<INDEX> GetRandomSet(INDEX n)
    {
        std::vector<INDEX>  set;
        std::vector<INDEX>  stash;

        // 指定個数取り出す
        for (INDEX i = 0; i < n; i++) {
            if (m_heap.empty()) {
                // 一通り割り当てたら利用済みを再利用
                std::vector<INDEX> heap(m_reserve.size());
                heap.assign(m_reserve.begin(), m_reserve.end());
                std::shuffle(heap.begin(), heap.end(), m_mt);
                m_heap.assign(heap.begin(), heap.end());
                m_reserve.clear();
            }
            
            // reserveで不足する場合は stash から回す
            if (m_heap.empty()) {
                std::vector<INDEX> heap(stash.size());
                heap.assign(stash.begin(), stash.end());
                std::shuffle(heap.begin(), heap.end(), m_mt);
                m_heap.assign(heap.begin(), heap.end());
                stash.clear();
            }

            // 先頭から取り出す
            auto item = m_heap.front();
            set.push_back(item);

            // 使ったものはstashに移す
            m_heap.pop_front();
            stash.push_back(item);
        }

        // stashに残っているものはリザーブに回す
        for (auto item : stash) {
            m_reserve.push_back(item);
        }

        return set;
    }
};


}
