

#pragma once

#include <vector>
#include <list>
#include <random>
#include <algorithm>


class ShuffleSet
{
public:
	ShuffleSet()
	{
	}
	
	ShuffleSet(int size, std::uint64_t seed = 1)
	{
		Setup(size, seed);
	}
	
	void Setup(int size, std::uint64_t seed = 1)
	{
		// 初期化
		m_mt.seed(seed);
		m_heap.clear();
		m_reserve.clear();

		// シャッフル
		std::vector<int> heap(size);
		for (int i = 0; i < size; i++) {
			heap[i] = i;
		}
		std::shuffle(heap.begin(), heap.end(), m_mt);

		// 設定
		m_heap.assign(heap.begin(), heap.end());
	}

	std::vector<int> GetSet(int n)
	{
		std::vector<int>	set;

		// 指定個数取り出す
		for (int i = 0; i < n; i++) {
			if (m_heap.empty()) {
				// 一通り割り当てたら利用済みを再利用
				std::vector<int> heap(m_reserve.size());
				heap.assign(m_reserve.begin(), m_reserve.end());
				std::shuffle(heap.begin(), heap.end(), m_mt);
				m_heap.assign(heap.begin(), heap.end());
				m_reserve.clear();
			}

			// 使ったものは取り外す
			set.push_back(m_heap.front());
			m_heap.pop_front();
		}

		// 使ったものはリザーブに回す
		for (auto s : set) {
			m_reserve.push_back(s);
		}

		return set;
	}

protected:
	std::mt19937_64	m_mt;	
	std::list<int>	m_heap;
	std::list<int>	m_reserve;
};


