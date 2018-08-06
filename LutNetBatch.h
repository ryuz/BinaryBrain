


#pragma once

#include <array>
#include <vector>
#include "BinaryNetBatch.h"


template <int N=6>
class LutNetBatch : public BinaryNetBatch
{
protected:
	struct LutNode {
		std::array< int, N >		input;
		std::array< bool, (1<<N) >	table;
	};

	int										m_batch_size = 0;
	std::vector< std::vector<LutNode> >		m_lut;
	std::vector<bool*>						m_value;

public:
	LutNetBatch()
	{
	}

	LutNetBatch(std::vector<int> vec_layer_size)
	{
		Setup(vec_layer_size);
	}

	~LutNetBatch()
	{
		for (bool* v : m_value) {
			delete[] v;
		}
	}

	void Setup(std::vector<int> vec_layer_size)
	{
		int layer_num = (int)vec_layer_size.size();
		m_lut.resize(layer_num);
		for (int i = 0; i < layer_num; i++) {
			m_lut[i].resize(vec_layer_size[i]);
		}
	}
	
	int  GetLayerNum(void) const
	{
		return (int)m_lut.size();
	}

	int  GetNodeNum(int layer) const
	{
		return (int)m_lut[layer].size();
	}

	int  GetInputNum(int layer, int node) const
	{
		return N;
	}

	void SetConnection(int layer, int node, int input_index, int input_node)
	{
		m_lut[layer][node].input[input_index] = input_node;
	}

	int GetConnection(int layer, int node, int input_index) const
	{
		return m_lut[layer][node].input[input_index];
	}
	
	void SetLutBit(int layer, int node, int bit, bool value)
	{
		m_lut[layer][node].table[bit] = value;
	}

	bool GetLutBit(int layer, int node, int bit) const
	{
		return m_lut[layer][node].table[bit];
	}

	void SetBatchSize(int batch_size)
	{
		// Šù‘¶ƒƒ‚ƒŠŠJ•ú
		for (bool* v : m_value) {
			delete[] v;
		}

		// ƒƒ‚ƒŠŠm•Û
		m_batch_size = batch_size;
		int layer_num = GetLayerNum();
		m_value.resize(layer_num);
		for (int layer = 0; layer < layer_num; layer++) {
			int node_num = GetNodeNum(layer);
			m_value[layer] = new bool[node_num*batch_size];
		}
	}
	
	int  GetBatchSize(void)
	{
		return m_batch_size;
	}

	bool GetValue(int frame, int layer, int node) const
	{
		return at(frame, layer, node);
	}
	
	void SetValue(int frame, int layer, int node, bool value)
	{
		at(frame, layer, node) = value;
	}

	bool GetInputValue(int frame, int layer, int node, int index) const
	{
		return at(frame, layer - 1, m_lut[layer][node].input[index]);
	}

	void CalcForward(int start_layer = 0)
	{
		int layer_num = GetLayerNum();
		for (int layer = start_layer + 1; layer < layer_num; layer++) {
			int node_num = GetNodeNum(layer);
			for (int node = 0; node < node_num; node++) {
				for (int frame = 0; frame < m_batch_size; frame++) {
					at(frame, layer, node) = m_lut[layer][node].table[GetInputLutIndex(frame, layer, node)];
				}
			}
		}
	}
	
protected:
	bool& at(int frame, int layer, int node) const
	{
		return m_value[layer][node*m_batch_size + frame];
	}
};

