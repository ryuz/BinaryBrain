


#pragma once

#include <array>
#include "BinaryNet.h"


// 汎用LUTモデルの定義
template <int N=6>
class LutModel
{
public:
	LutModel()
	{
	}

	~LutModel()
	{
	}

	bool &operator[](size_t i)
	{
		return m_lut[i];
	}

	bool operator[](size_t i) const
	{
		return m_lut[i];
	}

	void SetConnection(int index, const LutModel* lut)
	{
		m_input[index] = lut;
	}

	bool GetValue(void) const
	{
		return m_output;
	}

	void SetValue(bool val)
	{
		m_output = val;
	}
	
	void CalcForward(void)
	{
		m_output = m_lut[GetIndex()];
	}
	
	bool GetInputValue(int index) const
	{
		return m_input[index]->GetValue();
	}
	
	int GetIndex(void) const
	{
		int index = 0;
		for (int i = 0; i < N; i++) {
			index >>= 1;
			index |= m_input[i]->GetValue() ? (1 << (N - 1)) : 0;
		}
		return index;
	}

protected:
	std::array< bool, (1<<N) >					m_lut;
	std::array< const LutModel*, N >			m_input;
	bool										m_output;
};



template <int N = 6>
class LutNet : public BinaryNet
{
public:
	LutNet(std::vector<int> layer_num)
	{
		// LUT構築
		m_lut.resize(layer_num.size());
		auto n = layer_num.begin();
		for ( auto& l : m_lut ) {
			l.resize(*n++);
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

	// 計算
	void CalcForward(int start_layer = 0)
	{
		for (int i = 1; i < (int)m_lut.size(); i++) {
			int len = (int)m_lut[i].size();
// #pragma omp parallel for 
			for (int j = 0; j < len; j++) {
				m_lut[i][j].CalcForward();
			}
		}
	}
	
	// 接続
	void SetConnection(int layer, int node, int input_num, int input_node)
	{
		m_lut[layer][node].SetConnection(input_num, &m_lut[layer - 1][input_node]);
	}

	bool GetValue(int layer, int node) const
	{
		return m_lut[layer][node].GetValue();
	}

	void SetValue(int layer, int node, bool value)
	{
		m_lut[layer][node].SetValue(value);
	}
	
	bool GetInputValue(int layer, int node, int index) const
	{
		return m_lut[layer][node].GetInputValue(index);
	}

	bool GetLutBit(int layer, int node, int bit) const
	{
		return m_lut[layer][node][bit];
	}
	
	void SetLutBit(int layer, int node, int bit, bool value)
	{
		m_lut[layer][node][bit] = value;
	}


	std::vector< LutModel<N> >& operator[](size_t i) {
		return m_lut[i];
	}

	std::vector< LutModel<N> >& Input(void) {
		return m_lut[0];
	}

	std::vector< LutModel<N> >& Output(void) {
		return m_lut[m_lut.size()-1];
	}


protected:
	std::vector< std::vector< LutModel<N> > >	m_lut;
};

