


#pragma once

#include <array>
#include "BinaryNet.h"

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
		int index = GetIndex();
		m_output = (m_lut[index] != m_reverse);
	}

	void SetReverse(bool reverse)
	{
		m_reverse = reverse;
	}

	void ResetScore(void)
	{
		for (int i = 0; i < (1 << N); i++) {
			m_score_val[i] = 0;
			m_score_n[i] = 0;
		}
	}

	void AddScore(double score)
	{
		int index = GetIndex();
		m_score_val[index] += score;
		m_score_n[index]++;
	}

	void Update(std::mt19937& mt)
	{
		std::uniform_int_distribution<int>	distribution(-20, 0);
		int th = distribution(mt);

		for (int i = 0; i < (1 << N); i++) {
			double score = m_score_val[i] / m_score_n[i];
			if ( score*10 < th ) {
				m_lut[i] = !m_lut[i];
			}
		}
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
	bool										m_reverse = false;

	std::array< double, (1 << N) >				m_score_val;
	std::array< int,    (1 << N) >				m_score_n;
};



template <int N = 6>
class LutNet : public BinaryNet
{
public:
	LutNet(std::vector<int> layer_num)
	{
		// LUTç\íz
		m_lut.resize(layer_num.size());
		auto n = layer_num.begin();
		for ( auto& l : m_lut ) {
			l.resize(*n++);
		}

		Reset();
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

	// åvéZ
	void CalcForward(int start_layer = 0)
	{
		for (int i = 1; i < (int)m_lut.size(); i++) {
			int len = (int)m_lut[i].size();
// #pragma omp parallel for 
			for (int j = 0; j < len; j++) {
				m_lut[i][j].CalcForward();
			}
		}

		n++;
		auto it = count.begin();
		for (auto& o : Output()) {
			if ( o.GetValue() ) {
				(*it)++;
			}
			++it;
		}
	}
	
	// ê⁄ë±
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


	/////////////////////


	int					n;
	std::vector<int>	count;
	void Reset(void) {
		n = 0;
		count.resize(Output().size(), 0);
		for (auto& c : count) {
			c = 0;
		}
	}

	double GetScore(int exp)
	{
		auto& out = Output();
		double score = 0;
		for (int i = 0; i < (int)out.size(); i++) {
			if (i == exp) {
				score += out[i].GetValue() ? +20.0 : -20.0;
			}
			else {
				score += out[i].GetValue() ? -1.0 : +1.0;
			}
		}
		return score;
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
	
	std::vector<int> MakeInitVec(size_t n)
	{
		std::vector<int>	vec(n);
		for (int i = 0; i < n; i++) {
			vec[i] = i;
		}
		return vec;
	}

	void ShuffleVec(std::mt19937& mt, std::vector<int>& vec, int n = LUT_SIZE)
	{
		std::uniform_int_distribution<int>	distribution(0, vec.size() - 1);

		for (int i = 0; i < n; i++) {
			std::swap(vec[i], vec[distribution(mt)]);
		}
	}
	

	std::vector< std::vector< LutModel<N> > >	m_lut;
};

