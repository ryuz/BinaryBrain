// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <array>
#include <vector>
#include <intrin.h>
#include "NeuralNetBinaryLut.h"


namespace bb {

// ”Ä—pLUT
template <int N = 6, bool feedback_bitwise = false, typename T = float, typename INDEX = size_t>
class NeuralNetBinaryLutN : public NeuralNetBinaryLut<feedback_bitwise, float, INDEX>
{
protected:
	struct LutNode {
		std::array< bool, (1 << N) >	table;
		std::array< INDEX, N >		input;
	};

	std::vector<LutNode>			m_lut;

public:
	NeuralNetBinaryLutN() {}

	NeuralNetBinaryLutN(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1)
	{
		Resize(input_node_size, output_node_size);
		InitializeCoeff(seed);
	}

	~NeuralNetBinaryLutN() {}


	int   GetLutInputSize(void) const { return N; }
	int   GetLutTableSize(void) const { return (1 << N); }
	void  SetLutInput(INDEX node, int input_index, INDEX input_node) { m_lut[node].input[input_index] = input_node; }
	INDEX GetLutInput(INDEX node, int input_index) const { return m_lut[node].input[input_index]; };
	void  SetLutTable(INDEX node, int bit, bool value) { m_lut[node].table[bit] = value; }
	bool  GetLutTable(INDEX node, int bit) const { return m_lut[node].table[bit]; }

	void Resize(INDEX input_node_size, INDEX output_node_size)
	{
		NeuralNetBinaryLut<feedback_bitwise, float, INDEX>::Resize(input_node_size, output_node_size);
		m_lut.resize(m_output_node_size);
	}

	void Update(void)
	{
	}
};


}