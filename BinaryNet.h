


#pragma once

#include <vector>



class BinaryNet
{
public:
	virtual ~BinaryNet() {}

	virtual int  GetLayerNum(void) const = 0;
	virtual int  GetNodeNum(int layer) const = 0;
	virtual int  GetInputNum(int layer, int node) const = 0;

	virtual void SetConnection(int layer, int node, int input_num, int input_node) = 0;

	virtual void CalcForward(int layer = 0) = 0;

	virtual bool GetValue(int layer, int node) const = 0;
	virtual void SetValue(int layer, int node, bool value) = 0;

	virtual bool GetInputValue(int layer, int node, int index) const = 0;

	virtual bool GetLutBit(int layer, int node, int bit) const = 0;
	virtual void SetLutBit(int layer, int node, int bit, bool value) = 0;

	virtual void InvertLut(int layer, int node)
	{
		int n = GetInputNum(layer, node);
		int size = (1 << n);
		for (int i = 0; i < size; i++) {
			SetLutBit(layer, node, i, !GetLutBit(layer, node, i));
		}
		SetValue(layer, node, !GetValue(layer, node));
	}

	void SetInput(std::vector<bool> input_vector)
	{
		for ( int i = 0; i < (int)input_vector.size(); i++ ) {
			SetValue(0, i, input_vector[i]);
		}
	}

	std::vector<bool> GetOutput(void)
	{
		int layer = GetLayerNum() - 1;
		std::vector<bool> output_vector(GetNodeNum(layer));
		for (int i = 0; i < (int)output_vector.size(); i++) {
			output_vector[i] = GetValue(layer, i);
		}
		return output_vector;
	}

	int  GetInputLutIndex(int layer, int node) const
	{
		int num = GetInputNum(layer, node);
		
		int idx = 0;
		int bit = 1;
		for (int i = 0; i < num; i++) {
			idx |= GetInputValue(layer, node, i) ? bit : 0;
			bit <<= 1;
		}

		return idx;
	}

protected:
	std::vector<int>	m_OutputConter;
};

