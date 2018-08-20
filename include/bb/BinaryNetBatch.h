


#pragma once


#include <vector>
#include "BinaryNetData.h"


// LUT想定のバイナリネットの抽象クラス
class BinaryNetBatch
{
public:
	virtual ~BinaryNetBatch() {}

	virtual	void Setup(std::vector<int> vec_layer_size) = 0;
	virtual int  GetLayerNum(void) const = 0;
	virtual int  GetNodeNum(int layer) const = 0;
	virtual int  GetInputNum(int layer, int node) const = 0;

	virtual void SetConnection(int layer, int node, int input_index, int input_node) = 0;
	virtual int  GetConnection(int layer, int node, int input_index) const = 0;

	virtual bool GetLutBit(int layer, int node, int bit) const = 0;
	virtual void SetLutBit(int layer, int node, int bit, bool value) = 0;

	virtual	void SetBatchSize(int batch_size) = 0;
	virtual int  GetBatchSize(void) = 0;

	virtual void CalcForward(int layer = 0) = 0;

	virtual bool GetValue(int frame, int layer, int node) const = 0;
	virtual void SetValue(int frame, int layer, int node, bool value) = 0;

	virtual bool GetInputValue(int frame, int layer, int node, int index) const = 0;

	virtual void InvertLut(int layer, int node)
	{
		int n = GetInputNum(layer, node);
		int lut_size = (1 << n);
		for (int i = 0; i < lut_size; i++) {
			SetLutBit(layer, node, i, !GetLutBit(layer, node, i));
		}

		int batch_size = GetBatchSize();
		for (int i = 0; i < batch_size; i++) {
			SetValue(i, layer, node, !GetValue(i, layer, node));
		}
	}

	void SetInput(int frame, std::vector<bool> input_vector)
	{
		for ( int i = 0; i < (int)input_vector.size(); i++ ) {
			SetValue(frame, 0, i, input_vector[i]);
		}
	}

	std::vector<bool> GetInput(int frame)
	{
		std::vector<bool> input_vector;
		
		int node_num = GetNodeNum(0);
		input_vector.reserve(node_num);
		for (int i = 0; i < node_num; i++) {
			input_vector.push_back(GetValue(frame, 0, i));
		}
		return input_vector;
	}

	std::vector<bool> GetOutput(int frame)
	{
		int layer = GetLayerNum() - 1;
		std::vector<bool> output_vector(GetNodeNum(layer));
		for (int i = 0; i < (int)output_vector.size(); i++) {
			output_vector[i] = GetValue(frame, layer, i);
		}
		return output_vector;
	}

	int  GetInputLutIndex(int frame, int layer, int node) const
	{
		int num = GetInputNum(layer, node);
		
		int idx = 0;
		int bit = 1;
		for (int i = 0; i < num; i++) {
			idx |= GetInputValue(frame, layer, node, i) ? bit : 0;
			bit <<= 1;
		}

		return idx;
	}

	virtual void CalcForwardNode(int layer, int node)
	{
		int batch_size = GetBatchSize();
		for (int frame = 0; frame < batch_size; frame++) {
			SetValue(frame, layer, node, GetLutBit(layer, node, GetInputLutIndex(frame, layer, node)));
		}
	}


	// データエクスポート
	BinaryNetData ExportData(void) {
		BinaryNetData	bnd;
		for (int layer = 0; layer < GetLayerNum(); layer++) {
			if (layer == 0) {
				bnd.input_num = GetNodeNum(layer);
			}
			else {
				int node_num = GetNodeNum(layer);
				BinaryLayerData bld;
				bld.node.reserve(node_num);
				for (int node = 0; node < node_num; node++) {
					LutData ld;
					int num = GetInputNum(layer, node);
					ld.connect.reserve(num);
					for (int i = 0; i < num; i++) {
						ld.connect.push_back(GetConnection(layer, node, i));
					}
					ld.lut.reserve((size_t)1<<num);
					for (int i = 0; i < (1<<num); i++) {
						ld.lut.push_back(GetLutBit(layer, node, i));
					}
					bld.node.push_back(ld);
				}
				bnd.layer.push_back(bld);
			}
		}

		return bnd;
	}
};

