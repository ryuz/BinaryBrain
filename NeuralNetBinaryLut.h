


#pragma once

#include <array>
#include <vector>
#include <intrin.h>
#include <omp.h>
#include <ppl.h>
#include "NeuralNetLayer.h"
#include "NeuralNetBufferAccessorBinary.h"
#include "ShuffleSet.h"


// LUT方式基底クラス
template <typename INDEX = size_t>
class NeuralNetBinaryLut : public NeuralNetLayer<INDEX>
{
protected:
	INDEX					m_mux_size;
	INDEX					m_frame_size;
	INDEX					m_input_node_size;
	INDEX					m_output_node_size;

	const void*				m_inputValue;
	void*					m_outputValue;
	void*					m_inputError;
	const void*				m_outputError;

public:
	// LUT操作の定義
	virtual int   GetLutInputSize(void) const = 0;
	virtual int   GetLutTableSize(void) const = 0;
	virtual void  SetLutInput(INDEX node, int input_index, INDEX input_node) = 0;
	virtual INDEX GetLutInput(INDEX node, int input_index) const = 0;
	virtual void  SetLutTable(INDEX node, int bit, bool value) = 0;
	virtual bool  GetLutTable(INDEX node, int bit) const = 0;

	void InitializeLut(std::uint64_t seed)
	{
		std::mt19937_64                     mt(seed);
		std::uniform_int_distribution<int>	rand(0, 1);

		INDEX node_size = GetOutputNodeSize();
		int   lut_input_size = GetLutInputSize();
		int   lut_table_size = GetLutTableSize();

		ShuffleSet	ss(GetInputNodeSize(), mt());
		for (INDEX node = 0; node < node_size; ++node ) {
			// 入力をランダム接続
			auto random_set = ss.GetRandomSet(GetLutInputSize());
			for (int i = 0; i < lut_input_size; ++i) {
				SetLutInput(node, i, random_set[i]);
			}

			// LUTテーブルをランダムに初期化
			for (int i = 0; i < lut_table_size; i++) {
				SetLutTable(node, i, rand(mt) != 0);
			}
		}
	}
	
	
	// 共通機能の定義
protected:
	void SetupBase(INDEX input_node_size, INDEX output_node_size, INDEX mux_size, INDEX batch_size = 1, std::uint64_t seed = 1)
	{
		m_input_node_size = input_node_size;
		m_output_node_size = output_node_size;
		m_mux_size = mux_size;
		m_frame_size = batch_size * mux_size;
	}

public:
	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size * m_mux_size; }

	void  SetInputValuePtr(const void* inputValue) { m_inputValue = inputValue; }
	void  SetOutputValuePtr(void* outputValue) { m_outputValue = outputValue; }
	void  SetOutputErrorPtr(const void* outputError) { m_outputError = outputError; }
	void  SetInputErrorPtr(void* inputError) { m_inputError = inputError; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_input_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_output_node_size; }

	int   GetInputValueBitSize(void) const { return 1; }
	int   GetInputErrorBitSize(void) const { return 1; }
	int   GetOutputValueBitSize(void) const { return 1; }
	int   GetOutputErrorBitSize(void) const { return 1; }


	void Forward(void)
	{
		INDEX node_size      = GetOutputNodeSize();
		int   lut_input_size = GetLutInputSize();
		concurrency::parallel_for<INDEX>(0, node_size, [&](INDEX node)
		{
			NeuralNetBufferAccessorBinary<float, INDEX>	acc_in((void*)m_inputValue,   m_frame_size);
			NeuralNetBufferAccessorBinary<float, INDEX>	acc_out((void*)m_outputValue, m_frame_size);
			for (INDEX frame = 0; frame < m_frame_size; ++frame) {
				int bit = 0;
				int msk = 1;
				for (int i = 0; i < lut_input_size; i++) {
					INDEX input_node = GetLutInput(node, i);
					bool input_value = acc_in.Get(frame, input_node);
					bit |= input_value ? msk : 0;
					msk <<= 1;
				}
				bool output_value = GetLutTable(node, bit);
				acc_out.Set(frame, node, output_value);
			}
		});
	}

	void Backward(void)
	{
	}

	void Update(double learning_rate)
	{
	}
};

