// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>
#include <random>

#include "bb/NeuralNetSparseLayer.h"
#include "bb/NeuralNetLutPre.h"
#include "bb/NeuralNetLutPost.h"
#include "bb/NeuralNetBatchNormalization.h"
#include "bb/NeuralNetReLU.h"
#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetBinarize.h"


namespace bb {


// 入力数制限Affine Binary Connect版
template <int N = 6, int M = 64, typename T = float, typename INDEX = size_t>
class NeuralNetLut : public NeuralNetSparseLayer<T, INDEX>
{
protected:
	INDEX									m_batch_size = 0;

	// 3層で構成
	NeuralNetLutPre<N, M, T, INDEX>			m_lut_pre;
//	NeuralNetReLU<T, INDEX>					m_act_pre;
	NeuralNetSigmoid<T, INDEX>				m_act_pre;
	NeuralNetLutPost<M, T, INDEX>			m_lut_post;
	NeuralNetBatchNormalization<T, INDEX>	m_batch_norm;
	NeuralNetSigmoid<T, INDEX>				m_act_post;

public:
	NeuralNetLut() {}

	NeuralNetLut(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1,
		const NeuralNetOptimizer<T, INDEX>* optimizer = nullptr)
		: m_lut_pre(input_node_size, output_node_size, seed, optimizer),
		m_act_pre(output_node_size * M),
		m_lut_post(output_node_size, seed, optimizer),
		m_batch_norm(output_node_size, optimizer),
		m_act_post(output_node_size)
	{
	}
	
	~NeuralNetLut() {}

	std::string GetClassName(void) const { return "NeuralNetLut"; }

	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		auto vec0 = m_lut_pre.CalcNode(node, input_value);
		auto vec1 = m_act_pre.CalcNode(node, vec0);
		auto vec2 = m_lut_post.CalcNode(node, vec1);
		auto vec3 = m_batch_norm.CalcNode(node, vec2);
		auto vec4 = m_act_post.CalcNode(node, vec3);
		return vec4;
	}


	void InitializeCoeff(std::uint64_t seed)
	{
		std::mt19937_64 mt(seed);
		m_lut_pre.InitializeCoeff(mt());
		m_lut_post.InitializeCoeff(mt());
	}

	void SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		m_lut_pre.SetOptimizer(optimizer);
		m_act_pre.SetOptimizer(optimizer);
		m_lut_post.SetOptimizer(optimizer);
		m_batch_norm.SetOptimizer(optimizer);
		m_act_post.SetOptimizer(optimizer);
	}

	void SetBinaryMode(bool enable)
	{
		m_lut_pre.SetBinaryMode(enable);
//		m_act_pre.SetBinaryMode(enable);
		m_lut_post.SetBinaryMode(enable);
//		m_batch_norm.SetBinaryMode(enable);
		m_act_post.SetBinaryMode(enable);
	}

	int   GetNodeInputSize(INDEX node) const { return m_lut_pre.GetNodeInputSize(node); }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { m_lut_pre.SetNodeInput(node, input_index, input_node); }
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_lut_pre.GetNodeInput(node, input_index); }

	void  SetBatchSize(INDEX batch_size)
	{
		m_lut_pre.SetBatchSize(batch_size);
		m_act_pre.SetBatchSize(batch_size);
		m_lut_post.SetBatchSize(batch_size);
		m_batch_norm.SetBatchSize(batch_size);
		m_act_post.SetBatchSize(batch_size);

		if (batch_size == m_batch_size) {
			return;
		}
		m_batch_size = batch_size;

		CheckConnection(m_lut_pre, m_act_pre);
		CheckConnection(m_act_pre, m_lut_post);
		CheckConnection(m_lut_post, m_batch_norm);
		CheckConnection(m_batch_norm, m_act_post);

		m_lut_pre.SetOutputSignalBuffer(m_lut_pre.CreateOutputSignalBuffer());
		m_lut_pre.SetOutputErrorBuffer(m_lut_pre.CreateOutputErrorBuffer());
		m_act_pre.SetInputSignalBuffer(m_lut_pre.GetOutputSignalBuffer());
		m_act_pre.SetInputErrorBuffer(m_lut_pre.GetOutputErrorBuffer());

		m_act_pre.SetOutputSignalBuffer(m_act_pre.CreateOutputSignalBuffer());
		m_act_pre.SetOutputErrorBuffer(m_act_pre.CreateOutputErrorBuffer());
		m_lut_post.SetInputSignalBuffer(m_act_pre.GetOutputSignalBuffer());
		m_lut_post.SetInputErrorBuffer(m_act_pre.GetOutputErrorBuffer());

		m_lut_post.SetOutputSignalBuffer(m_lut_post.CreateOutputSignalBuffer());
		m_lut_post.SetOutputErrorBuffer(m_lut_post.CreateOutputErrorBuffer());
		m_batch_norm.SetInputSignalBuffer(m_lut_post.GetOutputSignalBuffer());
		m_batch_norm.SetInputErrorBuffer(m_lut_post.GetOutputErrorBuffer());

		m_batch_norm.SetOutputSignalBuffer(m_batch_norm.CreateOutputSignalBuffer());
		m_batch_norm.SetOutputErrorBuffer(m_batch_norm.CreateOutputErrorBuffer());
		m_act_post.SetInputSignalBuffer(m_batch_norm.GetOutputSignalBuffer());
		m_act_post.SetInputErrorBuffer(m_batch_norm.GetOutputErrorBuffer());
	}

	
	// 入出力バッファ
	void  SetInputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_lut_pre.SetInputSignalBuffer(buffer); }
	void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_lut_pre.SetInputErrorBuffer(buffer); }
	void  SetOutputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_act_post.SetOutputSignalBuffer(buffer); }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_act_post.SetOutputErrorBuffer(buffer); }

	const NeuralNetBuffer<T, INDEX>& GetInputSignalBuffer(void) const { return m_lut_pre.GetInputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) const { return m_lut_pre.GetInputErrorBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputSignalBuffer(void) const { return m_act_post.GetOutputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) const { return m_act_post.GetOutputErrorBuffer(); }


	INDEX GetInputFrameSize(void) const { return m_lut_pre.GetInputFrameSize(); }
	INDEX GetInputNodeSize(void) const { return m_lut_pre.GetInputNodeSize(); }

	INDEX GetOutputFrameSize(void) const { return m_act_post.GetOutputFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return m_act_post.GetOutputNodeSize(); }

	int   GetInputSignalDataType(void) const { return m_lut_pre.GetInputSignalDataType(); }
	int   GetInputErrorDataType(void) const { return m_lut_pre.GetInputErrorDataType(); }
	int   GetOutputSignalDataType(void) const { return m_act_post.GetOutputSignalDataType(); }
	int   GetOutputErrorDataType(void) const { return m_act_post.GetOutputErrorDataType(); }


public:

	void Forward(bool train = true)
	{
		m_lut_pre.Forward(train);
		m_act_pre.Forward(train);
		m_lut_post.Forward(train);
		m_batch_norm.Forward(train);
		m_act_post.Forward(train);
	}

	void Backward(void)
	{
		m_act_post.Backward();
		m_batch_norm.Backward();
		m_lut_post.Backward();
		m_act_pre.Backward();
		m_lut_pre.Backward();
	}

	void Update(void)
	{
		m_lut_pre.Update();
		m_act_pre.Update();
		m_lut_post.Update();
		m_batch_norm.Update();
		m_act_post.Update();
	}

public:
	// Serialize
	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
	}

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
	}


	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		m_lut_pre.Save(archive);
		m_act_pre.Save(archive);
		m_lut_post.Save(archive);
		m_batch_norm.Save(archive);
		m_act_post.Save(archive);
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		m_lut_pre.Load(archive);
		m_act_pre.Load(archive);
		m_lut_post.Load(archive);
		m_batch_norm.Load(archive);
		m_act_post.Load(archive);
	}
};


}