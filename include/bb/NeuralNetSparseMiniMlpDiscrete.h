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
#include "bb/NeuralNetSparseMiniMlpPreAffine.h"
#include "bb/NeuralNetSparseMiniMlpPostAffine.h"
#include "bb/NeuralNetBatchNormalization.h"
#include "bb/NeuralNetReLU.h"
#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetBinarize.h"


namespace bb {


// 入力数制限Affine Binary Connect版
template <int N = 6, int M = 16, typename ACTIVATION0=NeuralNetReLU<float, size_t>, typename ACTIVATION1 = NeuralNetSigmoid<float, size_t>, typename T = float, typename INDEX = size_t>
class NeuralNetSparseMiniMlpDiscrete : public NeuralNetSparseLayer<T, INDEX>
{
	using super = NeuralNetSparseLayer<T, INDEX>;

protected:
public:
	INDEX									m_batch_size = 0;

	// 3層で構成
	NeuralNetSparseMiniMlpPreAffine<N, M, T, INDEX>		m_pre_affine;
	ACTIVATION0											m_pre_act;
//	NeuralNetSigmoid<T, INDEX>							m_pre_act;
	NeuralNetSparseMiniMlpPostAffine<M, T, INDEX>		m_post_affine;
	NeuralNetBatchNormalization<T, INDEX>				m_batch_norm;
	ACTIVATION1											m_act_post;

public:
	NeuralNetSparseMiniMlpDiscrete() {}

	NeuralNetSparseMiniMlpDiscrete(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1,
		const NeuralNetOptimizer<T, INDEX>* optimizer = nullptr)
		: m_pre_affine(input_node_size, output_node_size, seed, optimizer),
		m_pre_act(output_node_size * M),
		m_post_affine(output_node_size, seed, optimizer),
		m_batch_norm(output_node_size, optimizer),
		m_act_post(output_node_size)
	{
		InitializeCoeff(seed);
	}
	
	~NeuralNetSparseMiniMlpDiscrete() {}

	std::string GetClassName(void) const { return "NeuralNetSparseMiniMlpDiscrete"; }

	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		auto vec0 = m_pre_affine.CalcNode(node, input_value);
		auto vec1 = m_pre_act.CalcNode(node, vec0);
		auto vec2 = m_post_affine.CalcNode(node, vec1);
		auto vec3 = m_batch_norm.CalcNode(node, vec2);
		auto vec4 = m_act_post.CalcNode(node, vec3);
		return vec4;
	}


	void InitializeCoeff(std::uint64_t seed)
	{
		std::mt19937_64 mt(seed);
		m_pre_affine.InitializeCoeff(mt());
		m_post_affine.InitializeCoeff(mt());
		super::InitializeCoeff(mt());
	}

	void SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		m_pre_affine.SetOptimizer(optimizer);
		m_pre_act.SetOptimizer(optimizer);
		m_post_affine.SetOptimizer(optimizer);
		m_batch_norm.SetOptimizer(optimizer);
		m_act_post.SetOptimizer(optimizer);
	}

	void SetBinaryMode(bool enable)
	{
		m_pre_affine.SetBinaryMode(enable);
//		m_pre_act.SetBinaryMode(enable);
		m_post_affine.SetBinaryMode(enable);
//		m_batch_norm.SetBinaryMode(enable);
		m_act_post.SetBinaryMode(enable);
	}

	int   GetNodeInputSize(INDEX node) const { return m_pre_affine.GetNodeInputSize(node); }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { m_pre_affine.SetNodeInput(node, input_index, input_node); }
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_pre_affine.GetNodeInput(node, input_index); }

	void  SetBatchSize(INDEX batch_size)
	{
		m_pre_affine.SetBatchSize(batch_size);
		m_pre_act.SetBatchSize(batch_size);
		m_post_affine.SetBatchSize(batch_size);
		m_batch_norm.SetBatchSize(batch_size);
		m_act_post.SetBatchSize(batch_size);

		if (batch_size == m_batch_size) {
			return;
		}
		m_batch_size = batch_size;

		CheckConnection(m_pre_affine, m_pre_act);
		CheckConnection(m_pre_act, m_post_affine);
		CheckConnection(m_post_affine, m_batch_norm);
		CheckConnection(m_batch_norm, m_act_post);

		m_pre_affine.SetOutputSignalBuffer(m_pre_affine.CreateOutputSignalBuffer());
		m_pre_affine.SetOutputErrorBuffer(m_pre_affine.CreateOutputErrorBuffer());
		m_pre_act.SetInputSignalBuffer(m_pre_affine.GetOutputSignalBuffer());
		m_pre_act.SetInputErrorBuffer(m_pre_affine.GetOutputErrorBuffer());

		m_pre_act.SetOutputSignalBuffer(m_pre_act.CreateOutputSignalBuffer());
		m_pre_act.SetOutputErrorBuffer(m_pre_act.CreateOutputErrorBuffer());
		m_post_affine.SetInputSignalBuffer(m_pre_act.GetOutputSignalBuffer());
		m_post_affine.SetInputErrorBuffer(m_pre_act.GetOutputErrorBuffer());

		m_post_affine.SetOutputSignalBuffer(m_post_affine.CreateOutputSignalBuffer());
		m_post_affine.SetOutputErrorBuffer(m_post_affine.CreateOutputErrorBuffer());
		m_batch_norm.SetInputSignalBuffer(m_post_affine.GetOutputSignalBuffer());
		m_batch_norm.SetInputErrorBuffer(m_post_affine.GetOutputErrorBuffer());

		m_batch_norm.SetOutputSignalBuffer(m_batch_norm.CreateOutputSignalBuffer());
		m_batch_norm.SetOutputErrorBuffer(m_batch_norm.CreateOutputErrorBuffer());
		m_act_post.SetInputSignalBuffer(m_batch_norm.GetOutputSignalBuffer());
		m_act_post.SetInputErrorBuffer(m_batch_norm.GetOutputErrorBuffer());
	}

	
	// 入出力バッファ
	void  SetInputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_pre_affine.SetInputSignalBuffer(buffer); }
	void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_pre_affine.SetInputErrorBuffer(buffer); }
	void  SetOutputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_act_post.SetOutputSignalBuffer(buffer); }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_act_post.SetOutputErrorBuffer(buffer); }

	const NeuralNetBuffer<T, INDEX>& GetInputSignalBuffer(void) const { return m_pre_affine.GetInputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) const { return m_pre_affine.GetInputErrorBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputSignalBuffer(void) const { return m_act_post.GetOutputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) const { return m_act_post.GetOutputErrorBuffer(); }


	INDEX GetInputFrameSize(void) const { return m_pre_affine.GetInputFrameSize(); }
	INDEX GetInputNodeSize(void) const { return m_pre_affine.GetInputNodeSize(); }

	INDEX GetOutputFrameSize(void) const { return m_act_post.GetOutputFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return m_act_post.GetOutputNodeSize(); }

	int   GetInputSignalDataType(void) const { return m_pre_affine.GetInputSignalDataType(); }
	int   GetInputErrorDataType(void) const { return m_pre_affine.GetInputErrorDataType(); }
	int   GetOutputSignalDataType(void) const { return m_act_post.GetOutputSignalDataType(); }
	int   GetOutputErrorDataType(void) const { return m_act_post.GetOutputErrorDataType(); }


public:

	void Forward(bool train = true)
	{
		m_pre_affine.Forward(train);
		m_pre_act.Forward(train);
		m_post_affine.Forward(train);
		m_batch_norm.Forward(train);
		m_act_post.Forward(train);
	}

	void Backward(void)
	{
		m_act_post.Backward();
		m_batch_norm.Backward();
		m_post_affine.Backward();
		m_pre_act.Backward();
		m_pre_affine.Backward();
	}

	void Update(void)
	{
		m_pre_affine.Update();
		m_pre_act.Update();
		m_post_affine.Update();
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
		m_pre_affine.Save(archive);
		m_pre_act.Save(archive);
		m_post_affine.Save(archive);
		m_batch_norm.Save(archive);
		m_act_post.Save(archive);
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		m_pre_affine.Load(archive);
		m_pre_act.Load(archive);
		m_post_affine.Load(archive);
		m_batch_norm.Load(archive);
		m_act_post.Load(archive);
	}
};


}