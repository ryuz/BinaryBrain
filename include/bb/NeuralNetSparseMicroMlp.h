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
#include "bb/NeuralNetStackedMicroAffine.h"
#include "bb/NeuralNetBatchNormalization.h"
#include "bb/NeuralNetReLU.h"
#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetBinarize.h"


namespace bb {


// Sparce Mini-MLP(Multilayer perceptron) Layer [Affine-ReLU-Affine-BatchNorm-Binarize]
template <int N = 6, int M = 16, typename T = float>
class NeuralNetSparseMicroMlp : public NeuralNetSparseLayer<T>
{
	using super = NeuralNetSparseLayer<T>;

protected:
public:
	INDEX									m_batch_size = 0;

	// 3層で構成
	NeuralNetStackedMicroAffine<N, M, T>	m_affine;
	NeuralNetBatchNormalization<T>			m_batch_norm;
	NeuralNetSigmoid<T>						m_activation;

public:
	NeuralNetSparseMicroMlp() {}

	NeuralNetSparseMicroMlp(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1,
		const NeuralNetOptimizer<T>* optimizer = nullptr)
		: m_affine(input_node_size, output_node_size, seed, optimizer),
		m_batch_norm(output_node_size, optimizer),
		m_activation(output_node_size)
	{
		InitializeCoeff(seed);
	}
	
	~NeuralNetSparseMicroMlp() {}

	std::string GetClassName(void) const { return "NeuralNetSparseMicroMlp"; }

	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		auto vec0 = m_affine.CalcNode(node, input_value);
		auto vec1 = m_batch_norm.CalcNode(node, vec0);
		auto vec2 = m_activation.CalcNode(node, vec1);
		return vec2;
	}


	void InitializeCoeff(std::uint64_t seed)
	{
		std::mt19937_64 mt(seed);
		m_affine.InitializeCoeff(mt());
		super::InitializeCoeff(mt());
	}

	void SetOptimizer(const NeuralNetOptimizer<T>* optimizer)
	{
		m_affine.SetOptimizer(optimizer);
		m_batch_norm.SetOptimizer(optimizer);
		m_activation.SetOptimizer(optimizer);
	}

	void SetBinaryMode(bool enable)
	{
		m_affine.SetBinaryMode(enable);
		m_batch_norm.SetBinaryMode(enable);
		m_activation.SetBinaryMode(enable);
	}

	int   GetNodeInputSize(INDEX node) const { return m_affine.GetNodeInputSize(node); }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { m_affine.SetNodeInput(node, input_index, input_node); }
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_affine.GetNodeInput(node, input_index); }

	void  SetBatchSize(INDEX batch_size)
	{
		m_affine.SetBatchSize(batch_size);
		m_batch_norm.SetBatchSize(batch_size);
		m_activation.SetBatchSize(batch_size);

		if (batch_size == m_batch_size) {
			return;
		}
		m_batch_size = batch_size;

		CheckConnection(m_affine, m_batch_norm);
		CheckConnection(m_batch_norm, m_activation);

		m_affine.SetOutputSignalBuffer(m_affine.CreateOutputSignalBuffer());
		m_affine.SetOutputErrorBuffer(m_affine.CreateOutputErrorBuffer());
		m_batch_norm.SetInputSignalBuffer(m_affine.GetOutputSignalBuffer());
		m_batch_norm.SetInputErrorBuffer(m_affine.GetOutputErrorBuffer());

		m_batch_norm.SetOutputSignalBuffer(m_batch_norm.CreateOutputSignalBuffer());
		m_batch_norm.SetOutputErrorBuffer(m_batch_norm.CreateOutputErrorBuffer());
		m_activation.SetInputSignalBuffer(m_batch_norm.GetOutputSignalBuffer());
		m_activation.SetInputErrorBuffer(m_batch_norm.GetOutputErrorBuffer());
	}

	
	// 入出力バッファ
	void  SetInputSignalBuffer(NeuralNetBuffer<T> buffer) { m_affine.SetInputSignalBuffer(buffer); }
	void  SetInputErrorBuffer(NeuralNetBuffer<T> buffer) { m_affine.SetInputErrorBuffer(buffer); }
	void  SetOutputSignalBuffer(NeuralNetBuffer<T> buffer) { m_activation.SetOutputSignalBuffer(buffer); }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T> buffer) { m_activation.SetOutputErrorBuffer(buffer); }

	const NeuralNetBuffer<T>& GetInputSignalBuffer(void) const { return m_affine.GetInputSignalBuffer(); }
	const NeuralNetBuffer<T>& GetInputErrorBuffer(void) const { return m_affine.GetInputErrorBuffer(); }
	const NeuralNetBuffer<T>& GetOutputSignalBuffer(void) const { return m_activation.GetOutputSignalBuffer(); }
	const NeuralNetBuffer<T>& GetOutputErrorBuffer(void) const { return m_activation.GetOutputErrorBuffer(); }


	INDEX GetInputFrameSize(void) const { return m_affine.GetInputFrameSize(); }
	INDEX GetInputNodeSize(void) const { return m_affine.GetInputNodeSize(); }

	INDEX GetOutputFrameSize(void) const { return m_activation.GetOutputFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return m_activation.GetOutputNodeSize(); }

	int   GetInputSignalDataType(void) const { return m_affine.GetInputSignalDataType(); }
	int   GetInputErrorDataType(void) const { return m_affine.GetInputErrorDataType(); }
	int   GetOutputSignalDataType(void) const { return m_activation.GetOutputSignalDataType(); }
	int   GetOutputErrorDataType(void) const { return m_activation.GetOutputErrorDataType(); }


public:

	void Forward(bool train = true)
	{
		m_affine.Forward(train);
		m_batch_norm.Forward(train);
		m_activation.Forward(train);
	}

	void Backward(void)
	{
		m_activation.Backward();
		m_batch_norm.Backward();
		m_affine.Backward();
	}

	void Update(void)
	{
		m_affine.Update();
		m_batch_norm.Update();
		m_activation.Update();
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
		m_affine.Save(archive);
		m_batch_norm.Save(archive);
		m_activation.Save(archive);
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		m_affine.Load(archive);
		m_batch_norm.Load(archive);
		m_activation.Load(archive);
	}
};


}