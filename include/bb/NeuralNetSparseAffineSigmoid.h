// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>

#include "bb/NeuralNetSparseLayer.h"
#include "bb/NeuralNetSparseAffine.h"
#include "bb/NeuralNetBatchNormalization.h"
#include "bb/NeuralNetBinarize.h"


namespace bb {


// 入力数制限Affine Binary Connect版
template <int N = 6, typename T = float, typename INDEX = size_t>
class NeuralNetSparseAffineSigmoid : public NeuralNetSparseLayer<T, INDEX>
{
protected:
	INDEX									m_batch_size = 0;

	// 3層で構成
	NeuralNetSparseAffine<N, T, INDEX>		m_affine;
	NeuralNetBatchNormalization<T, INDEX>	m_norm;
	NeuralNetSigmoid<T, INDEX>				m_activation;
	

public:
	NeuralNetSparseAffineSigmoid() {}

	NeuralNetSparseAffineSigmoid(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1,
		const NeuralNetOptimizer<T, INDEX>* optimizer = nullptr)
		: m_affine(input_node_size, output_node_size, seed, optimizer),
		m_norm(output_node_size),
		m_activation(output_node_size)
	{
	}
	
	~NeuralNetSparseAffineSigmoid() {}

	std::string GetClassName(void) const { return "NeuralNetSparseAffineSigmoid"; }

	T CalcNode(INDEX node, std::vector<T> input_value) const
	{
		std::vector<T> vec(1);
		vec[0] = m_affine.CalcNode(node, input_value);
		vec[0] = m_norm.CalcNode(node, vec);
		return m_activation.CalcNode(node, vec);
	}


	void InitializeCoeff(std::uint64_t seed)
	{
		m_affine.InitializeCoeff(seed);
		m_norm.InitializeCoeff(seed);
		m_activation.InitializeCoeff(seed);
	}

	void  SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		m_affine.SetOptimizer(optimizer);
		m_norm.SetOptimizer(optimizer);
		m_activation.SetOptimizer(optimizer);
	}

	void SetBinaryMode(bool enable)
	{
		m_affine.SetBinaryMode(enable);
		m_norm.SetBinaryMode(enable);
		m_activation.SetBinaryMode(enable);
	}

	int   GetNodeInputSize(INDEX node) const { return m_affine.GetNodeInputSize(node); }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { m_affine.SetNodeInput(node, input_index, input_node); }
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_affine.GetNodeInput(node, input_index); }

	void  SetMuxSize(INDEX mux_size)
	{
		m_affine.SetMuxSize(mux_size);
		m_norm.SetMuxSize(mux_size);
		m_activation.SetMuxSize(mux_size);
	}

	void  SetBatchSize(INDEX batch_size)
	{
		m_affine.SetBatchSize(batch_size);
		m_norm.SetBatchSize(batch_size);
		m_activation.SetBatchSize(batch_size);

		if (batch_size == m_batch_size) {
			return;
		}
		m_batch_size = batch_size;

		m_affine.SetOutputSignalBuffer(m_affine.CreateOutputSignalBuffer());
		m_affine.SetOutputErrorBuffer(m_affine.CreateOutputErrorBuffer());
		m_norm.SetInputSignalBuffer(m_affine.GetOutputSignalBuffer());
		m_norm.SetInputErrorBuffer(m_affine.GetOutputErrorBuffer());

		m_norm.SetOutputSignalBuffer(m_norm.CreateOutputSignalBuffer());
		m_norm.SetOutputErrorBuffer(m_norm.CreateOutputErrorBuffer());
		m_activation.SetInputSignalBuffer(m_norm.GetOutputSignalBuffer());
		m_activation.SetInputErrorBuffer(m_norm.GetOutputErrorBuffer());
	}

	
	// 入出力バッファ
	void  SetInputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_affine.SetInputSignalBuffer(buffer); }
	void  SetOutputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_activation.SetOutputSignalBuffer(buffer); }
	void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_affine.SetInputErrorBuffer(buffer); }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_activation.SetOutputErrorBuffer(buffer); }

	const NeuralNetBuffer<T, INDEX>& GetInputSignalBuffer(void) const { return m_affine.GetInputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputSignalBuffer(void) const { return m_activation.GetOutputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) const { return m_affine.GetInputErrorBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) const { return m_activation.GetOutputErrorBuffer(); }


	INDEX GetInputFrameSize(void) const { return m_affine.GetInputFrameSize(); }
	INDEX GetOutputFrameSize(void) const { return m_activation.GetOutputFrameSize(); }

	INDEX GetInputNodeSize(void) const { return m_affine.GetInputNodeSize(); }
	INDEX GetOutputNodeSize(void) const { return m_activation.GetOutputNodeSize(); }

	int   GetInputSignalDataType(void) const { return m_affine.GetInputSignalDataType(); }
	int   GetInputErrorDataType(void) const { return m_affine.GetInputErrorDataType(); }
	int   GetOutputSignalDataType(void) const { return m_activation.GetOutputSignalDataType(); }
	int   GetOutputErrorDataType(void) const { return m_activation.GetOutputErrorDataType(); }


public:

	void Forward(bool train = true)
	{
		m_affine.Forward(train);
		m_norm.Forward(train);
		m_activation.Forward(train);
	}

	void Backward(void)
	{
		m_activation.Backward();
		m_norm.Backward();
		m_affine.Backward();
	}

	void Update(void)
	{
		m_affine.Update();
		m_norm.Update();
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
		m_norm.Save(archive);
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		m_affine.Load(archive);
		m_norm.Load(archive);
	}
};


}