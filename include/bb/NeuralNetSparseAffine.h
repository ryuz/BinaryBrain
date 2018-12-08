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

#include "cereal/archives/json.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/array.hpp"

#include "bb/NeuralNetSparseLayer.h"
#include "bb/NeuralNetOptimizerSgd.h"


namespace bb {


// 入力数制限Affine
template <int N = 6, typename T = float, typename INDEX = size_t>
class NeuralNetSparseAffine : public NeuralNetSparseLayer<T, INDEX>
{
	typedef NeuralNetSparseLayer<T, INDEX>	super;

protected:
	struct Node {
		std::array<INDEX, N>	input;
		std::array<T, N>		W;
		T						b;
		std::array<T, N>		dW;
		T						db;

		std::unique_ptr< ParamOptimizer<T, INDEX> >	optimizer_W;
		std::unique_ptr< ParamOptimizer<T, INDEX> >	optimizer_b;

		template<class Archive>
		void serialize(Archive & archive, std::uint32_t const version)
		{
			archive(cereal::make_nvp("input", input));
			archive(cereal::make_nvp("W", W));
			archive(cereal::make_nvp("b", b));
		}

	};

	INDEX						m_frame_size = 1;
	std::vector<Node>			m_node;

	bool						m_binary_mode = false;
	
public:
	NeuralNetSparseAffine() {
	}
	
	NeuralNetSparseAffine(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1,
		const NeuralNetOptimizer<T, INDEX>* optimizer = nullptr)
	{
		NeuralNetOptimizerSgd<T, INDEX> DefOptimizer;
		if (optimizer == nullptr) {
			optimizer = &DefOptimizer;
		}

		Resize(input_node_size, output_node_size);
		InitializeCoeff(seed);
		SetOptimizer(optimizer);
	}

	~NeuralNetSparseAffine() {}

	std::string GetClassName(void) const { return "NeuralNetSparseAffine"; }

	T& W(INDEX output, INDEX input) { return m_node[output].W[input]; }
	T& b(INDEX output) { return m_node[output].b; }
	T& dW(INDEX output, INDEX input) { return m_node[output].dW[input]; }
	T& db(INDEX output) { return m_node[output].db; }

	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		std::vector<T>	val(1);

		auto& nd = m_node[node];
		val[0] = nd.b;
		for (int i = 0; i < N; ++i) {
			val[0] += input_value[i] * nd.W[i];
		}
		
		return val;
	}


	void Resize(INDEX input_node_size, INDEX output_node_size)
	{
		super::Resize(input_node_size, output_node_size);
		
		m_node.resize(this->m_output_node_size);
	}

	void InitializeCoeff(std::uint64_t seed)
	{
		super::InitializeCoeff(seed);

		std::mt19937_64 mt(seed);
		std::normal_distribution<T> distribution((T)0.0, (T)1.0);
		
		for (auto& node : m_node) {
			for (auto& w : node.W) {
				w = distribution(mt);
			}
			node.b = distribution(mt);

			for (auto& dw : node.dW) {
				dw = 0;
			}
			node.db = 0;
		}
	}

	void  SetBinaryMode(bool enable)
	{
		m_binary_mode = enable;
	}

	void  SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		for (auto& node : m_node) {
			node.optimizer_W.reset(optimizer->Create(N));
			node.optimizer_b.reset(optimizer->Create(1));
		}
	}


	int   GetNodeInputSize(INDEX node) const { return N; }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { m_node[node].input[input_index] = input_node; }
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_node[node].input[input_index]; }

	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }


protected:

	inline void ForwardNode(INDEX node) {
		if (typeid(T) == typeid(float)) {
			INDEX frame_size = (m_frame_size + 7) / 8;

			auto in_sig_buf = this->GetInputSignalBuffer();
			auto out_sig_buf = this->GetOutputSignalBuffer();
			
			float*	in_sig_ptr[N];
			float*	out_sig_ptr;
			for (int i = 0; i < N; ++i) {
				in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(m_node[node].input[i]);
			}
			out_sig_ptr = (float*)out_sig_buf.GetPtr(node);

			__m256	W[N];
			for (int i = 0; i < N; ++i) {
				W[i] = _mm256_set1_ps(m_node[node].W[i]);
			}
			__m256 b = _mm256_set1_ps(m_node[node].b);

			for (INDEX frame = 0; frame < frame_size; ++frame) {
				__m256	acc = b;
				for (int i = 0; i < N; ++i) {
					__m256 sig = _mm256_load_ps(in_sig_ptr[i]);	in_sig_ptr[i] += 8;
					acc = _mm256_fmadd_ps(W[i], sig, acc);
				}
				_mm256_store_ps(out_sig_ptr, acc);	out_sig_ptr += 8;
			}
		}
	}

public:

	void Forward(bool train = true)
	{
		auto node_size = this->GetOutputNodeSize();

		#pragma omp parallel for
		for ( int node = 0; node < (int)node_size; ++node ) {
			ForwardNode(node);
		}
	}

	void Backward(void)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		auto in_err_buf = this->GetInputErrorBuffer();
		auto out_err_buf = this->GetOutputErrorBuffer();

		auto node_size = this->GetOutputNodeSize();

		INDEX frame_size = (m_frame_size + 7) / 8;

		in_err_buf.Clear();

		for (int node = 0; node < (int)node_size; ++node ) {
			if (typeid(T) == typeid(float)) {
				auto& nd = m_node[node];

				__m256	dW[N];
				__m256	W[N];
				for (int i = 0; i < N; i++) {
					dW[i] = _mm256_set1_ps(0);
					W[i]  = _mm256_set1_ps(nd.W[i]);
				}
				__m256	db = _mm256_set1_ps(0);

				float*	out_err_ptr;
				float*	in_err_ptr[N];
				float*	in_sig_ptr[N];

				out_err_ptr = (float*)out_err_buf.GetPtr(node);
				for (int i = 0; i < N; ++i) {
					in_err_ptr[i] = (float*)in_err_buf.GetPtr(nd.input[i]);
					in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(nd.input[i]);
				}

	//			#pragma omp parallel for
				for (int frame = 0; frame < (int)frame_size; ++frame) {
					__m256 out_err = _mm256_load_ps(out_err_ptr);	out_err_ptr += 8;
					db = _mm256_add_ps(db, out_err);
					for (int i = 0; i < N; ++i) {
						__m256 in_sig = _mm256_load_ps(in_sig_ptr[i]);	in_sig_ptr[i] += 8;
						__m256 in_err = _mm256_load_ps(in_err_ptr[i]);
						in_err = _mm256_fmadd_ps(W[i], out_err, in_err);
						_mm256_store_ps(in_err_ptr[i], in_err);	in_err_ptr[i] += 8;

						dW[i] = _mm256_fmadd_ps(in_sig, out_err, dW[i]);
					}
				}

				for (int i = 0; i < N; i++) {
					nd.dW[i] = bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW[i]));
				}
				nd.db = bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db));
			}
		}
	}


	void Update(void)
	{
		auto node_size = this->GetOutputNodeSize();

		// update
		for (auto& nd : m_node) {
			nd.optimizer_W->Update(nd.W, nd.dW);
			nd.optimizer_b->Update(nd.b, nd.db);
		}
		
		// clear
	//	for (auto& nd : m_node) {
	//		for (int i = 0; i < N; ++i) {
	//			nd.dW[i] = 0;
	//		}
	//		nd.db = 0;
	//	}

		// clip
		if (m_binary_mode) {
			for (auto& nd : m_node) {
				for (int i = 0; i < N; ++i) {
					nd.W[i] = std::min((T)+1, std::max((T)-1, nd.W[i]));
				}
			}
		}
	}

public:

	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
		archive(cereal::make_nvp("NeuralNetSparseLayer", *(super *)this));
		archive(cereal::make_nvp("node", m_node));
		archive(cereal::make_nvp("binary_mode", m_binary_mode));
	}

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("NeuralNetSparseLayer", *(super *)this));
		archive(cereal::make_nvp("node", m_node));
		archive(cereal::make_nvp("binary_mode", m_binary_mode));
	}

	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		archive(cereal::make_nvp("NeuralNetSparseAffine", *this));
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		archive(cereal::make_nvp("NeuralNetSparseAffine", *this));
	}
};


}