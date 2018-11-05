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

#include "bb/NeuralNetLayerBuf.h"
#include "bb/NeuralNetOptimizerSgd.h"


namespace bb {


// 入力数制限Affine
template <int M, typename T = float, typename INDEX = size_t>
class NeuralNetSparseMiniMlpPostAffine : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	struct Node {
		std::array<T, M>	W;
		T					b;
		std::array<T, M>	dW;
		T					db;

		std::unique_ptr< ParamOptimizer<T, INDEX> >	optimizer_W;
		std::unique_ptr< ParamOptimizer<T, INDEX> >	optimizer_b;

		template<class Archive>
		void serialize(Archive & archive, std::uint32_t const version)
		{
			archive(cereal::make_nvp("W", W));
			archive(cereal::make_nvp("b", b));
		}
	};

	INDEX						m_node_size  = 0;
	INDEX						m_frame_size = 1;
	std::vector<Node>			m_node;
	bool						m_binary_mode = false;
	
public:
	NeuralNetSparseMiniMlpPostAffine() {
	}
	
	NeuralNetSparseMiniMlpPostAffine(INDEX output_node_size, std::uint64_t seed = 1,
		const NeuralNetOptimizer<T, INDEX>* optimizer = nullptr)
	{
		NeuralNetOptimizerSgd<T, INDEX> DefOptimizer;
		if (optimizer == nullptr) {
			optimizer = &DefOptimizer;
		}

		Resize(output_node_size);
		InitializeCoeff(seed);
		SetOptimizer(optimizer);
	}

	~NeuralNetSparseMiniMlpPostAffine() {}

	std::string GetClassName(void) const { return "NeuralNetSparseMiniMlpPostAffine"; }

	T& W(INDEX output, INDEX input)  { return m_node[output].W[input]; }
	T& b(INDEX output)               { return m_node[output].b; }
	T& dW(INDEX output, INDEX input) { return m_node[output].dW[input]; }
	T& db(INDEX output)              { return m_node[output].db; }

	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		std::vector<T> val(1);

		auto& nd = m_node[node];
		val[0] = nd.b;
		for (int i = 0; i < M; ++i) {
			val[0] += input_value[i] * nd.W[i];
		}

		return val;
	}


	void Resize(INDEX ouput_node_size)
	{
		m_node_size = ouput_node_size;
		m_node.resize(ouput_node_size);
	}

	void InitializeCoeff(std::uint64_t seed)
	{
		std::mt19937_64 mt(seed);
		std::normal_distribution<T> distribution((T)0.0, (T)1.0);
		
		for (auto& nd : m_node) {
			for (auto& W : nd.W) {
				W = distribution(mt);
			}
			nd.b = distribution(mt);

			for (auto& dW : nd.dW) {
				dW = 0;
			}
			nd.db = 0;
		}
	}

	void  SetBinaryMode(bool enable)
	{
		m_binary_mode = enable;
	}

	void  SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		for (auto& node : m_node) {
			node.optimizer_W.reset(optimizer->Create(M));
			node.optimizer_b.reset(optimizer->Create(1));
		}
	}

	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }

	INDEX GetInputNodeSize(void) const { return m_node_size * M; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }


public:

	void Forward(bool train = true)
	{
		auto node_size = this->GetOutputNodeSize();

		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();

#pragma omp parallel for
		for (int node = 0; node < (int)node_size; ++node) {
			float*	in_sig_ptr[M];
			float*	out_sig_ptr;
			for (int i = 0; i < M; ++i) {
				in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(node * M + i);
			}
			out_sig_ptr = (float*)out_sig_buf.GetPtr(node);

			__m256	W[M];
			for (int i = 0; i < M; ++i) {
				W[i] = _mm256_set1_ps(m_node[node].W[i]);
			}
			__m256 b = _mm256_set1_ps(m_node[node].b);

			INDEX frame_size = (m_frame_size + 7) / 8 * 8;
			for (INDEX frame = 0; frame < frame_size; frame += 8) {
				__m256	acc = b;
				for (int i = 0; i < M; ++i) {
					__m256 sig = _mm256_load_ps(&in_sig_ptr[i][frame]);
					acc = _mm256_fmadd_ps(W[i], sig, acc);
				}
				_mm256_store_ps(&out_sig_ptr[frame], acc);
			}
		}

		out_sig_buf.ClearMargin();
	}

	void Backward(void)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		auto in_err_buf = this->GetInputErrorBuffer();
		auto out_err_buf = this->GetOutputErrorBuffer();

		auto node_size = this->GetOutputNodeSize();

		for (int node = 0; node < (int)node_size; ++node) {
			if (typeid(T) == typeid(float)) {
				auto& nd = m_node[node];

				__m256	dW[M];
				__m256	W[M];
				for (int i = 0; i < M; i++) {
					dW[i] = _mm256_set1_ps(0);
					W[i] = _mm256_set1_ps(nd.W[i]);
				}
				__m256	db = _mm256_set1_ps(0);

				float*	out_err_ptr;
				float*	in_err_ptr[M];
				float*	in_sig_ptr[M];

				out_err_ptr = (float*)out_err_buf.GetPtr(node);
				for (int i = 0; i < M; ++i) {
					in_err_ptr[i] = (float*)in_err_buf.GetPtr(node*M + i);
					in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(node*M + i);
				}

				INDEX frame_size = (m_frame_size + 7) / 8 * 8;
				#pragma omp parallel for
				for (int frame = 0; frame < (int)frame_size; frame += 8) {
					__m256 out_err = _mm256_load_ps(&out_err_ptr[frame]);
					db = _mm256_add_ps(db, out_err);
					for (int i = 0; i < M; ++i) {
						__m256 in_sig = _mm256_load_ps(&in_sig_ptr[i][frame]);
						__m256 in_err = _mm256_mul_ps(W[i], out_err);
						_mm256_store_ps(&in_err_ptr[i][frame], in_err);
						dW[i] = _mm256_fmadd_ps(in_sig, out_err, dW[i]);
					}
				}

				for (int i = 0; i < M; i++) {
					nd.dW[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW[i]));

				}
				nd.db += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db));
			}
		}
	}


	void Update(void)
	{
		for (auto& nd : m_node) {
			nd.optimizer_W->Update(nd.W, nd.dW);
			nd.optimizer_b->Update(nd.b, nd.db);

			if (m_binary_mode) {
				for ( auto& W : nd.W ) {
					W = std::min((T)+1, std::max((T)-1, W));
				}
			}
		}

		// clear
		for (auto& nd : m_node) {
			for (auto& dW : nd.dW) {
				dW = 0;
			}
			nd.db = 0;
		}
	}

public:

	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
		archive(cereal::make_nvp("node", m_node));
		archive(cereal::make_nvp("binary_mode", m_binary_mode));
	}

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("node", m_node));
		archive(cereal::make_nvp("binary_mode", m_binary_mode));
	}

	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		archive(cereal::make_nvp("NeuralNetLutPost", *this));
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		archive(cereal::make_nvp("NeuralNetLutPost", *this));
	}
};


}