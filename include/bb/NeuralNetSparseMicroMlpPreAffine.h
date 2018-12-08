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
template <int N, int M, typename T = float, typename INDEX = size_t>
class NeuralNetSparseMicroMlpPreAffine : public NeuralNetLayerBuf<T, INDEX>
{
	typedef NeuralNetLayerBuf<T, INDEX>	super;

protected:
	struct Node {
		std::array<INDEX, N>	input;
		std::array<T, N * M>	W;
		std::array<T, M>		b;
		std::array<T, N * M>	dW;
		std::array<T, M>		db;

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

	INDEX						m_input_node_size = 0;
	INDEX						m_output_node_size = 0;
	INDEX						m_frame_size = 1;
	std::vector<Node>			m_node;

	bool						m_binary_mode = false;
	
public:
	NeuralNetSparseMicroMlpPreAffine() {
	}
	
	NeuralNetSparseMicroMlpPreAffine(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1,
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

	~NeuralNetSparseMicroMlpPreAffine() {}

	std::string GetClassName(void) const { return "NeuralNetSparseMicroMlpPreAffine"; }

	T& W(INDEX node, INDEX output, INDEX input)  { return m_node[node].W[output*N +input]; }
	T& b(INDEX node, INDEX output)               { return m_node[node].b[output]; }
	T& dW(INDEX node, INDEX output, INDEX input) { return m_node[node].dW[output*N + input]; }
	T& db(INDEX node, INDEX output)              { return m_node[node].db[output]; }

	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		std::vector<T> val(M);

		auto& nd = m_node[node];
		for (int i = 0; i < M; ++i) {
			val[i] = nd.b[i];
			for (int j = 0; j < N; ++j) {
				val[i] += input_value[j] * nd.W[i*N + j];
			}
		}

		return val;
	}


	void Resize(INDEX input_node_size, INDEX ouput_node_size)
	{
		m_input_node_size = input_node_size;
		m_output_node_size = ouput_node_size;
		m_node.resize(ouput_node_size);
	}

	void InitializeCoeff(std::uint64_t seed)
	{
		super::InitializeCoeff(seed);

		std::mt19937_64 mt(seed);
		std::normal_distribution<T> distribution((T)0.0, (T)1.0);
		
		for (auto& node : m_node) {
			for (auto& W : node.W) {
				W = distribution(mt);
			}
			for (auto& b : node.b) {
				b = distribution(mt);
			}

			for (auto& dW : node.dW) {
				dW = 0;
			}
			for (auto& db : node.db) {
				db = 0;
			}
		}
	}

	void  SetBinaryMode(bool enable)
	{
		m_binary_mode = enable;
	}

	void  SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		for (auto& node : m_node) {
			node.optimizer_W.reset(optimizer->Create(M*N));
			node.optimizer_b.reset(optimizer->Create(M));
		}
	}


	int   GetNodeInputSize(INDEX node) const { return N; }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { m_node[node].input[input_index] = input_node; }
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_node[node].input[input_index]; }

	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }

	INDEX GetInputNodeSize(void) const { return m_input_node_size; }
	INDEX GetOutputNodeSize(void) const { return m_output_node_size * M; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }



public:

	void Forward(bool train = true)
	{
		int node_size = (int)this->m_output_node_size;

		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();

		#pragma omp parallel for
		for ( int node = 0; node < node_size; ++node ) {

			float*	in_sig_ptr[N];
			float*	out_sig_ptr[M];
			for (int i = 0; i < N; ++i) {
				in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(m_node[node].input[i]);
			}
			for (int i = 0; i < M; ++i) {
				out_sig_ptr[i] = (float*)out_sig_buf.GetPtr(node * M + i);
			}

			__m256	W[M][N];
			__m256	b[M];
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					W[i][j] = _mm256_set1_ps(m_node[node].W[i*N + j]);
				}
				b[i] = _mm256_set1_ps(m_node[node].b[i]);
			}

			INDEX frame_size = ((m_frame_size + 7) / 8) * 8;
			for (INDEX frame = 0; frame < frame_size; frame += 8) {
				for (int i = 0; i < M; ++i) {
					__m256	acc = b[i];
					for (int j = 0; j < N; ++j) {
						__m256 sig = _mm256_load_ps(&in_sig_ptr[j][frame]);
						acc = _mm256_fmadd_ps(W[i][j], sig, acc);
					}
					_mm256_store_ps(&out_sig_ptr[i][frame], acc);
				}
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

		in_err_buf.Clear();

		for (INDEX node = 0; node < m_output_node_size; ++node ) {
			if (typeid(T) == typeid(float)) {
				auto& nd = m_node[node];

				__m256	dW[M][N];
				__m256	W[M][N];
				__m256	db[M];
				for (int i = 0; i < M; i++) {
					for (int j = 0; j < N; j++) {
						dW[i][j] = _mm256_set1_ps(0);
						W[i][j] = _mm256_set1_ps(nd.W[i*N + j]);
					}
					db[i] = _mm256_set1_ps(0);
				}

				float*	out_err_ptr[M];
				float*	in_err_ptr[N];
				float*	in_sig_ptr[N];

				for (int i = 0; i < M; ++i) {
					out_err_ptr[i] = (float*)out_err_buf.GetPtr(node*M + i);
				}
				for (int i = 0; i < N; ++i) {
					in_err_ptr[i] = (float*)in_err_buf.GetPtr(nd.input[i]);
					in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(nd.input[i]);
				}

				int frame_size = (int)((m_frame_size + 7) / 8 * 8);
				#pragma omp parallel for
				for (int frame = 0; frame < frame_size; frame += 8) {
					for (int i = 0; i < M; ++i) {
						__m256 out_err = _mm256_load_ps(&out_err_ptr[i][frame]);
						db[i] = _mm256_add_ps(db[i], out_err);
						for (int j = 0; j < N; ++j) {
							__m256 in_sig = _mm256_load_ps(&in_sig_ptr[j][frame]);
							__m256 in_err = _mm256_load_ps(&in_err_ptr[j][frame]);
							in_err = _mm256_fmadd_ps(W[i][j], out_err, in_err);
							_mm256_store_ps(&in_err_ptr[j][frame], in_err);
							dW[i][j] = _mm256_fmadd_ps(in_sig, out_err, dW[i][j]);
						}
					}
				}

				for (int i = 0; i < M; i++) {
					for (int j = 0; j < N; j++) {
						nd.dW[i*N + j] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW[i][j]));
					}
					nd.db[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db[i]));
				}
			}
		}
	}


	void Update(void)
	{
		for ( auto& nd : m_node) {
			if (m_binary_mode) {
				for (auto& dW : nd.dW) {
					dW = std::min((T)+1, std::max((T)-1, dW));
				}
			}

			nd.optimizer_W->Update(nd.W, nd.dW);
			nd.optimizer_b->Update(nd.b, nd.db);
		}

		// clear
		for ( auto& nd : m_node) {
			for (auto& dW : nd.dW) { dW = 0; }
			for (auto& db : nd.db) { db = 0; }
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
		archive(cereal::make_nvp("NeuralNetLutPre", *this));
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		archive(cereal::make_nvp("NeuralNetLutPre", *this));
	}
};


}