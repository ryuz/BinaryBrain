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


// Mini-MLP(Affine-ReLU-Affine-BatchNorm-Binarize)
template <int N = 6, int M = 16, typename T = float, typename INDEX = size_t>
class NeuralNetStackedMiniAffine : public NeuralNetSparseLayer<T, INDEX>
{
	using super = NeuralNetSparseLayer<T, INDEX>;

protected:
public:
	struct Node {
		std::array<INDEX, N>	input;

		std::array<T, M*N>		W0;
		std::array<T, M>		b0;
		std::array<T, M*N>		dW0;
		std::array<T, M>		db0;

		std::array<T, M>		W1;
		T						b1;
		std::array<T, M>		dW1;
		T						db1;

		std::unique_ptr< ParamOptimizer<T, INDEX> >	optimizer_W0;
		std::unique_ptr< ParamOptimizer<T, INDEX> >	optimizer_b0;
		std::unique_ptr< ParamOptimizer<T, INDEX> >	optimizer_W1;
		std::unique_ptr< ParamOptimizer<T, INDEX> >	optimizer_b1;

		template<class Archive>
		void serialize(Archive & archive, std::uint32_t const version)
		{
			archive(cereal::make_nvp("input", input));
			archive(cereal::make_nvp("W0", W0));
			archive(cereal::make_nvp("b0", b0));
			archive(cereal::make_nvp("W1", W1));
			archive(cereal::make_nvp("b1", b1));
		}
	};

	INDEX					m_frame_size = 1;
	std::vector<Node>		m_node;
	bool					m_binary_mode = false;

public:
	NeuralNetStackedMiniAffine() {}

	NeuralNetStackedMiniAffine(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1,
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
	
	~NeuralNetStackedMiniAffine() {}

	std::string GetClassName(void) const { return "NeuralNetStackedMiniAffine"; }

	T& W0(INDEX output, INDEX hidden, INDEX input) { return m_node[output].W0[hidden*N + input]; }
	T& b0(INDEX output, INDEX hidden) { return m_node[output].b0[hidden]; }
	T& dW0(INDEX output, INDEX hidden, INDEX input) { return m_node[output].dW0[hidden*N + input]; }
	T& db0(INDEX output, INDEX hidden) { return m_node[output].db0[hidden]; }

	T& W1(INDEX output, INDEX hidden) { return m_node[output].W1[hidden]; }
	T& b1(INDEX output) { return m_node[output].b1; }
	T& dW1(INDEX output, INDEX hidden) { return m_node[output].dW1[hidden]; }
	T& db1(INDEX output) { return m_node[output].db1; }

	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		auto& nd = m_node[node];

		// affine0
		std::vector<T> value0(M);
		for (INDEX i = 0; i < M; ++i) {
			value0[i] = nd.b0[i];
			for (INDEX j = 0; j < N; ++j) {
				value0[i] += input_value[j] * nd.W0[i*N + j];
			}
		}

		// ReLU
		for (INDEX i = 0; i < M; ++i) {
			value0[i] = std::max(value0[i], (T)0);;
		}

		// affine1
		std::vector<T> value1(1);
		value1[0] = nd.b1;
		for (INDEX i = 0; i < M; ++i) {
			value1[0] += value0[i] * nd.W1[i];
		}
		
		return value1;
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
			for (auto& W0 : node.W0) { W0 = distribution(mt); }
			for (auto& b0 : node.b0) { b0 = distribution(mt); }
			for (auto& W1 : node.W1) { W1 = distribution(mt); }
			node.b1 = distribution(mt);

			std::fill(node.dW0.begin(), node.dW0.end(), (T)0);
			std::fill(node.db0.begin(), node.db0.end(), (T)0);
			std::fill(node.dW1.begin(), node.dW1.end(), (T)0);
			node.db1 = 0;
		}
	}

	void SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		for (auto& node : m_node) {
			node.optimizer_W0.reset(optimizer->Create(M*N));
			node.optimizer_b0.reset(optimizer->Create(M));
			node.optimizer_W1.reset(optimizer->Create(M));
			node.optimizer_b1.reset(optimizer->Create(1));
		}
	}

	void  SetBinaryMode(bool enable)
	{
		m_binary_mode = enable;
	}

	int   GetNodeInputSize(INDEX node) const { return N; }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) {
		BB_ASSERT(node >= 0 && node < GetOutputNodeSize());
		BB_ASSERT(input_index >= 0 && input_index < GetNodeInputSize(node));
		BB_ASSERT(input_node >= 0 && input_node < GetInputNodeSize());
		m_node[node].input[input_index] = input_node;
	}
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_node[node].input[input_index]; }

	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }



public:
	void Forward(bool train=true)
	{
		const int		node_size = (int)this->GetOutputNodeSize();
		const int		frame_size = (int)((m_frame_size + 7) / 8 * 8);
		const __m256	zero = _mm256_set1_ps(0);

		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();

		if (typeid(T) == typeid(float)) {

#pragma omp parallel for
			for (int node = 0; node < (int)node_size; ++node) {
				auto& nd = m_node[node];

				__m256	W0[M][N];
				__m256	b0[M];
				__m256	W1[M];
				__m256	b1;
				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						W0[i][j] = _mm256_set1_ps(nd.W0[i*N + j]);
					}
					b0[i] = _mm256_set1_ps(nd.b0[i]);
					W1[i] = _mm256_set1_ps(nd.W1[i]);
				}
				b1 = _mm256_set1_ps(nd.b1);

				float*	in_sig_ptr[N];
				float*	out_sig_ptr;
				for (int i = 0; i < N; ++i) {
					in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(m_node[node].input[i]);
				}
				out_sig_ptr = (float*)out_sig_buf.GetPtr(node);

				for (int frame = 0; frame < frame_size; frame += 8) {
					__m256	in_sig[N];
					for (int i = 0; i < N; ++i) {
						in_sig[i] = _mm256_load_ps(&in_sig_ptr[i][frame]);
					}

					__m256	sum1 = b1;
					for (int i = 0; i < M; ++i) {
						// sub-layer0
						__m256	sum0 = b0[i];
						for (int j = 0; j < N; ++j) {
							sum0 = _mm256_fmadd_ps(in_sig[j], W0[i][j], sum0);
						}

						// ReLU
						sum0 = _mm256_max_ps(sum0, zero);

						// sub-layer1
						sum1 = _mm256_fmadd_ps(sum0, W1[i], sum1);
					}

					_mm256_store_ps(&out_sig_ptr[frame], sum1);
				}
			}
		}
	}

#if 0
	void Backward(void)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		auto in_err_buf = this->GetInputErrorBuffer();
		auto out_err_buf = this->GetOutputErrorBuffer();

		auto node_size = this->GetOutputNodeSize();
		const __m256	zero = _mm256_set1_ps(0);

		in_err_buf.Clear();

		if (typeid(T) == typeid(float)) {
			INDEX frame_size = (m_frame_size + 7) / 8 * 8;

			for (int node = 0; node < (int)node_size; ++node) {
				auto& nd = m_node[node];

				__m256	W0[M][N];
				__m256	b0[M];
				__m256	dW0[M][N];
				__m256	db0[M];
				__m256	W1[M];
				__m256	dW1[M];
				__m256	db1;
				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						W0[i][j] = _mm256_set1_ps(nd.W0[i*N + j]);
						dW0[i][j] = _mm256_set1_ps(nd.dW0[i*N + j]);
					}
					b0[i] = _mm256_set1_ps(nd.b0[i]);
					db0[i] = _mm256_set1_ps(nd.db0[i]);
					W1[i] = _mm256_set1_ps(nd.W1[i]);
					dW1[i] = _mm256_set1_ps(nd.dW1[i]);
				}
				db1 = _mm256_set1_ps(nd.db1);

				float*	out_err_ptr;
				float*	in_err_ptr[N];
				float*	in_sig_ptr[N];

				out_err_ptr = (float*)out_err_buf.GetPtr(node);
				for (int i = 0; i < N; ++i) {
					in_err_ptr[i] = (float*)in_err_buf.GetPtr(nd.input[i]);
					in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(nd.input[i]);
				}

#pragma omp parallel for
				for (int frame = 0; frame < frame_size; frame += 8) {
					__m256	in_sig[N];
					__m256	in_err[N];
					for (int i = 0; i < N; ++i) {
						in_sig[i] = _mm256_load_ps(&in_sig_ptr[i][frame]);
						in_err[i] = _mm256_load_ps(&in_err_ptr[i][frame]);
					}

					// 一層目の信号を再構成
					__m256	sig0[M];
					for (int i = 0; i < M; ++i) {
						// sub-layer0
						__m256	sum0 = b0[i];
						for (int j = 0; j < N; ++j) {
							sum0 = _mm256_fmadd_ps(in_sig[j], W0[i][j], sum0);
						}

						// ReLU
						sum0 = _mm256_max_ps(sum0, zero);

						sig0[i] = sum0;
					}

					// 逆伝播
					__m256 out_err = _mm256_load_ps(&out_err_ptr[frame]);
					db1 = _mm256_add_ps(db1, out_err);
					for (int i = 0; i < M; ++i) {
						__m256 err0 = _mm256_mul_ps(W1[i], out_err);
						__m256 mask = _mm256_cmp_ps(sig0[i], zero, _CMP_GT_OS);
						dW1[i] = _mm256_fmadd_ps(sig0[i], out_err, dW1[i]);

						err0 = _mm256_and_ps(err0, mask);		// ReLU

						db0[i] = _mm256_add_ps(db0[i], err0);
						for (int j = 0; j < N; ++j) {
							in_err[j] = _mm256_fmadd_ps(err0, W0[i][j], in_err[j]);
							dW0[i][j] = _mm256_fmadd_ps(err0, in_sig[j], dW0[i][j]);
						}
					}

					for (int i = 0; i < N; ++i) {
						_mm256_store_ps(&in_err_ptr[i][frame], in_err[i]);
					}
				}

				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						nd.dW0[i*N + j] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW0[i][j]));
					}
					nd.db0[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db0[i]));
					nd.dW1[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW1[i]));
				}
				nd.db1 += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db1));
			}
		}
	}
#else
	void Backward(void)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		auto out_err_buf = this->GetOutputErrorBuffer();

		auto node_size = this->GetOutputNodeSize();
		const __m256	zero = _mm256_set1_ps(0);
		
		if (typeid(T) == typeid(float)) {
			INDEX frame_size = (m_frame_size + 7) / 8 * 8;

			float* tmp_err_buf = (float *)aligned_memory_alloc(node_size*N*frame_size*sizeof(float), 32);
			
#pragma omp parallel for
			for (int node = 0; node < (int)node_size; ++node) {
				auto& nd = m_node[node];

				__m256	W0[M][N];
				__m256	b0[M];
				__m256	dW0[M][N];
				__m256	db0[M];
				__m256	W1[M];
				__m256	dW1[M];
				__m256	db1;
				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						W0[i][j] = _mm256_set1_ps(nd.W0[i*N + j]);
						dW0[i][j] = _mm256_set1_ps(nd.dW0[i*N + j]);
					}
					b0[i] = _mm256_set1_ps(nd.b0[i]);
					db0[i] = _mm256_set1_ps(nd.db0[i]);
					W1[i] = _mm256_set1_ps(nd.W1[i]);
					dW1[i] = _mm256_set1_ps(nd.dW1[i]);
				}
				db1 = _mm256_set1_ps(nd.db1);

				float*	out_err_ptr;
				float*	in_sig_ptr[N];

				float*	tmp_err_ptr = &tmp_err_buf[node * N*frame_size];


				out_err_ptr = (float*)out_err_buf.GetPtr(node);
				for (int i = 0; i < N; ++i) {
					in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(nd.input[i]);
				}

				for (int frame = 0; frame < frame_size; frame += 8) {
					__m256	in_sig[N];
					for (int i = 0; i < N; ++i) {
						in_sig[i] = _mm256_load_ps(&in_sig_ptr[i][frame]);
					}

					// 一層目の信号を再構成
					__m256	sig0[M];
					for (int i = 0; i < M; ++i) {
						// sub-layer0
						__m256	sum0 = b0[i];
						for (int j = 0; j < N; ++j) {
							sum0 = _mm256_fmadd_ps(in_sig[j], W0[i][j], sum0);
						}

						// ReLU
						sum0 = _mm256_max_ps(sum0, zero);

						sig0[i] = sum0;
					}

					// 逆伝播
					__m256	in_err[N];
					for (int i = 0; i < N; ++i) {
						in_err[i] = zero;
					}

					__m256 out_err = _mm256_load_ps(&out_err_ptr[frame]);
					db1 = _mm256_add_ps(db1, out_err);
					for (int i = 0; i < M; ++i) {
						__m256 err0 = _mm256_mul_ps(W1[i], out_err);
						__m256 mask = _mm256_cmp_ps(sig0[i], zero, _CMP_GT_OS);
						dW1[i] = _mm256_fmadd_ps(sig0[i], out_err, dW1[i]);

						err0 = _mm256_and_ps(err0, mask);		// ReLU

						db0[i] = _mm256_add_ps(db0[i], err0);
						for (int j = 0; j < N; ++j) {
							in_err[j] = _mm256_fmadd_ps(err0, W0[i][j], in_err[j]);
							dW0[i][j] = _mm256_fmadd_ps(err0, in_sig[j], dW0[i][j]);
						}
					}

					for (int i = 0; i < N; ++i) {
						_mm256_store_ps(&tmp_err_ptr[i*frame_size + frame], in_err[i]);
					}
				}

				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						nd.dW0[i*N + j] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW0[i][j]));
					}
					nd.db0[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db0[i]));
					nd.dW1[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW1[i]));
				}
				nd.db1 += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db1));
			}

			// 足しこみ
			auto in_err_buf = this->GetInputErrorBuffer();
			in_err_buf.Clear();
			for (int node = 0; node < (int)node_size; ++node) {
				auto& nd = m_node[node];

				float*	in_err_ptr[N];
				for (int i = 0; i < N; ++i) {
					in_err_ptr[i] = (float*)in_err_buf.GetPtr(nd.input[i]);
				}
				float*	tmp_err_ptr = &tmp_err_buf[node * N*frame_size];

#pragma omp parallel for
				for (int frame = 0; frame < frame_size; frame += 8) {
					for (int i = 0; i < N; ++i) {
						__m256 in_err = _mm256_load_ps(&in_err_ptr[i][frame]);
						__m256 tmp_err = _mm256_load_ps(&tmp_err_ptr[i*frame_size + frame]);
						in_err = _mm256_add_ps(in_err, tmp_err);
						_mm256_store_ps(&in_err_ptr[i][frame], in_err);
					}
				}

			}

			aligned_memory_free(tmp_err_buf);
		}
	}
#endif

	void Update(void)
	{
		auto node_size = this->GetOutputNodeSize();

		// update
		for (auto& nd : m_node) {
			nd.optimizer_W0->Update(nd.W0, nd.dW0);
			nd.optimizer_b0->Update(nd.b0, nd.db0);
			nd.optimizer_W1->Update(nd.W1, nd.dW1);
			nd.optimizer_b1->Update(nd.b1, nd.db1);
		}

		// clear
		for (auto& nd : m_node) {
			std::fill(nd.dW0.begin(), nd.dW0.end(), (T)0);
			std::fill(nd.db0.begin(), nd.db0.end(), (T)0);
			std::fill(nd.dW1.begin(), nd.dW1.end(), (T)0);
			nd.db1 = 0;
		}

		// clip
		if (m_binary_mode) {
			for (auto& nd : m_node) {
				for (auto& W0 : nd.W0) { W0 = std::min((T)+1, std::max((T)-1, W0)); }
				for (auto& W1 : nd.W1) { W1 = std::min((T)+1, std::max((T)-1, W1)); }
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
		archive(cereal::make_nvp("NeuralNetLutStackedAffine", *this));
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		archive(cereal::make_nvp("NeuralNetLutStackedAffine", *this));
	}
};


}
