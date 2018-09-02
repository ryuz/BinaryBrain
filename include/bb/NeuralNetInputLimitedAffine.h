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
#include <intrin.h>
#include <omp.h>
#include <ppl.h>
#include "NeuralNetInputLimited.h"


namespace bb {


// ì¸óÕêîêßå¿Affine
template <int N = 6, typename T = float, typename INDEX = size_t>
class NeuralNetInputLimitedAffine : public NeuralNetInputLimited<T, INDEX>
{
	typedef NeuralNetInputLimited<T, INDEX>	super;

protected:
	struct Node {
		std::array<INDEX, N>	input;
		std::array<T, N>		W;
		T						b;
		std::array<T, N>		dW;
		T						db;
	};

	INDEX						m_frame_size = 1;
	std::vector<Node>			m_node;

	
public:
	NeuralNetInputLimitedAffine() {}

	NeuralNetInputLimitedAffine(INDEX input_node_size, INDEX output_node_size, std::uint64_t seed = 1)
	{
		Resize(input_node_size, output_node_size);
		InitializeCoeff(seed);
	}

	~NeuralNetInputLimitedAffine() {}

	T& W(INDEX input, INDEX output) { return m_node[output].W[input]; }
	T& b(INDEX output) { return m_node[output].b; }
	T& dW(INDEX input, INDEX output) { return m_node[output].dW[input]; }
	T& db(INDEX output) { return[output].db; }


	T CalcNode(INDEX node, std::vector<T> input_value) const
	{
		auto& nd = m_node[node];
		T val = nd.b;
		for (int i = 0; i < N; ++i) {
			val += input_value[i] * nd.W[i];
		}
		return val;
	}


	void Resize(INDEX input_node_size, INDEX output_node_size)
	{
		super::Resize(input_node_size, output_node_size);
		
		m_node.resize(m_output_node_size);
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
		}
	}
	
	int   GetNodeInputSize(INDEX node) const { return N; }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { m_node[node].input[input_index] = input_node; }
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_node[node].input[input_index]; }

	void SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }

	int   GetInputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputValueDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }


protected:

	inline void ForwardNode(INDEX node) {
		if (typeid(T) == typeid(float)) {
			INDEX frame_size = (m_frame_size + 7) / 8;

			auto in_buf = GetInputValueBuffer();
			auto out_buf = GetOutputValueBuffer();
			
			float*	in_ptr[N];
			float*	out_ptr;
			for (int i = 0; i < N; ++i) {
				in_ptr[i] = (float*)in_buf.GetPtr(m_node[node].input[i]);
			}
			out_ptr = (float*)out_buf.GetPtr(node);

			__m256	W[N];
			for (int i = 0; i < N; ++i) {
				W[i] = _mm256_set1_ps(m_node[node].W[i]);
			}
			__m256 b = _mm256_set1_ps(m_node[node].b);

			for (INDEX frame = 0; frame < frame_size; ++frame) {
				__m256	acc = b;
				for (int i = 0; i < N; ++i) {
					__m256 val = _mm256_load_ps(in_ptr[i]);	in_ptr[i] += 8;
					acc = _mm256_fmadd_ps(W[i], val, acc);
				}
				_mm256_store_ps(out_ptr, acc);	out_ptr += 8;
			}
		}
	}

public:

	void Forward(void)
	{
		auto node_size = GetOutputNodeSize();
		concurrency::parallel_for<INDEX>(0, node_size, [&](INDEX node)
		{
			ForwardNode(node);
		});
	}

	void Backward(void)
	{
		auto in_val_buf = GetInputValueBuffer();
		auto out_val_buf = GetOutputValueBuffer();
		auto in_err_buf = GetInputErrorBuffer();
		auto out_err_buf = GetOutputErrorBuffer();

		auto node_size = GetOutputNodeSize();

		INDEX frame_size = (m_frame_size + 7) / 8;

		in_err_buf.Clear();

		for (INDEX node = 0; node < node_size; ++ node ) {
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
				float*	in_val_ptr[N];

				out_err_ptr = (float*)out_err_buf.GetPtr(node);
				for (int i = 0; i < N; ++i) {
					in_err_ptr[i] = (float*)in_err_buf.GetPtr(nd.input[i]);
					in_val_ptr[i] = (float*)in_val_buf.GetPtr(nd.input[i]);
				}

				for (size_t frame = 0; frame < frame_size; ++frame) {
					__m256 out_err = _mm256_load_ps(out_err_ptr);	out_err_ptr += 8;
					db = _mm256_add_ps(db, out_err);
					for (int i = 0; i < N; ++i) {
						__m256 in_val = _mm256_load_ps(in_val_ptr[i]);	in_val_ptr[i] += 8;
						__m256 in_err = _mm256_load_ps(in_err_ptr[i]);
						in_err = _mm256_fmadd_ps(W[i], out_err, in_err);
						_mm256_store_ps(in_err_ptr[i], in_err);	in_err_ptr[i] += 8;

						dW[i] = _mm256_fmadd_ps(in_val, out_err, dW[i]);
					}
				}

				for (int i = 0; i < N; i++) {
					nd.dW[i] = 0;
					for (int j = 0; j < 8; j++) {
						nd.dW[i] += dW[i].m256_f32[j];
					}
				}
				nd.db = 0;
				for (int j = 0; j < 8; j++) {
					nd.db += db.m256_f32[j];
				}
			}
		}
	}


	void Update(double learning_rate)
	{
		auto node_size = GetOutputNodeSize();

		for (INDEX node = 0; node < node_size; ++node) {
			auto& nd = m_node[node];
			for (int i = 0; i < N; ++i) {
				nd.W[i] -= nd.dW[i] * (T)learning_rate;
			}
			nd.b -= nd.db * (T)learning_rate;
		}
	}

};


}