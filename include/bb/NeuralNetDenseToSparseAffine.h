// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>
#include <algorithm>

#include <Eigen/Core>

#include "bb/NeuralNetLayerBuf.h"
#include "bb/NeuralNetOptimizerSgd.h"


namespace bb {


// 徐々にSparse化
template <typename T = float>
class NeuralNetDenseToSparseAffine : public NeuralNetLayerBuf<T>
{
protected:
	using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using Stride = Eigen::Stride<Eigen::Dynamic, 1>;
	using MatMap = Eigen::Map<Matrix, 0, Stride>;

	bool		m_decimate_enable       = true;
	double		m_decimate_update_rate  = 0.999;
	double		m_decimate_current_rate = 1.0;
	INDEX		m_decimate_current_size;
	INDEX		m_decimate_target_size;
	std::vector< std::vector<INDEX> >	m_sort_table;

	INDEX		m_frame_size = 1;
	INDEX		m_input_size = 0;
	INDEX		m_output_size = 0;

	Matrix		m_W;
	Vector		m_b;
	Matrix		m_dW;
	Vector		m_db;

public:
	std::mt19937_64	m_mt;
	std::vector<bool>	m_mask_vec;
	Matrix				m_Mask;
	Matrix				m_N;
	Matrix				m_ErrSum;
protected:

	std::unique_ptr< ParamOptimizer<T> >	m_optimizer_W;
	std::unique_ptr< ParamOptimizer<T> >	m_optimizer_b;

	bool		m_binary_mode = false;

public:
	NeuralNetDenseToSparseAffine() {}

	NeuralNetDenseToSparseAffine(int target, double rate, INDEX input_size, INDEX output_size, std::uint64_t seed = 1,
		const NeuralNetOptimizer<T>* optimizer = nullptr)
	{
		NeuralNetOptimizerSgd<T> DefOptimizer;
		if (optimizer == nullptr) {
			optimizer = &DefOptimizer;
		}

		Resize(target, rate, input_size, output_size);
		InitializeCoeff(seed);
		SetOptimizer(optimizer);
	}

	~NeuralNetDenseToSparseAffine() {}		// デストラクタ

	std::string GetClassName(void) const { return "NeuralNetDenseToSparseAffine"; }

	void Resize(int target, double rate, INDEX input_size, INDEX output_size)
	{
		m_input_size = input_size;
		m_output_size = output_size;
		m_W = Matrix::Random(input_size, output_size);
		m_b = Vector::Random(output_size);
		m_dW = Matrix::Zero(input_size, output_size);
		m_db = Vector::Zero(output_size);

		m_Mask = Matrix::Zero(input_size, output_size);
		m_N    = Matrix::Zero(input_size, output_size);
		m_ErrSum = Matrix::Zero(input_size, output_size);
		m_mask_vec.resize(input_size * output_size);
		for (size_t i = 0; i < m_mask_vec.size(); ++i) {
			m_mask_vec[i] = 1;// (i % 2 == 1);
		}
		std::random_shuffle(m_mask_vec.begin(), m_mask_vec.end());
		
		m_decimate_current_size = m_input_size;
		m_decimate_target_size = target;
		m_decimate_current_rate = 1.0;
		m_decimate_update_rate  = rate;
		m_sort_table.resize(m_output_size);
		for (INDEX i = 0; i < m_output_size; ++i) {
			m_sort_table[i].resize(m_input_size);
			for (INDEX j = 0; j < m_input_size; ++j) {
				m_sort_table[i][j] = j;
			}

			// ちょっと実験
			std::random_shuffle(m_sort_table[i].begin(), m_sort_table[i].end());
		}
	}

	void InitializeCoeff(std::uint64_t seed)
	{
		std::mt19937_64 mt(seed);
		std::normal_distribution<T>		real_dist((T)0.0, (T)1.0);

		for (INDEX i = 0; i < m_input_size; ++i) {
			for (INDEX j = 0; j < m_output_size; ++j) {
				m_W(i, j) = real_dist(mt);
			}
		}

		for (INDEX j = 0; j < m_output_size; ++j) {
			m_b(j) = real_dist(mt);
		}

		m_mt.seed(mt());
	}

	void SetOptimizer(const NeuralNetOptimizer<T>* optimizer)
	{
		m_optimizer_W.reset(optimizer->Create(m_input_size * m_output_size));
		m_optimizer_b.reset(optimizer->Create(m_output_size));
	}

	void SetDecimateEnable(bool enable)
	{
		m_decimate_enable = enable;
	}

	INDEX GetDecimatedInputSize(void)       { return m_decimate_current_size; }
	void  SetDecimatedInputSize(INDEX size) { m_decimate_current_size = size; }


	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_input_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_output_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	T& W(INDEX output, INDEX input) { return m_W(input, output); }
	T& b(INDEX output) { return m_b(output); }
	T& dW(INDEX output, INDEX input) { return m_dW(input, output); }
	T& db(INDEX output) { return m_db(output); }

	void  SetBinaryMode(bool enable)
	{
		m_binary_mode = enable;
	}

	void  SetBatchSize(INDEX batch_size)
	{
		m_frame_size = batch_size;
	}

	void Forward(bool train = true)
	{
		// mask作成
		std::random_shuffle(m_mask_vec.begin(), m_mask_vec.end());
		for (INDEX output_node = 0; output_node < m_output_size; ++output_node) {
			for (INDEX input_node = 0; input_node < m_input_size; ++input_node) {
				m_Mask(input_node, output_node) = m_mask_vec[output_node*m_input_size + input_node] ? (T)1 : (T)0;
			}
		}

		for (INDEX output_node = 0; output_node < m_output_size; ++output_node) {
			for (INDEX input_node = 0; input_node < m_input_size; ++input_node) {
				auto& W = m_W(input_node, output_node);
				if (isnan(W)) {
					std::cout << "\n\nNaN\n\n" << std::endl;
					W = (T)0.000000;
				}
				if (W == 0) {
					W = (T)0.000001;
				}
			}
		}
		
//		Eigen::Map<Matrix> x((T*)m_input_signal_buffer.GetBuffer(), m_input_signal_buffer.GetFrameStride() / sizeof(T), m_input_size);
//		Eigen::Map<Matrix> y((T*)m_output_signal_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_output_size);
		MatMap x((T*)this->m_input_signal_buffer.GetBuffer(), m_frame_size, m_input_size, Stride(this->m_input_signal_buffer.GetFrameStride() / sizeof(T), 1));
		MatMap y((T*)this->m_output_signal_buffer.GetBuffer(), m_frame_size, m_output_size, Stride(this->m_output_signal_buffer.GetFrameStride() / sizeof(T), 1));
		
		Matrix W = m_W.array() * m_Mask.array();

//		y = x * m_W;
		y = x * W;
		y.rowwise() += m_b;
	}
	
	void Backward(void)
	{
//		Eigen::Map<Matrix> dy((T*)m_output_error_buffer.GetBuffer(), m_output_error_buffer.GetFrameStride() / sizeof(T), m_output_size);
//		Eigen::Map<Matrix> dx((T*)m_input_error_buffer.GetBuffer(), m_input_error_buffer.GetFrameStride() / sizeof(T), m_input_size);
//		Eigen::Map<Matrix> x((T*)m_input_signal_buffer.GetBuffer(), m_input_signal_buffer.GetFrameStride() / sizeof(T), m_input_size);
		MatMap dy((T*)this->m_output_error_buffer.GetBuffer(), m_frame_size, m_output_size, Stride(this->m_output_error_buffer.GetFrameStride() / sizeof(T), 1));
		MatMap dx((T*)this->m_input_error_buffer.GetBuffer(), m_frame_size, m_input_size, Stride(this->m_input_error_buffer.GetFrameStride() / sizeof(T), 1));
		MatMap x((T*)this->m_input_signal_buffer.GetBuffer(), m_frame_size, m_input_size, Stride(this->m_input_signal_buffer.GetFrameStride() / sizeof(T), 1));

//		Matrix dy2 = dy.array() * dy.array();
//		T err_sum = dy2.sum();
//		m_N      += m_Mask;
//		m_ErrSum += (m_Mask * err_sum);

		Matrix W = m_W.array() * m_Mask.array();

//		dx = dy * m_W.transpose();
		dx = dy * W.transpose();
		m_dW = x.transpose() * dy;
		m_db = dy.colwise().sum();

		Matrix dx2 = dx.array() * dx.array();
		m_ErrSum += dx2;
	}

	void Update(void)
	{
		m_optimizer_W->Update(m_W, m_dW);
		m_optimizer_b->Update(m_b, m_db);

		// バイナリモードでは (-1, +1) でクリップ
		if ( m_binary_mode ) {
			for (INDEX output_node = 0; output_node < m_output_size; ++output_node) {
				for (INDEX input_node = 0; input_node < m_input_size; ++input_node) {
					m_W(input_node, output_node) = std::min((T)+1, std::max((T)-1, m_W(input_node, output_node)));
				}
			}
		}

//		m_W *= 0.998;
//		m_b *= 0.998;

		if (!m_decimate_enable) {
			return;
		}

		// Sparse化した部分をマスク
		for (INDEX i = 0; i < m_output_size; ++i) {
			for (INDEX j = m_decimate_current_size; j < m_input_size; ++j) {
				m_W(m_sort_table[i][j], i) = 0;
			}
		}

		// 目的のサイズまでSparse化していれば何もしない
		if (m_decimate_current_size <= m_decimate_target_size) {
			return;
		}

		// 更新
		m_decimate_current_rate *= m_decimate_update_rate;
		m_decimate_current_size = (INDEX)(m_input_size * m_decimate_current_rate);
		if (m_decimate_current_size < m_decimate_target_size) {
			m_decimate_current_size = m_decimate_target_size;
		}

		// sort
		for (INDEX i = 0; i < m_output_size; ++i) {
			std::sort(m_sort_table[i].begin(), m_sort_table[i].end(), 
				[&](INDEX i0, INDEX i1) -> int {
				return abs(m_W(i0, i)) > abs(m_W(i1, i));
			});
		}
	}


	public:
		template <class Archive>
		void save(Archive &archive, std::uint32_t const version) const
		{
			archive(cereal::make_nvp("input_size", m_input_size));
			archive(cereal::make_nvp("output_size", m_output_size));

			std::vector< std::vector<T> >	W(m_output_size);
			std::vector<T>					b(m_output_size);
			for (INDEX i = 0; i < m_output_size; ++i) {
				W[i].resize(m_input_size);
				for (INDEX j = 0; j < m_input_size; ++j) {
					W[i][j] = m_W(j, i);
				}
				b[i] = m_b(i);
			}

			archive(cereal::make_nvp("W", W));
			archive(cereal::make_nvp("b", b));
		}

		template <class Archive>
		void load(Archive &archive, std::uint32_t const version)
		{
			archive(cereal::make_nvp("input_size", m_input_size));
			archive(cereal::make_nvp("output_size", m_output_size));

			std::vector< std::vector<T> >	W(m_output_size);
			std::vector<T>					b(m_output_size);
			archive(cereal::make_nvp("W", W));
			archive(cereal::make_nvp("b", b));

			for (INDEX i = 0; i < m_output_size; ++i) {
				W[i].resize(m_input_size);
				for (INDEX j = 0; j < m_input_size; ++j) {
					m_W(j, i) = W[i][j];
				}
				m_b(i) = b[i];
			}
		}

		virtual void Save(cereal::JSONOutputArchive& archive) const
		{
			archive(cereal::make_nvp("NeuralNetDenseAffine", *this));
		}

		virtual void Load(cereal::JSONInputArchive& archive)
		{
			archive(cereal::make_nvp("NeuralNetDenseAffine", *this));
		}
};

}