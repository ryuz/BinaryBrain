// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

#include "bb/NeuralNetLayerBuf.h"
#include "bb/NeuralNetOptimizerSgd.h"
#include "bb/SimdSupport.h"


namespace bb {


// NeuralNetの抽象クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNetBatchNormalizationAvx : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	INDEX		m_frame_size = 1;
	INDEX		m_node_size = 0;
	
	std::vector<T>	m_gamma;
	std::vector<T>	m_beta;
	std::vector<T>	m_dgamma;
	std::vector<T>	m_dbeta;

	std::unique_ptr< ParamOptimizer<T, INDEX> >	m_optimizer_gamma;
	std::unique_ptr< ParamOptimizer<T, INDEX> >	m_optimizer_beta;

	std::vector<T>	m_mean;		// 平均値
	std::vector<T>	m_rstd;		// 標準偏差の逆数

	T				m_momentum = (T)0.001;
	std::vector<T>	m_running_mean;
	std::vector<T>	m_running_var;

public:
	NeuralNetBatchNormalizationAvx() {}

	NeuralNetBatchNormalizationAvx(INDEX node_size, const NeuralNetOptimizer<T, INDEX>* optimizer = nullptr)
	{
		NeuralNetOptimizerSgd<T, INDEX> DefOptimizer;
		if (optimizer == nullptr) {
			optimizer = &DefOptimizer;
		}

		Resize(node_size);
		SetOptimizer(optimizer);
	}

	~NeuralNetBatchNormalizationAvx() {}		// デストラクタ

	std::string GetClassName(void) const { return "NeuralNetBatchNormalizationAvx"; }


	T& gamma(INDEX node) { return m_gamma(node); }
	T& beta(INDEX node) { return m_beta(node); }
	T& dgamma(INDEX node) { return m_dgamma(node); }
	T& dbeta(INDEX node) { return m_dbeta(node); }
	T& mean(INDEX node) { return m_running_mean(node); }
	T& var(INDEX node) { return m_running_var(node); }


	void Resize(INDEX node_size)
	{
		m_node_size = node_size;
		
		m_gamma.resize(m_node_size);
		std::fill(m_gamma.begin(), m_gamma.end(), (T)1.0);

		m_beta.resize(m_node_size);
		std::fill(m_beta.begin(), m_beta.end(), (T)0.0);

		m_dgamma.resize(m_node_size);
		std::fill(m_dgamma.begin(), m_dgamma.end(), (T)0.0);

		m_dbeta.resize(m_node_size);
		std::fill(m_dbeta.begin(), m_dbeta.end(), (T)0.0);

		m_mean.resize(m_node_size);
		m_rstd.resize(m_node_size);

		m_running_mean.resize(m_node_size);
		std::fill(m_running_mean.begin(), m_running_mean.end(), (T)0.0);

		m_running_var.resize(m_node_size);
		std::fill(m_running_var.begin(), m_running_var.end(), (T)1.0);
	}

	void  SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		m_optimizer_gamma.reset(optimizer->Create(m_node_size));
		m_optimizer_beta.reset(optimizer->Create(m_node_size));
	}

	void SetBatchSize(INDEX batch_size) {
		m_frame_size = batch_size;
	}

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	std::vector<T> CalcNode(INDEX node, std::vector<T> input_signals) const
	{
		std::vector<T> sig(input_signals.size());
		for (size_t i = 0; i < input_signals.size(); ++i) {
			sig[i] = input_signals[i];
			sig[i] -= m_running_mean[node];
			sig[i] /= (T)sqrt(m_running_var[node] + 10e-7);
			sig[i] = sig[i] * m_gamma[node] + m_beta[node];
		}
		return sig;
	}

public:
	void Forward(bool train = true)
	{
		if (typeid(T) == typeid(float)) {
			const int	mm256_frame_size = ((int)m_frame_size + 7) / 8 * 8;

			auto in_sig_buf = this->GetInputSignalBuffer();
			auto out_sig_buf = this->GetOutputSignalBuffer();

			if (train) {
				const __m256	reciprocal_frame_size = _mm256_set1_ps(1.0f / (float)m_frame_size);
				const __m256	epsilon = _mm256_set1_ps(10e-7f);

				#pragma omp parallel for
				for (int node = 0; node < (int)m_node_size; ++node) {
					float* x_ptr = (float*)in_sig_buf.GetPtr(node);
					float* y_ptr = (float*)out_sig_buf.GetPtr(node);

					// 平均と分散計算
#if 0
					__m256 mean0 = _mm256_set1_ps(0.0f);
					__m256 mean1 = _mm256_set1_ps(0.0f);
					__m256 var0 = _mm256_set1_ps(0.0f);
					__m256 var1 = _mm256_set1_ps(0.0f);
					int frame;
					for (frame = 0; frame < mm256_frame_size - 8; frame += 16) {
						__m256 x0 = _mm256_load_ps(&x_ptr[frame + 0]);
						mean0 = _mm256_add_ps(x0, mean0);
						var0 = _mm256_fmadd_ps(x0, x0, var0);
						__m256 x1 = _mm256_load_ps(&x_ptr[frame + 8]);
						mean1 = _mm256_add_ps(x1, mean1);
						var1 = _mm256_fmadd_ps(x1, x1, var1);
					}
					for (; frame < mm256_frame_size; frame += 8) {
						__m256 x0 = _mm256_load_ps(&x_ptr[frame + 0]);
						mean0 = _mm256_add_ps(x0, mean0);
						var0 = _mm256_fmadd_ps(x0, x0, var0);
					}
					__m256 mean = _mm256_mul_ps(bb_mm256_hsum_ps(_mm256_add_ps(mean0, mean1)), reciprocal_frame_size);
					__m256 var = bb_mm256_hsum_ps(_mm256_add_ps(var0, var1));
					var = _mm256_fmsub_ps(var, reciprocal_frame_size, _mm256_mul_ps(mean, mean));
					var = _mm256_max_ps(var, _mm256_set1_ps(0.0f));	// 誤差対策(負にならないようにクリップ)
#else
					__m256 mean_sum = _mm256_set1_ps(0.0f);
					__m256 mean_c   = _mm256_set1_ps(0.0f);
					__m256 var_sum  = _mm256_set1_ps(0.0f);
					__m256 var_c    = _mm256_set1_ps(0.0f);
					for ( int frame = 0; frame < mm256_frame_size; frame += 8) {
						__m256 x = _mm256_load_ps(&x_ptr[frame + 0]);
						__m256 mean_y = _mm256_sub_ps(x, mean_c);
						__m256 mean_t = _mm256_add_ps(mean_sum, mean_y);
						__m256 mean_c = _mm256_sub_ps(_mm256_sub_ps(mean_t, mean_sum), mean_y);
						mean_sum = mean_t;

						__m256 var_y = _mm256_fmsub_ps(x, x, var_c);
						__m256 var_t = _mm256_add_ps(var_sum, var_y);
						__m256 var_c = _mm256_sub_ps(_mm256_sub_ps(var_t, var_sum), var_y);
						var_sum = var_t;
					}
					__m256 mean = _mm256_mul_ps(bb_mm256_hsum_ps(mean_sum), reciprocal_frame_size);
					__m256 var = _mm256_fmsub_ps(bb_mm256_hsum_ps(var_sum), reciprocal_frame_size, _mm256_mul_ps(mean, mean));
					var = _mm256_max_ps(var, _mm256_set1_ps(0.0f));	// 誤差対策(負にならないようにクリップ)
#endif

					__m256 varx = _mm256_max_ps(var, epsilon);
					__m256 rstd = _mm256_rsqrt_ps(varx);

					varx = _mm256_mul_ps(varx, _mm256_set1_ps(0.5f));
					rstd = _mm256_mul_ps(rstd, _mm256_fnmadd_ps(varx, _mm256_mul_ps(rstd, rstd), _mm256_set1_ps(1.5f)));
					rstd = _mm256_mul_ps(rstd, _mm256_fnmadd_ps(varx, _mm256_mul_ps(rstd, rstd), _mm256_set1_ps(1.5f)));

					// 実行時の mean と var 保存
					m_running_mean[node] = m_running_mean[node] * m_momentum + bb_mm256_cvtss_f32(mean) * (1 - m_momentum);
					m_running_var[node] = m_running_var[node] * m_momentum + bb_mm256_cvtss_f32(var) * (1 - m_momentum);

					// 結果の保存
					m_mean[node] = bb_mm256_cvtss_f32(mean);
					m_rstd[node] = bb_mm256_cvtss_f32(rstd);

					// 正規化 と gamma/beta 処理
					__m256 gamma = _mm256_set1_ps(m_gamma[node]);
					__m256 beta = _mm256_set1_ps(m_beta[node]);
//					for (int frame = 0; frame < mm256_frame_size; frame += 8) {
					for (int frame = mm256_frame_size-8; frame >= 0; frame -= 8) {
					__m256 x = _mm256_load_ps(&x_ptr[frame]);
						__m256 xn = _mm256_mul_ps(_mm256_sub_ps(x, mean), rstd);
						__m256 y = _mm256_fmadd_ps(xn, gamma, beta);
						_mm256_store_ps(&y_ptr[frame], y);
					}
				}
			}
			else {
				#pragma omp parallel for
				for (int node = 0; node < (int)m_node_size; ++node) {
					float* x_ptr = (float*)in_sig_buf.GetPtr(node);
					float* y_ptr = (float*)out_sig_buf.GetPtr(node);

					__m256 running_mean = _mm256_set1_ps(m_running_mean[node]);
					__m256 running_var = _mm256_set1_ps(1.0f / sqrt(m_running_var[node] + 10e-7f));

					__m256 gamma = _mm256_set1_ps(m_gamma[node]);
					__m256 beta = _mm256_set1_ps(m_beta[node]);

					for (int frame = 0; frame < mm256_frame_size; frame += 8) {
						__m256 x = _mm256_load_ps(&x_ptr[frame]);
						__m256 xc = _mm256_sub_ps(x, running_mean);
						__m256 xn = _mm256_mul_ps(xc, running_var);
						__m256 y = _mm256_fmadd_ps(xn, gamma, beta);
						_mm256_store_ps(&y_ptr[frame], y);
					}
				}
			}

			out_sig_buf.ClearMargin();
		}
		else {

		}
	}

	void Backward(void)
	{
		if (typeid(T) == typeid(float)) {
			const int mm256_frame_size = ((int)m_frame_size + 7) / 8 * 8;
			// 逆数生成
			const __m256	reciprocal_frame_size = _mm256_set1_ps(1.0f / (float)m_frame_size);

			auto in_sig_buf = this->GetInputSignalBuffer();
			auto out_sig_buf = this->GetOutputSignalBuffer();
			auto in_err_buf = this->GetInputErrorBuffer();
			auto out_err_buf = this->GetOutputErrorBuffer();
			#pragma omp parallel for
			for (int node = 0; node < (int)m_node_size; ++node) {
				float* dy_ptr = (float*)out_err_buf.GetPtr(node);
				float* dx_ptr = (float*)in_err_buf.GetPtr(node);
				float* x_ptr = (float*)in_sig_buf.GetPtr(node);

				__m256 mean   = _mm256_set1_ps(m_mean[node]);
				__m256 rstd   = _mm256_set1_ps(m_rstd[node]);
				__m256 gamma  = _mm256_set1_ps(m_gamma[node]);
				__m256 dbeta  = _mm256_set1_ps(0);
				__m256 dgamma = _mm256_set1_ps(0);
				__m256 dstd = _mm256_set1_ps(0);
				__m256 dmeanx = _mm256_set1_ps(0);
				__m256 rstd2 = _mm256_mul_ps(rstd, rstd);

				for (int frame = 0; frame < mm256_frame_size; frame += 8) {
					__m256 x = _mm256_load_ps(&x_ptr[frame]);
					__m256 xc = _mm256_sub_ps(x, mean);
					__m256 xn = _mm256_mul_ps(xc, rstd);

					__m256 dy = _mm256_load_ps(&dy_ptr[frame]);
					dbeta = _mm256_add_ps(dy, dbeta);
					dgamma = _mm256_fmadd_ps(xn, dy, dgamma);

					__m256 dxn = _mm256_mul_ps(dy, gamma);
					dstd = _mm256_fnmadd_ps(_mm256_mul_ps(dxn, xc), rstd2, dstd);
					dmeanx = _mm256_fnmadd_ps(dxn, rstd, dmeanx);
				}
				dbeta = bb_mm256_hsum_ps(dbeta);
				dgamma = bb_mm256_hsum_ps(dgamma);
				m_dgamma[node] = bb_mm256_cvtss_f32(dgamma);
				m_dbeta[node] = bb_mm256_cvtss_f32(dbeta);

				dstd = bb_mm256_hsum_ps(dstd);
				dmeanx = bb_mm256_hsum_ps(dmeanx);

				__m256 dvar  = _mm256_mul_ps(dstd, rstd);
				__m256 dmean = _mm256_mul_ps(_mm256_fnmadd_ps(mean, dvar, dmeanx), reciprocal_frame_size);

//				for (int frame = 0; frame < mm256_frame_size; frame += 8) {
				for (int frame = mm256_frame_size - 8; frame >= 0; frame -= 8) {
					__m256 dy = _mm256_load_ps(&dy_ptr[frame]);
					__m256 x = _mm256_load_ps(&x_ptr[frame]);
					__m256 dxn = _mm256_mul_ps(dy, gamma);
					__m256 dxc = _mm256_fmadd_ps(dxn, rstd, dmean);
					__m256 dx = _mm256_fmadd_ps(_mm256_mul_ps(x, dvar), reciprocal_frame_size, dxc);
					_mm256_store_ps(&dx_ptr[frame], dx);
				}
			}

			in_err_buf.ClearMargin();
		}
		else {
		}
	}

	void Update(void)
	{
		// update
		m_optimizer_gamma->Update(m_gamma, m_dgamma);
		m_optimizer_beta->Update(m_beta, m_dbeta);
	}

public:
	// Serialize
	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
		archive(cereal::make_nvp("gamma", m_gamma));
		archive(cereal::make_nvp("beta", m_beta));
		archive(cereal::make_nvp("running_mean", m_running_mean));
		archive(cereal::make_nvp("running_var", m_running_var));
	}

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("gamma", m_gamma));
		archive(cereal::make_nvp("beta", m_beta));
		archive(cereal::make_nvp("running_mean", m_running_mean));
		archive(cereal::make_nvp("running_var", m_running_var));
	}

	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		archive(cereal::make_nvp("NeuralNetBatchNormalizationAvx", *this));
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		archive(cereal::make_nvp("NeuralNetBatchNormalizationAvx", *this));
	}

};

}
