// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

//#ifndef EIGEN_MPL2_ONLY
//#define EIGEN_MPL2_ONLY
//#endif

#include <Eigen/Core>

#include "NeuralNetLayerBuf.h"
#include "NeuralNetOptimizerSgd.h"

namespace bb {


// NeuralNetの抽象クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNetBatchNormalizationAvx : public NeuralNetLayerBuf<T, INDEX>
{
protected:
	typedef Eigen::Matrix<T, -1, -1, Eigen::ColMajor>	Matrix;
	typedef Eigen::Matrix<T, 1, -1>						Vector;

	INDEX		m_mux_size = 1;
	INDEX		m_frame_size = 1;
	INDEX		m_node_size = 0;
	
	std::vector<T>	m_gamma;
	std::vector<T>	m_beta;
	std::vector<T>	m_dgamma;
	std::vector<T>	m_dbeta;

	std::unique_ptr< ParamOptimizer<T, INDEX> >	m_optimizer_gamma;
	std::unique_ptr< ParamOptimizer<T, INDEX> >	m_optimizer_beta;

	Matrix		m_xn;
	Matrix		m_xc;

	std::vector<T>	m_mean;		// 平均値
	std::vector<T>	m_rstd;		// 標準偏差の逆数

	T				m_momentum = (T)0.01;
	std::vector<T>	m_running_mean;
	std::vector<T>	m_running_var;

public:
	NeuralNetBatchNormalizationAvx() {}

	NeuralNetBatchNormalizationAvx(INDEX node_size, const NeuralNetOptimizer<T, INDEX>* optimizer = &NeuralNetOptimizerSgd<>())
	{
		Resize(node_size);
		SetOptimizer(optimizer);
	}

	~NeuralNetBatchNormalizationAvx() {}		// デストラクタ


	T& gamma(INDEX node) { return m_gamma(node); }
	T& beta(INDEX node) { return m_beta(node); }
	T& dgamma(INDEX node) { return m_dgamma(node); }
	T& dbeta(INDEX node) { return m_dbeta(node); }
	T& mean(INDEX node) { return m_running_mean(node); }
	T& var(INDEX node) { return m_running_var(node); }


	void Resize(INDEX node_size)
	{
		m_node_size = node_size;
	//	m_gamma = Vector::Ones(m_node_size);
	//	m_beta = Vector::Zero(m_node_size);
	//	m_dgamma = Vector::Zero(m_node_size);
	//	m_dbeta = Vector::Zero(m_node_size);
	//	m_running_mean = Vector::Zero(m_node_size);
	//	m_running_var = Vector::Ones(m_node_size);

		m_mean.resize(m_node_size);
		m_rstd.resize(m_node_size);


		m_gamma.resize(m_node_size);
		std::fill(m_gamma.begin(), m_gamma.end(), (T)1.0);

		m_beta.resize(m_node_size);
		std::fill(m_beta.begin(), m_beta.end(), (T)0.0);

		m_dgamma.resize(m_node_size);
		std::fill(m_dgamma.begin(), m_dgamma.end(), (T)0.0);

		m_dbeta.resize(m_node_size);
		std::fill(m_dbeta.begin(), m_dbeta.end(), (T)0.0);


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

	void  SetMuxSize(INDEX mux_size) {
		m_mux_size = mux_size;
	}

	void SetBatchSize(INDEX batch_size) {
		m_frame_size = batch_size * m_mux_size;
	}

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

	T CalcNode(INDEX node, std::vector<T> input_signals) const
	{
		T sig = input_signals[0];
		sig -= m_running_mean[node];
		sig /= (T)sqrt(m_running_var[node] + 10e-7);
		sig = sig * m_gamma[node] + m_beta[node];
		return sig;
	}

protected:
	inline static __m256 my_mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
	{
#ifdef __FMA__
		return _mm256_fmadd_ps(a, b, c);
#else
		return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
	}

	inline static __m256 my_mm256_hsum_ps(__m256 r)
	{
		r = _mm256_hadd_ps(r, r);
		r = _mm256_hadd_ps(r, r);
		__m256 tmp = _mm256_permute2f128_ps(r, r, 0x1);
		r = _mm256_unpacklo_ps(r, tmp);
		return _mm256_hadd_ps(r, r);
	}

	// 水平加算
	inline static __m256 my_horizontal_sum(const float* ptr, int size)
	{
		__m256 sum0 = _mm256_set1_ps(0.0f);
		__m256 sum1 = _mm256_set1_ps(0.0f);
		__m256 sum2 = _mm256_set1_ps(0.0f);
		__m256 sum3 = _mm256_set1_ps(0.0f);
		int i;
		for (i = 0; i < size - 3; i += 4) {
			// パイプラインを埋めるため並列化
			sum0 = _mm256_add_ps(sum0, _mm256_load_ps(&ptr[i + 0]));
			sum1 = _mm256_add_ps(sum1, _mm256_load_ps(&ptr[i + 1]));
			sum2 = _mm256_add_ps(sum2, _mm256_load_ps(&ptr[i + 2]));
			sum3 = _mm256_add_ps(sum3, _mm256_load_ps(&ptr[i + 3]));
		}
		for (; i < size; ++i) {
			// 端数処理
			sum0 = _mm256_add_ps(sum0, _mm256_load_ps(&ptr[i]));
		}
		sum0 = _mm256_add_ps(sum0, sum1);
		sum2 = _mm256_add_ps(sum2, sum3);
		return my_mm256_hsum_ps(_mm256_add_ps(sum0, sum2));
	}

	// 分散計算
	inline static __m256 my_horizontal_serr_sum(const float* ptr, int size, __m256 mean)
	{
		__m256 sum0 = _mm256_set1_ps(0.0f);
		__m256 sum1 = _mm256_set1_ps(0.0f);
		__m256 sum2 = _mm256_set1_ps(0.0f);
		__m256 sum3 = _mm256_set1_ps(0.0f);
		int	i;
		// キャッシュが当たりやすいように、逆走査
		for ( i = size; i > 3; i -= 4) {
			__m256 diff0 = _mm256_sub_ps(_mm256_load_ps(&ptr[i - 4]), mean);
			sum0 = my_mm256_fmadd_ps(diff0, diff0, sum0);

			__m256 diff1 = _mm256_sub_ps(_mm256_load_ps(&ptr[i - 3]), mean);
			sum1 = my_mm256_fmadd_ps(diff1, diff1, sum1);

			__m256 diff2 = _mm256_sub_ps(_mm256_load_ps(&ptr[i - 2]), mean);
			sum2 = my_mm256_fmadd_ps(diff2, diff2, sum2);

			__m256 diff3 = _mm256_sub_ps(_mm256_load_ps(&ptr[i - 1]), mean);
			sum2 = my_mm256_fmadd_ps(diff3, diff3, sum2);
		}
		for (; i > 0; --i) {
			__m256 diff0 = _mm256_sub_ps(_mm256_load_ps(&ptr[i - 1]), mean);
			sum0 = my_mm256_fmadd_ps(diff0, diff0, sum0);
		}
		sum0 = _mm256_add_ps(sum0, sum1);
		sum2 = _mm256_add_ps(sum2, sum3);
		return my_mm256_hsum_ps(_mm256_add_ps(sum0, sum2));
	}

public:
	void Forward(bool train = true)
	{
		if (typeid(T) == typeid(float)) {
			int		mm256_frame_size = ((int)m_frame_size + 7) / 8 * 8;

			const __m256	epsilon = _mm256_set1_ps(10e-7f);

			// 逆数生成
			const __m256	reciprocal_frame_size = _mm256_set1_ps(1.0f / (float)m_frame_size);

#if 0
			// 境界マスク生成
			__m256i	border_mask_i;
			if (m_frame_size % 8 == 0) {
				border_mask_i = _mm256_set1_epi32(0);
			}
			else {
				for (int i = 0; i < 8; ++i) {
					border_mask_i.m256i_i32[i] = (i < m_frame_size % 8) ? 0 : -1;
				}
			}
			__m256	border_mask = _mm256_castsi256_ps(border_mask_i);
#endif

			auto in_sig_buf = GetInputSignalBuffer();
			auto out_sig_buf = GetOutputSignalBuffer();

			for (int node = 0; node < (int)m_node_size; ++node) {
				float* x_ptr = (float*)in_sig_buf.GetPtr(node);
				float* y_ptr = (float*)out_sig_buf.GetPtr(node);

#if 0
				// mean
				__m256 mean = _mm256_mul_ps(my_horizontal_sum(x_ptr, mm256_frame_size / 8), reciprocal_frame_size);
				
				// 端数を平均で埋める
				__m256 border = _mm256_load_ps(&x_ptr[mm256_frame_size - 8]);
				border = _mm256_blendv_ps(border, mean, border_mask);
				_mm256_store_ps(&x_ptr[mm256_frame_size - 8], border);

				// 分散と偏差を求める
				__m256 var  = _mm256_mul_ps(my_horizontal_serr_sum(x_ptr, mm256_frame_size / 8, mean), reciprocal_frame_size);
				__m256 rstd = _mm256_rsqrt_ps(_mm256_add_ps(var, epsilon));
#endif

				__m256 mean0 = _mm256_set1_ps(0.0f);
				__m256 mean1 = _mm256_set1_ps(0.0f);
				__m256 var0 = _mm256_set1_ps(0.0f);
				__m256 var1 = _mm256_set1_ps(0.0f);
				int frame;
				for ( frame = 0; frame < mm256_frame_size-8; frame += 16) {
					__m256 x0 = _mm256_load_ps(&x_ptr[frame + 0]);
					mean0 = _mm256_add_ps(x0, mean0);
					var0  = _mm256_fmadd_ps(x0, x0, var0);
					__m256 x1 = _mm256_load_ps(&x_ptr[frame + 8]);
					mean1 = _mm256_add_ps(x1, mean1);
					var1 = _mm256_fmadd_ps(x1, x1, var1);
				}
				for (; frame < mm256_frame_size; frame += 8) {
					__m256 x0 = _mm256_load_ps(&x_ptr[frame + 0]);
					mean0 = _mm256_add_ps(x0, mean0);
					var0 = _mm256_fmadd_ps(x0, x0, var0);
				}
				__m256 mean = _mm256_mul_ps(my_mm256_hsum_ps(_mm256_add_ps(mean0, mean1)), reciprocal_frame_size);
				__m256 var  = my_mm256_hsum_ps(_mm256_add_ps(var0, var1));
				var = _mm256_fmsub_ps(var, reciprocal_frame_size, _mm256_mul_ps(mean, mean));
				
				__m256 varx = _mm256_add_ps(var, epsilon);
				__m256 rstd = _mm256_rsqrt_ps(varx);
				varx = _mm256_mul_ps(varx, _mm256_set1_ps(0.5f));
				rstd = _mm256_mul_ps(rstd, _mm256_fnmadd_ps(varx, _mm256_mul_ps(rstd, rstd), _mm256_set1_ps(1.5f)));
				rstd = _mm256_mul_ps(rstd, _mm256_fnmadd_ps(varx, _mm256_mul_ps(rstd, rstd), _mm256_set1_ps(1.5f)));

				// 正規化 と gamma/beta 処理
				__m256 gamma = _mm256_set1_ps(m_gamma[node]);
				__m256 beta = _mm256_set1_ps(m_beta[node]);
				for (int frame = 0; frame < mm256_frame_size; frame += 8) {
					__m256 x = _mm256_load_ps(&x_ptr[frame]);
					__m256 xn = _mm256_mul_ps(_mm256_sub_ps(x, mean), rstd);
					__m256 y = my_mm256_fmadd_ps(xn, gamma, beta);
					_mm256_store_ps(&y_ptr[frame], y);
				}

				// 実行時の mean と var 保存
				m_running_mean[node] = m_running_mean[node] * m_momentum + mean.m256_f32[0] * (1 - m_momentum);
				m_running_var[node] = m_running_var[node] * m_momentum + var.m256_f32[0] * (1 - m_momentum);

				// 結果の保存
				m_mean[node] = mean.m256_f32[0];
				m_rstd[node] = rstd.m256_f32[0];
			}
		}
		else {
#if 0
			Eigen::Map<Matrix> x((T*)m_input_signal_buffer.GetBuffer(), m_input_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
			Eigen::Map<Matrix> y((T*)m_output_signal_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);

			Matrix xc;
			Matrix xn;

			if (train) {
				Vector mu = x.colwise().mean();
				xc = x.rowwise() - mu;
				Vector var = (xc.array() * xc.array()).colwise().mean();
				Vector std = (var.array() + (T)10e-7).array().sqrt();
				xn = xc.array().rowwise() / std.array();
				m_xn = xn;
				m_xc = xc;
				m_std = std;
				m_running_mean = m_running_mean * m_momentum + mu * (1 - m_momentum);
				m_running_var = m_running_var * m_momentum + var * (1 - m_momentum);
			}
			else {
				xc = x.rowwise() - m_running_mean;
				xn = xc.array().rowwise() / (m_running_var.array() + 10e-7).array().sqrt();
			}
			y = (xn.array().rowwise() * m_gamma.array()).array().rowwise() + m_beta.array();
#endif
		}
	}

	void Backward(void)
	{
		if (typeid(T) == typeid(float)) {
			const int mm256_frame_size = ((int)m_frame_size + 7) / 8 * 8;
			// 逆数生成
			const __m256	reciprocal_frame_size = _mm256_set1_ps(1.0f / (float)m_frame_size);

			auto in_sig_buf = GetInputSignalBuffer();
			auto out_sig_buf = GetOutputSignalBuffer();
			auto in_err_buf = GetInputErrorBuffer();
			auto out_err_buf = GetOutputErrorBuffer();

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
				dbeta = my_mm256_hsum_ps(dbeta);
				dgamma = my_mm256_hsum_ps(dgamma);
				m_dgamma[node] = dgamma.m256_f32[0];
				m_dbeta[node] = dbeta.m256_f32[0];

				dstd = my_mm256_hsum_ps(dstd);
				dmeanx = my_mm256_hsum_ps(dmeanx);

				__m256 dvar  = _mm256_mul_ps(dstd, rstd);
				__m256 dmean = _mm256_mul_ps(_mm256_fnmadd_ps(mean, dvar, dmeanx), reciprocal_frame_size);

				for (int frame = 0; frame < mm256_frame_size; frame += 8) {
					__m256 dy = _mm256_load_ps(&dy_ptr[frame]);
					__m256 x = _mm256_load_ps(&x_ptr[frame]);
					__m256 dxn = _mm256_mul_ps(dy, gamma);
					__m256 dxc = _mm256_fmadd_ps(dxn, rstd, dmean);
					__m256 dx = _mm256_fmadd_ps(_mm256_mul_ps(x, dvar), reciprocal_frame_size, dxc);
					_mm256_store_ps(&dx_ptr[frame], dx);
				}
			}
		}
		else {
#if 0
			Eigen::Map<Matrix> y((T*)m_output_signal_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
			Eigen::Map<Matrix> dy((T*)m_output_error_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
			Eigen::Map<Matrix> dx((T*)m_input_error_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_node_size);
			INDEX frame_szie = GetOutputFrameSize();

			Vector dbeta = dy.colwise().sum();
			Vector dgamma = (m_xn.array() * dy.array()).colwise().sum();
			Matrix dxn = dy.array().rowwise() * m_gamma.array();
			Matrix dxc = dxn.array().rowwise() / m_std.array();
			Vector dstd = -((dxn.array() * m_xc.array()).array().rowwise() / (m_std.array() * m_std.array()).array()).array().colwise().sum();
			Vector dvar = m_std.array().inverse() * dstd.array() * (T)0.5;
			dxc = dxc.array() + (m_xc.array().rowwise() * dvar.array() * ((T)2.0 / (T)frame_szie)).array();

			Vector dmu = dxc.colwise().sum();
			dx = dxc.array().rowwise() - (dmu.array() / (T)frame_szie);

			m_dgamma = dgamma;
			m_dbeta = dbeta;
#endif
		}
	}

	void Update(void)
	{
		// update
		m_optimizer_gamma->Update(m_gamma, m_dgamma);
		m_optimizer_beta->Update(m_beta, m_dbeta);

#if 0
		std::vector<T> vec_gamma(m_node_size);
		std::vector<T> vec_dgamma(m_node_size);
		std::vector<T> vec_beta(m_node_size);
		std::vector<T> vec_dbeta(m_node_size);

		// copy
		for (INDEX node = 0; node < m_node_size; ++node) {
			vec_gamma[node] = m_gamma(node);
			vec_dgamma[node] = m_dgamma(node);
			vec_beta[node] = m_beta(node);
			vec_dbeta[node] = m_dbeta(node);
		}

		// update
		m_optimizer_gamma->Update(vec_gamma, vec_dgamma);
		m_optimizer_beta->Update(vec_beta, vec_dbeta);

		// copy back
		for (INDEX node = 0; node < m_node_size; ++node) {
			m_gamma(node) = vec_gamma[node];
			m_beta(node) = vec_beta[node];
		}
#endif

//		m_gamma -= m_dgamma * learning_rate;
//		m_beta -= m_dbeta * learning_rate;

		// clear
//		m_dgamma *= 0;
//		m_dbeta *= 0;
	}

};

}
