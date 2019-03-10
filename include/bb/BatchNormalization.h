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

#include "bb/DataType.h"
#include "bb/Activation.h"
#include "bb/FrameBuffer.h"
#include "bb/SimdSupport.h"


namespace bb {


// BatchNormalization
template <typename T = float>
class BatchNormalization : public Activation<T, T>
{
    using _super = Activation<T, T>;

protected:
    index_t 		            m_node_size;
    
    FrameBuffer                 m_x;
    FrameBuffer                 m_y;
    FrameBuffer                 m_dx;

    std::shared_ptr<Tensor>  	m_gamma;
    std::shared_ptr<Tensor>    	m_beta;
    std::shared_ptr<Tensor>	    m_dgamma;
    std::shared_ptr<Tensor>  	m_dbeta;

    Tensor_<T>  	            m_mean;		// 平均値
    Tensor_<T>  	            m_rstd;		// 標準偏差の逆数

    T				            m_momentum = (T)0.001;
    Tensor_<T> 	                m_running_mean;
    Tensor_<T>  	            m_running_var;

protected:
    BatchNormalization() {
        m_gamma  = std::make_shared<Tensor>();
        m_beta   = std::make_shared<Tensor>();
        m_dgamma = std::make_shared<Tensor>();
        m_dbeta  = std::make_shared<Tensor>();
    }

public:
    ~BatchNormalization() {}

    struct create_t
    {
        T   momentum = (T)0.001;
    };

    static std::shared_ptr<BatchNormalization> Create(create_t const &create)
    {
        auto self = std::shared_ptr<BatchNormalization>(new BatchNormalization);
        self->m_momentum = create.momentum;
        return self;
    }

    static std::shared_ptr<BatchNormalization> Create(T momentum = (T)0.001)
    {
        auto self = std::shared_ptr<BatchNormalization>(new BatchNormalization);
        self->m_momentum = momentum;
        return self;
    }

    std::string GetClassName(void) const { return "BatchNormalization"; }



    // Serialize
    void Save(std::ostream &os) const 
    {
        SaveIndex(os, m_node_size);
        bb::Save(os, m_momentum);
        m_gamma->Save(os);
        m_beta->Save(os);
        m_running_mean.Save(os);
        m_running_var.Save(os);
    }

    void Load(std::istream &is)
    {
        m_node_size = LoadIndex(is);
        bb::Load(is, m_momentum);
        m_gamma->Load(is);
        m_beta->Load(is);
        m_running_mean.Load(is);
        m_running_var.Load(is);
    }


#ifdef BB_WITH_CEREAL
	template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
	{
        _super::save(archive, version);
        archive(cereal::make_nvp("node_size",    m_node_size));
        archive(cereal::make_nvp("gamma",        *m_gamma));
        archive(cereal::make_nvp("beta",         *m_beta));
        archive(cereal::make_nvp("running_mean", m_running_mean));
        archive(cereal::make_nvp("running_var",  m_running_var));
    }

	template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
	{
        _super::load(archive, version);
        archive(cereal::make_nvp("node_size",    m_node_size));
        archive(cereal::make_nvp("gamma",        *m_gamma));
        archive(cereal::make_nvp("beta",         *m_beta));
        archive(cereal::make_nvp("running_mean", m_running_mean));
        archive(cereal::make_nvp("running_var",  m_running_var));
     }

	void Save(cereal::JSONOutputArchive& archive) const
	{
        archive(cereal::make_nvp("BatchNormalization", *this));
	}

	void Load(cereal::JSONInputArchive& archive)
	{
        archive(cereal::make_nvp("BatchNormalization", *this));
	}
#endif





    auto lock_gamma(void)              { return m_gamma->GetPtr<T>(); }
    auto lock_gamma_const(void)  const { return m_gamma->LockConst<T>(); }
    auto lock_beta(void)               { return m_beta->GetPtr<T>(); }
    auto lock_beta_const(void)   const { return m_beta->LockConst<T>(); }
    auto lock_dgamma(void)             { return m_dgamma->GetPtr<T>(); }
    auto lock_dgamma_const(void) const { return m_dgamma->LockConst<T>(); }
    auto lock_dbeta(void)              { return m_dbeta->GetPtr<T>(); }
    auto lock_dbeta_const(void)  const { return m_dbeta->LockConst<T>(); }
    auto lock_mean(void)               { return m_running_mean.GetPtr(); }
    auto lock_mean_const(void)   const { return m_running_mean.LockConst(); }
    auto lock_var(void)                { return m_running_var.GetPtr(); }
    auto lock_var_const(void)    const { return m_running_var.LockConst(); }


    /**
     * @brief  入力形状設定
     * @detail 入力形状を設定する
     *         内部変数を初期化し、以降、GetOutputShape()で値取得可能となることとする
     *         同一形状を指定しても内部変数は初期化されるものとする
     * @param  shape      1フレームのノードを構成するshape
     * @return 出力形状を返す
     */
    indices_t SetInputShape(indices_t shape)
    {
        _super::SetInputShape(shape);

        m_node_size = GetShapeSize(shape);
        
        // パラメータ初期化
        m_gamma->Resize(DataType<T>::type, m_node_size);    *m_gamma  = (T)1.0;
        m_beta->Resize(DataType<T>::type, m_node_size);     *m_beta   = (T)0.0;
        m_dgamma->Resize(DataType<T>::type, m_node_size);   *m_dgamma = (T)0.0;
        m_dbeta->Resize(DataType<T>::type, m_node_size);    *m_dbeta  = (T)0.0;

        m_mean.Resize(m_node_size);
        m_rstd.Resize(m_node_size);

        m_running_mean.Resize(m_node_size); m_running_mean = (T)0.0;
        m_running_var.Resize(m_node_size);  m_running_var  = (T)1.0;

        return shape;
    }


public:
   /**
     * @brief  パラメータ取得
     * @detail パラメータを取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetParameters(void)
    {
        Variables parameters;
        parameters.PushBack(m_gamma);
        parameters.PushBack(m_beta);
        return parameters;
    }

    /**
     * @brief  勾配取得
     * @detail 勾配を取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetGradients(void)
    {
        Variables gradients;
        gradients.PushBack(m_dgamma);
        gradients.PushBack(m_dbeta);
        return gradients;
    }
    

    // ノード単位でのForward計算
    std::vector<T> ForwardNode(index_t node, std::vector<T> x_vec) const
 	{
        BB_DEBUG_ASSERT(node >= 0 && node < m_node_size);

        auto gamma_ptr        = lock_gamma_const();
        auto beta_ptr         = lock_beta_const();
        auto running_mean_ptr = m_running_mean.LockConst();
        auto running_var_ptr  = m_running_var.LockConst();

        std::vector<T> y_vec(x_vec.size());
		for (size_t i = 0; i < x_vec.size(); ++i) {
			y_vec[i]  = x_vec[i];
			y_vec[i] -= running_mean_ptr(node);
			y_vec[i] /= (T)sqrt(running_var_ptr(node) + 10e-7);
			y_vec[i]  = y_vec[i] * gamma_ptr(node) + beta_ptr(node);
		}
		return y_vec;
	}


    /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    FrameBuffer Forward(FrameBuffer x, bool train=true)
    {
        // forwardの為に保存
        m_x = x;

        // 出力設定
        m_y.Resize(x.GetType(), x.GetFrameSize(), x.GetNodeSize());

        auto frame_size = x.GetFrameSize();
        
        const int	mm256_frame_size = ((int)frame_size + 7) / 8 * 8;

        auto x_buf_ptr = m_x.LockConst<T>();
        auto y_buf_ptr = m_y.GetPtr<T>();

        auto gamma_ptr        = lock_gamma_const();
        auto beta_ptr         = lock_beta_const();

        auto mean_ptr         = m_mean.GetPtr();
        auto rstd_ptr         = m_rstd.GetPtr();        
        auto running_mean_ptr = m_running_mean.GetPtr();
        auto running_var_ptr  = m_running_var.GetPtr();

        if (train) {
            const __m256	reciprocal_frame_size = _mm256_set1_ps(1.0f / (float)frame_size);
            const __m256	epsilon = _mm256_set1_ps(10e-7f);

		  	#pragma omp parallel for
            for (int node = 0; node < (int)m_node_size; ++node) {
                float const *x_ptr = x_buf_ptr.GetAddr(node);
                float       *y_ptr = y_buf_ptr.GetAddr(node);

                // 平均と分散計算
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

                __m256 varx = _mm256_max_ps(var, epsilon);
                __m256 rstd = _mm256_rsqrt_ps(varx);

                varx = _mm256_mul_ps(varx, _mm256_set1_ps(0.5f));
                rstd = _mm256_mul_ps(rstd, _mm256_fnmadd_ps(varx, _mm256_mul_ps(rstd, rstd), _mm256_set1_ps(1.5f)));
                rstd = _mm256_mul_ps(rstd, _mm256_fnmadd_ps(varx, _mm256_mul_ps(rstd, rstd), _mm256_set1_ps(1.5f)));

                // 実行時の mean と var 保存
                running_mean_ptr[node] = running_mean_ptr[node] * m_momentum + bb_mm256_cvtss_f32(mean) * (1 - m_momentum);
                running_var_ptr[node]  = running_var_ptr[node] * m_momentum + bb_mm256_cvtss_f32(var) * (1 - m_momentum);

                // 結果の保存
                mean_ptr[node] = bb_mm256_cvtss_f32(mean);
                rstd_ptr[node] = bb_mm256_cvtss_f32(rstd);

                // 正規化 と gamma/beta 処理
                __m256 gamma = _mm256_set1_ps(gamma_ptr[node]);
                __m256 beta = _mm256_set1_ps(beta_ptr[node]);
//				for (int frame = 0; frame < mm256_frame_size; frame += 8) {
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
                auto x_ptr = x_buf_ptr.GetAddr(node);
                auto y_ptr = y_buf_ptr.GetAddr(node);

                __m256 running_mean = _mm256_set1_ps(running_mean_ptr[node]);
                __m256 running_var = _mm256_set1_ps(1.0f / sqrt(running_var_ptr[node] + 10e-7f));

                __m256 gamma = _mm256_set1_ps(gamma_ptr[node]);
                __m256 beta = _mm256_set1_ps(beta_ptr[node]);

                for (int frame = 0; frame < mm256_frame_size; frame += 8) {
                    __m256 x = _mm256_load_ps(&x_ptr[frame]);
                    __m256 xc = _mm256_sub_ps(x, running_mean);
                    __m256 xn = _mm256_mul_ps(xc, running_var);
                    __m256 y = _mm256_fmadd_ps(xn, gamma, beta);
                    _mm256_store_ps(&y_ptr[frame], y);
                }
            }
        }

        return m_y;
    }


    /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    FrameBuffer Backward(FrameBuffer dy)
    {
        // 出力設定
        m_dx.Resize(dy.GetType(), dy.GetFrameSize(), dy.GetNodeSize());

        auto frame_size = dy.GetFrameSize();
        
        const int	mm256_frame_size = ((int)frame_size + 7) / 8 * 8;

 
        auto gamma_ptr        = lock_gamma_const();
        auto beta_ptr         = lock_beta_const();
        auto dgamma_ptr       = lock_dgamma();
        auto dbeta_ptr        = lock_dbeta();

        auto mean_ptr         = m_mean.GetPtr();
        auto rstd_ptr         = m_rstd.GetPtr();        
        auto running_mean_ptr = m_running_mean.GetPtr();
        auto running_var_ptr  = m_running_var.GetPtr();
        
        
        // 逆数生成
        const __m256	reciprocal_frame_size = _mm256_set1_ps(1.0f / (float)frame_size);

//		auto in_sig_buf = this->GetInputSignalBuffer();
//		auto out_sig_buf = this->GetOutputSignalBuffer();
//		auto in_err_buf = this->GetInputErrorBuffer();
//		auto out_err_buf = this->GetOutputErrorBuffer();

        auto x_buf_ptr  = m_x.LockConst<T>();
        auto y_buf_ptr  = m_y.LockConst<T>();
        auto dx_buf_ptr = m_dx.GetPtr<T>();
        auto dy_buf_ptr = dy.LockConst<T>();

        #pragma omp parallel for
        for (int node = 0; node < (int)m_node_size; ++node) {
            auto dy_ptr = dy_buf_ptr.GetAddr(node);
            auto dx_ptr = dx_buf_ptr.GetAddr(node);
            auto x_ptr  = x_buf_ptr.GetAddr(node);

            __m256 mean   = _mm256_set1_ps(mean_ptr[node]);
            __m256 rstd   = _mm256_set1_ps(rstd_ptr[node]);
            __m256 gamma  = _mm256_set1_ps(gamma_ptr[node]);
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
            dgamma_ptr[node] = bb_mm256_cvtss_f32(dgamma);
            dbeta_ptr[node] = bb_mm256_cvtss_f32(dbeta);

            dstd = bb_mm256_hsum_ps(dstd);
            dmeanx = bb_mm256_hsum_ps(dmeanx);

            __m256 dvar  = _mm256_mul_ps(dstd, rstd);
            __m256 dmean = _mm256_mul_ps(_mm256_fnmadd_ps(mean, dvar, dmeanx), reciprocal_frame_size);

//			for (int frame = 0; frame < mm256_frame_size; frame += 8) {
            for (int frame = mm256_frame_size - 8; frame >= 0; frame -= 8) {
                __m256 dy = _mm256_load_ps(&dy_ptr[frame]);
                __m256 x = _mm256_load_ps(&x_ptr[frame]);
                __m256 dxn = _mm256_mul_ps(dy, gamma);
                __m256 dxc = _mm256_fmadd_ps(dxn, rstd, dmean);
                __m256 dx = _mm256_fmadd_ps(_mm256_mul_ps(x, dvar), reciprocal_frame_size, dxc);
                _mm256_store_ps(&dx_ptr[frame], dx);
            }
        }

//      in_err_buf.ClearMargin();

        return m_dx;
    }


    #if 0

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
#endif
};

}
