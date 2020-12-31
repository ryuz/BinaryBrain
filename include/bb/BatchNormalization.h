﻿// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdlib>

#ifdef BB_WITH_CEREAL
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#endif

#include "bb/Manager.h"
#include "bb/DataType.h"
#include "bb/Model.h"
#include "bb/Activation.h"
#include "bb/FrameBuffer.h"
#include "bb/SimdSupport.h"

#ifdef BB_WITH_CUDA
#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#endif


namespace bb {


// BatchNormalization
template <typename T = float>
class BatchNormalization : public Activation
{
    using _super = Activation;

protected:
    bool                        m_bypass    = false;
    bool                        m_host_only = false;
    bool                        m_host_simd = true;
    bool                        m_fix_gamma = false;
    bool                        m_fix_beta  = false;

//  indices_t                   m_node_shape;
    
    FrameBuffer                 m_x_buf;

    std::shared_ptr<Tensor>     m_gamma;
    std::shared_ptr<Tensor>     m_beta;
    std::shared_ptr<Tensor>     m_dgamma;
    std::shared_ptr<Tensor>     m_dbeta;

    Tensor_<T>                  m_mean;     // 平均値
    Tensor_<T>                  m_rstd;     // 標準偏差の逆数

    Tensor_<T>                  m_running_mean;
    Tensor_<T>                  m_running_var;

    T                           m_momentum = (T)0.9;
    T                           m_init_gamma;
    T                           m_init_beta;

public:
    struct create_t
    {
        T       momentum  = (T)0.9;
        T       gamma     = (T)1.0;
        T       beta      = (T)0.0;
        bool    fix_gamma = false;
        bool    fix_beta  = false;
    };

protected:
    BatchNormalization(create_t const &create)
    {
        m_gamma  = std::make_shared<Tensor>();
        m_beta   = std::make_shared<Tensor>();
        m_dgamma = std::make_shared<Tensor>();
        m_dbeta  = std::make_shared<Tensor>();

        m_momentum   = create.momentum;
        m_init_gamma = create.gamma;
        m_init_beta  = create.beta;
        m_fix_gamma  = create.fix_gamma;
        m_fix_beta   = create.fix_beta;
    }

    void CommandProc(std::vector<std::string> args)
    {
        _super::CommandProc(args);

        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "bypass")
        {
            m_bypass = EvalBool(args[1]);
        }

        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }

        // Host SIMDモード設定
        if (args.size() == 2 && args[0] == "host_simd")
        {
            m_host_simd = EvalBool(args[1]);
        }

        if (args.size() == 2 && args[0] == "set_momentum") {
            m_momentum = (T)std::atof(args[1].c_str());
        }
        if (args.size() == 2 && args[0] == "fix_gamma") {
            m_fix_gamma = EvalBool(args[1]);
        }
        if (args.size() == 2 && args[0] == "fix_beta") {
            m_fix_beta = EvalBool(args[1]);
        }
        if (args.size() == 2 && args[0] == "set_gamma") {
            *m_gamma = (T)std::atof(args[1].c_str());
        }
        if (args.size() == 2 && args[0] == "set_beta") {
            *m_beta  = (T)std::atof(args[1].c_str());
        }
    }
    
    void PrintInfoText(std::ostream& os, std::string indent, int columns, int nest, int depth)
    {
        _super::PrintInfoText(os, indent, columns, nest, depth);
        os << indent << " momentum : " << m_momentum << std::endl;
    }

public:
    ~BatchNormalization() {}

    static std::shared_ptr<BatchNormalization> Create(create_t const &create)
    {
        return std::shared_ptr<BatchNormalization>(new BatchNormalization(create));
    }

    static std::shared_ptr<BatchNormalization> Create(T momentum = (T)0.9, T gamma=(T)1.0, T beta=(T)0.0)
    {
        create_t create;
        create.momentum = momentum;
        create.gamma    = gamma;
        create.beta     = beta;
        return Create(create);
    }

    // for python
    static std::shared_ptr<BatchNormalization> CreateEx(
            T       momentum  = (T)0.9,
            T       gamma     = (T)1.0,
            T       beta      = (T)0.0,
            bool    fix_gamma = false,
            bool    fix_beta  = false
        )
    {
        create_t create;
        create.momentum  = momentum;
        create.gamma     = gamma;
        create.beta      = beta;
        create.fix_gamma = fix_gamma;
        create.fix_beta  = fix_beta;
        return Create(create);
    }


    std::string GetClassName(void) const { return "BatchNormalization"; }
    
    // Serialize
    void Save(std::ostream &os) const 
    {
        SaveIndices(os, _super::m_shape);
        bb::SaveValue(os, m_momentum);
        m_gamma->Save(os);
        m_beta->Save(os);
        m_running_mean.Save(os);
        m_running_var.Save(os);
    }

    void Load(std::istream &is)
    {
        _super::m_shape = LoadIndices(is);
        bb::LoadValue(is, m_momentum);
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
        archive(cereal::make_nvp("shape",        _super::m_shape));
        archive(cereal::make_nvp("gamma",        *m_gamma));
        archive(cereal::make_nvp("beta",         *m_beta));
        archive(cereal::make_nvp("running_mean", m_running_mean));
        archive(cereal::make_nvp("running_var",  m_running_var));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("shape",        _super::m_shape));
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


    auto lock_gamma(void)              { return m_gamma->Lock<T>(); }
    auto lock_gamma_const(void)  const { return m_gamma->LockConst<T>(); }
    auto lock_beta(void)               { return m_beta->Lock<T>(); }
    auto lock_beta_const(void)   const { return m_beta->LockConst<T>(); }
    auto lock_dgamma(void)             { return m_dgamma->Lock<T>(); }
    auto lock_dgamma_const(void) const { return m_dgamma->LockConst<T>(); }
    auto lock_dbeta(void)              { return m_dbeta->Lock<T>(); }
    auto lock_dbeta_const(void)  const { return m_dbeta->LockConst<T>(); }
    auto lock_mean(void)               { return m_running_mean.Lock(); }
    auto lock_mean_const(void)   const { return m_running_mean.LockConst(); }
    auto lock_var(void)                { return m_running_var.Lock(); }
    auto lock_var_const(void)    const { return m_running_var.LockConst(); }
    
    // debug
    auto lock_tmp_mean_const(void)   const { return m_mean.LockConst(); }
    auto lock_tmp_rstd_const(void)   const { return m_rstd.LockConst(); }

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
        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

        _super::SetInputShape(shape);

        auto node_size = GetShapeSize(shape);
        
        // パラメータ初期化
        m_gamma->Resize ({node_size}, DataType<T>::type); *m_gamma  = m_init_gamma;
        m_beta->Resize  ({node_size}, DataType<T>::type); *m_beta   = m_init_beta;
        m_dgamma->Resize({node_size}, DataType<T>::type); *m_dgamma = (T)0.0;
        m_dbeta->Resize ({node_size}, DataType<T>::type); *m_dbeta  = (T)0.0;

        m_mean.Resize(node_size);
        m_rstd.Resize(node_size);

        m_running_mean.Resize(node_size); m_running_mean = (T)0.0;
        m_running_var.Resize(node_size);  m_running_var  = (T)1.0;

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
        if ( !this->m_parameter_lock ) {
            if ( !m_fix_gamma ) { parameters.PushBack(m_gamma); }
            if ( !m_fix_beta  ) { parameters.PushBack(m_beta);  }
        }
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
        if ( !this->m_parameter_lock ) {
            if ( !m_fix_gamma ) { gradients.PushBack(m_dgamma); }
            if ( !m_fix_beta  ) { gradients.PushBack(m_dbeta);  }
        }
        return gradients;
    }
    

    // ノード単位でのForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {
        BB_DEBUG_ASSERT(node >= 0 && node < GetShapeSize(_super::GetOutputShape()));

        auto gamma_ptr        = lock_gamma_const();
        auto beta_ptr         = lock_beta_const();
        auto running_mean_ptr = m_running_mean.LockConst();
        auto running_var_ptr  = m_running_var.LockConst();

        std::vector<double> y_vec(x_vec.size());
        for (size_t i = 0; i < x_vec.size(); ++i) {
            y_vec[i]  = x_vec[i];
            y_vec[i] -= (double)running_mean_ptr(node);
            y_vec[i] /= sqrt((double)running_var_ptr(node)) + 1.0e-7;
            y_vec[i]  = y_vec[i] * (double)gamma_ptr(node) + (double)beta_ptr(node);
        }
        return y_vec;
    }

    void        SetFrameBufferX(FrameBuffer x_buf) { m_x_buf = x_buf; }
    FrameBuffer GetFrameBufferX(void)              { return m_x_buf; }

    /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    FrameBuffer Forward(FrameBuffer x_buf, bool train=true)
    {
        // bypass
        if (m_bypass) {
            return x_buf;
        }

        // 出力設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), x_buf.GetShape(), x_buf.GetType());

        // backwardの為に保存
        if ( train ) {
            m_x_buf = x_buf;
        }
        
#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            if ( train ) {
                auto dev_x_ptr     = x_buf.LockDeviceMemoryConst();
                auto dev_y_ptr     = y_buf.LockDeviceMemory(true);
                auto dev_gamma_ptr = m_gamma->LockDeviceMemoryConst();
                auto dev_beta_ptr  = m_beta->LockDeviceMemoryConst();
                auto dev_mean_ptr = m_mean.LockDeviceMemory(true);
                auto dev_rstd_ptr = m_rstd.LockDeviceMemory(true);
                auto dev_running_mean_ptr = m_running_mean.LockDeviceMemory();
                auto dev_running_var_ptr = m_running_var.LockDeviceMemory();

                bbcu_fp32_BatchNormalization_ForwardTraining
                    (
                        (float const *)dev_x_ptr.GetAddr(),
                        (float       *)dev_y_ptr.GetAddr(),
                        (float const *)dev_gamma_ptr.GetAddr(),
                        (float const *)dev_beta_ptr.GetAddr(),
                        (float       *)dev_mean_ptr.GetAddr(),
                        (float       *)dev_rstd_ptr.GetAddr(),
                        (float       *)dev_running_mean_ptr.GetAddr(),
                        (float       *)dev_running_var_ptr.GetAddr(),
                        (float        )m_momentum,
                        (int          )x_buf.GetNodeSize(),
                        (int          )x_buf.GetFrameSize(),
                        (int          )x_buf.GetFrameStride() / sizeof(float)
                    );
                return y_buf;
            }
            else {
                auto dev_x_ptr            = x_buf.LockDeviceMemoryConst();
                auto dev_y_ptr            = y_buf.LockDeviceMemory(true);
                auto dev_gamma_ptr        = m_gamma->LockDeviceMemoryConst();
                auto dev_beta_ptr         = m_beta->LockDeviceMemoryConst();
                auto dev_running_mean_ptr = m_running_mean.LockDeviceMemoryConst();
                auto dev_running_var_ptr  = m_running_var.LockDeviceMemoryConst();

                bbcu_fp32_BatchNormalization_ForwardInference
                    (
                        (float const *)dev_x_ptr.GetAddr(),
                        (float       *)dev_y_ptr.GetAddr(),
                        (float const *)dev_gamma_ptr.GetAddr(),
                        (float const *)dev_beta_ptr.GetAddr(),
                        (float       *)dev_running_mean_ptr.GetAddr(),
                        (float       *)dev_running_var_ptr.GetAddr(),
                        (int          )x_buf.GetNodeSize(),
                        (int          )x_buf.GetFrameSize(),
                        (int          )x_buf.GetFrameStride() / sizeof(float)
                    );
                return y_buf;
            }
        }
#endif


        if ( DataType<T>::type == BB_TYPE_FP32 && m_host_simd ) {
            // SIMD版
            auto node_size    = x_buf.GetNodeSize();
            auto frame_size   = x_buf.GetFrameSize();
    //      auto frame_stride = x_buf.GetFrameStride() / sizeof(float);
        
            const int   mm256_frame_size = ((int)frame_size + 7) / 8 * 8;

            auto x_ptr            = x_buf.LockConst<T>();
            auto y_ptr            = y_buf.Lock<T>();

            auto gamma_ptr        = lock_gamma_const();
            auto beta_ptr         = lock_beta_const();

            auto mean_ptr         = m_mean.Lock();
            auto rstd_ptr         = m_rstd.Lock();        
            auto running_mean_ptr = m_running_mean.Lock();
            auto running_var_ptr  = m_running_var.Lock();

            if (train) {
                const __m256    reciprocal_frame_size = _mm256_set1_ps(1.0f / (float)frame_size);
                const __m256    epsilon = _mm256_set1_ps(1.0e-7f);

                #pragma omp parallel for
                for (int node = 0; node < (int)node_size; ++node) {
                    float const *x_addr = x_ptr.GetAddr(node);
                    float       *y_addr = y_ptr.GetAddr(node);

                    // 平均と分散計算
                    __m256 mean_sum = _mm256_set1_ps(0.0f);
                    __m256 mean_c   = _mm256_set1_ps(0.0f);
                    __m256 var_sum  = _mm256_set1_ps(0.0f);
                    __m256 var_c    = _mm256_set1_ps(0.0f);
                    for ( int frame = 0; frame < mm256_frame_size; frame += 8) {
                        __m256 x = _mm256_load_ps(&x_addr[frame + 0]);
                        __m256 mean_y = _mm256_sub_ps(x, mean_c);
                        __m256 mean_t = _mm256_add_ps(mean_sum, mean_y);
                               mean_c = _mm256_sub_ps(_mm256_sub_ps(mean_t, mean_sum), mean_y);
                        mean_sum = mean_t;

                        __m256 var_y = _mm256_fmsub_ps(x, x, var_c);
                        __m256 var_t = _mm256_add_ps(var_sum, var_y);
                               var_c = _mm256_sub_ps(_mm256_sub_ps(var_t, var_sum), var_y);
                        var_sum = var_t;
                    }
                    __m256 mean = _mm256_mul_ps(bb_mm256_hsum_ps(mean_sum), reciprocal_frame_size);
                    __m256 var = _mm256_fmsub_ps(bb_mm256_hsum_ps(var_sum), reciprocal_frame_size, _mm256_mul_ps(mean, mean));
                    var = _mm256_max_ps(var, _mm256_set1_ps(0.0f)); // 誤差対策(負にならないようにクリップ)

                    __m256 varx = _mm256_max_ps(var, epsilon);
                    __m256 rstd = _mm256_rsqrt_ps(varx);

                    varx = _mm256_mul_ps(varx, _mm256_set1_ps(0.5f));
                    rstd = _mm256_mul_ps(rstd, _mm256_fnmadd_ps(varx, _mm256_mul_ps(rstd, rstd), _mm256_set1_ps(1.5f)));
                    rstd = _mm256_mul_ps(rstd, _mm256_fnmadd_ps(varx, _mm256_mul_ps(rstd, rstd), _mm256_set1_ps(1.5f)));

                    // 実行時の mean と var 保存
                    running_mean_ptr[node] = running_mean_ptr[node] * m_momentum + bb_mm256_cvtss_f32(mean) * (1.0f - m_momentum);
                    running_var_ptr[node]  = running_var_ptr[node]  * m_momentum + bb_mm256_cvtss_f32(var)  * (1.0f - m_momentum);
                    
                    // 結果の保存
                    mean_ptr[node] = bb_mm256_cvtss_f32(mean);
                    rstd_ptr[node] = bb_mm256_cvtss_f32(rstd);

                    // 正規化 と gamma/beta 処理
                    __m256 gamma = _mm256_set1_ps(gamma_ptr[node]);
                    __m256 beta = _mm256_set1_ps(beta_ptr[node]);
    //              for (int frame = 0; frame < mm256_frame_size; frame += 8) {
                    for (int frame = mm256_frame_size-8; frame >= 0; frame -= 8) {
                    __m256 x = _mm256_load_ps(&x_addr[frame]);
                        __m256 xn = _mm256_mul_ps(_mm256_sub_ps(x, mean), rstd);
                        __m256 y = _mm256_fmadd_ps(xn, gamma, beta);
                        _mm256_store_ps(&y_addr[frame], y);
                    }
                }
            }
            else {
                #pragma omp parallel for
                for (int node = 0; node < (int)node_size; ++node) {
                    auto x_addr = x_ptr.GetAddr(node);
                    auto y_addr = y_ptr.GetAddr(node);

                    __m256 running_mean = _mm256_set1_ps(running_mean_ptr[node]);
                    __m256 running_var = _mm256_set1_ps(1.0f / (sqrt(running_var_ptr[node]) + 1.0e-7f));

                    __m256 gamma = _mm256_set1_ps(gamma_ptr[node]);
                    __m256 beta = _mm256_set1_ps(beta_ptr[node]);

                    for (int frame = 0; frame < mm256_frame_size; frame += 8) {
                        __m256 x = _mm256_load_ps(&x_addr[frame]);
                        __m256 xc = _mm256_sub_ps(x, running_mean);
                        __m256 xn = _mm256_mul_ps(xc, running_var);
                        __m256 y = _mm256_fmadd_ps(xn, gamma, beta);
                        _mm256_store_ps(&y_addr[frame], y);
                    }
                }
            }

            return y_buf;
        }
        
        {
            // 汎用版
            auto node_size    = x_buf.GetNodeSize();
            auto frame_size   = x_buf.GetFrameSize();

            auto x_ptr            = x_buf.LockConst<T>();
            auto y_ptr            = y_buf.Lock<T>();
            
            auto gamma_ptr        = lock_gamma_const();
            auto beta_ptr         = lock_beta_const();

            auto mean_ptr         = m_mean.Lock();
            auto rstd_ptr         = m_rstd.Lock();        
            auto running_mean_ptr = m_running_mean.Lock();
            auto running_var_ptr  = m_running_var.Lock();

            if ( train ) {
                #pragma omp parallel for
                for (index_t node = 0; node < node_size; ++node) {
                    // カハンの加算アルゴリズム(Kahan summation algorithm) [意味があるかは分からないが、どうせバス律速だろうからついで]
                    T s1 = 0, c1 = 0, y1, t1;
                    T s2 = 0, c2 = 0, y2, t2;
                    for ( index_t frame = 0; frame < frame_size; ++frame) {
                        T x = x_ptr.Get(frame, node);

                        y1 = x - c1;
                        t1 = s1 + y1;
                        c1 = (t1 - s1) - y1;
                        s1 = t1;

                        y2 = (x * x) - c2;
                        t2 = s2 + y2;
                        c2 = (t2 - s2) - y2;
                        s2 = t2;
                    }

                    // 集計
                    T mean = s1 / (T)frame_size;
                    T var  = (s2 / (T)frame_size) - (mean * mean);
                    var = std::max((T)0, var);  // 演算誤差で負にならないようにクリップ  // 演算誤差で負にならないようにクリップ
                    T std  = std::sqrt(var);
                    T rstd = (T)1.0 / (std + (T)1.0e-7);

                    running_mean_ptr[node] = (running_mean_ptr[node] * m_momentum) + (mean * ((T)1.0 - m_momentum));
                    running_var_ptr[node]  = (running_var_ptr[node]  * m_momentum) + (var *  ((T)1.0 - m_momentum));
                    
                    mean_ptr[node] = mean;
                    rstd_ptr[node] = rstd;

                    // 正規化
                    T   gamma = gamma_ptr[node];
                    T   beta  = beta_ptr[node];
                    for ( index_t frame = 0; frame < frame_size; ++frame) {
                        T x = x_ptr.Get(frame, node);
                        x = (x - mean) * rstd;
                        x = x * gamma + beta;
                        y_ptr.Set(frame, node, x);
                    }
                }
            }
            else {
                #pragma omp parallel for
                for (index_t node = 0; node < node_size; ++node) {
                    T   gamma = gamma_ptr[node];
                    T   beta  = beta_ptr[node];
                    T   mean  = running_mean_ptr[node];
                    T   var   = running_var_ptr[node];

                    T   rstd  = (T)1.0 / (std::sqrt(var) + (T)1.0e-7);

                    for ( index_t frame = 0; frame < frame_size; ++frame) {
                        T x = x_ptr.Get(frame, node);
                        y_ptr.Set(frame, node, ((x - mean) * rstd) * gamma + beta);
                    }
                }
            }

            return y_buf;
        }
 
    }


    // forward 再計算
    FrameBuffer ReForward(FrameBuffer x_buf)
    {
        // bypass
        if (m_bypass) {
            return x_buf;
        }

        // 出力設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), x_buf.GetShape(), x_buf.GetType());

        // backwardの為に保存
        m_x_buf = x_buf;

        
#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto dev_x_ptr     = x_buf.LockDeviceMemoryConst();
            auto dev_y_ptr     = y_buf.LockDeviceMemory(true);
            auto dev_gamma_ptr = m_gamma->LockDeviceMemoryConst();
            auto dev_beta_ptr  = m_beta->LockDeviceMemoryConst();
            auto dev_mean_ptr  = m_mean.LockDeviceMemoryConst();
            auto dev_rstd_ptr  = m_rstd.LockDeviceMemoryConst();

            bbcu_fp32_BatchNormalization_ReForward
                (
                    (float const *)dev_x_ptr.GetAddr(),
                    (float       *)dev_y_ptr.GetAddr(),
                    (float const *)dev_gamma_ptr.GetAddr(),
                    (float const *)dev_beta_ptr.GetAddr(),
                    (float       *)dev_mean_ptr.GetAddr(),
                    (float       *)dev_rstd_ptr.GetAddr(),
                    (int          )x_buf.GetNodeSize(),
                    (int          )x_buf.GetFrameSize(),
                    (int          )x_buf.GetFrameStride() / sizeof(float)
                );
            return y_buf;
        }
#endif

        {
            // 汎用版
            auto node_size        = x_buf.GetNodeSize();
            auto frame_size       = x_buf.GetFrameSize();

            auto x_ptr            = x_buf.LockConst<T>();
            auto y_ptr            = y_buf.Lock<T>();
            
            auto gamma_ptr        = lock_gamma_const();
            auto beta_ptr         = lock_beta_const();

            auto mean_ptr         = m_mean.Lock();
            auto rstd_ptr         = m_rstd.Lock();        
            auto running_mean_ptr = m_running_mean.Lock();
            auto running_var_ptr  = m_running_var.Lock();

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                // 集計
                T mean = mean_ptr[node];
                T rstd = rstd_ptr[node];

                // 正規化
                T   gamma = gamma_ptr[node];
                T   beta  = beta_ptr[node];
                for ( index_t frame = 0; frame < frame_size; ++frame) {
                    T x = x_ptr.Get(frame, node);
                    x = (x - mean) * rstd;
                    x = x * gamma + beta;
                    y_ptr.Set(frame, node, x);
                }
            }

            return y_buf;
        }
    }


    /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        if (m_bypass) {
            return dy_buf;
        }

        // 出力設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), dy_buf.GetShape(), dy_buf.GetType());

        // forward時のxを取得
        FrameBuffer x_buf = m_x_buf;
        m_x_buf = FrameBuffer();

#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && !m_host_only && dy_buf.IsDeviceAvailable() && x_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto dev_x_ptr      = x_buf.LockDeviceMemoryConst();
            auto dev_dy_ptr     = dy_buf.LockDeviceMemoryConst();
            auto dev_dx_ptr     = dx_buf.LockDeviceMemory(true);
            auto dev_gamma_ptr  = m_gamma->LockDeviceMemoryConst();
            auto dev_dgamma_ptr = m_dgamma->LockDeviceMemory();
            auto dev_dbeta_ptr  = m_dbeta->LockDeviceMemory();
            auto dev_mean_ptr   = m_mean.LockDeviceMemoryConst();
            auto dev_rstd_ptr   = m_rstd.LockDeviceMemoryConst();
            bbcu_fp32_BatchNormalization_Backward
                (
                    (const float *)dev_x_ptr.GetAddr(),
                    (const float *)dev_dy_ptr.GetAddr(),
                    (float       *)dev_dx_ptr.GetAddr(),
                    (float const *)dev_gamma_ptr.GetAddr(),
                    (float       *)dev_dgamma_ptr.GetAddr(),
                    (float       *)dev_dbeta_ptr.GetAddr(),
                    (float const *)dev_mean_ptr.GetAddr(),
                    (float const *)dev_rstd_ptr.GetAddr(),
                    (float        )1.0f / (float)dy_buf.GetFrameSize(),
                    (int          )dy_buf.GetNodeSize(),
                    (int          )dy_buf.GetFrameSize(),
                    (int          )dy_buf.GetFrameStride() / sizeof(float)
                );

            return dx_buf;
        }
#endif

        if ( DataType<T>::type == BB_TYPE_FP32 && m_host_simd ) {
            auto node_size    = dy_buf.GetNodeSize();
            auto frame_size   = dy_buf.GetFrameSize();
    //      auto frame_stride = dy_buf.GetFrameStride() / sizeof(float);
            
            const int   mm256_frame_size = ((int)frame_size + 7) / 8 * 8;
            
            auto gamma_ptr        = lock_gamma_const();
    //      auto beta_ptr         = lock_beta_const();
            auto dgamma_ptr       = lock_dgamma();
            auto dbeta_ptr        = lock_dbeta();

            auto mean_ptr         = m_mean.LockConst();
            auto rstd_ptr         = m_rstd.LockConst();
       
        
            // 逆数生成
            const __m256    reciprocal_frame_size = _mm256_set1_ps(1.0f / (float)frame_size);

            auto x_ptr  = x_buf.LockConst<T>();
//          auto y_ptr  = y_buf.LockConst<T>();
            auto dx_ptr = dx_buf.Lock<T>();
            auto dy_ptr = dy_buf.LockConst<T>();

            #pragma omp parallel for
            for (int node = 0; node < (int)node_size; ++node) {
                auto dy_addr = dy_ptr.GetAddr(node);
                auto dx_addr = dx_ptr.GetAddr(node);
                auto x_addr  = x_ptr.GetAddr(node);

                __m256 mean   = _mm256_set1_ps(mean_ptr[node]);
                __m256 rstd   = _mm256_set1_ps(rstd_ptr[node]);
                __m256 gamma  = _mm256_set1_ps(gamma_ptr[node]);
                __m256 dbeta  = _mm256_set1_ps(0);
                __m256 dgamma = _mm256_set1_ps(0);
                __m256 dstd = _mm256_set1_ps(0);
                __m256 dmeanx = _mm256_set1_ps(0);
                __m256 rstd2 = _mm256_mul_ps(rstd, rstd);

                for (int frame = 0; frame < mm256_frame_size; frame += 8) {
                    __m256 x = _mm256_load_ps(&x_addr[frame]);
                    __m256 xc = _mm256_sub_ps(x, mean);
                    __m256 xn = _mm256_mul_ps(xc, rstd);

                    __m256 dy = _mm256_load_ps(&dy_addr[frame]);
                    dbeta = _mm256_add_ps(dy, dbeta);
                    dgamma = _mm256_fmadd_ps(xn, dy, dgamma);

                    __m256 dxn = _mm256_mul_ps(dy, gamma);
                    dstd = _mm256_fnmadd_ps(_mm256_mul_ps(dxn, xc), rstd2, dstd);
                    dmeanx = _mm256_fnmadd_ps(dxn, rstd, dmeanx);
                }
                dbeta = bb_mm256_hsum_ps(dbeta);
                dgamma = bb_mm256_hsum_ps(dgamma);
                dgamma_ptr[node] += bb_mm256_cvtss_f32(dgamma);
                dbeta_ptr[node]  += bb_mm256_cvtss_f32(dbeta);

                dstd = bb_mm256_hsum_ps(dstd);
                dmeanx = bb_mm256_hsum_ps(dmeanx);

                __m256 dvar  = _mm256_mul_ps(dstd, rstd);
                __m256 dmean = _mm256_mul_ps(_mm256_fnmadd_ps(mean, dvar, dmeanx), reciprocal_frame_size);

    //          for (int frame = 0; frame < mm256_frame_size; frame += 8) {
                for (int frame = mm256_frame_size - 8; frame >= 0; frame -= 8) {
                    __m256 dy = _mm256_load_ps(&dy_addr[frame]);
                    __m256 x = _mm256_load_ps(&x_addr[frame]);
                    __m256 dxn = _mm256_mul_ps(dy, gamma);
                    __m256 dxc = _mm256_fmadd_ps(dxn, rstd, dmean);
                    __m256 dx = _mm256_fmadd_ps(_mm256_mul_ps(x, dvar), reciprocal_frame_size, dxc);
                    _mm256_store_ps(&dx_addr[frame], dx);
                }
            }

            return dx_buf;
        }


        {
            // 汎用版
            auto node_size    = dy_buf.GetNodeSize();
            auto frame_size   = dy_buf.GetFrameSize();
    //      auto frame_stride = dy_buf.GetFrameStride() / sizeof(float);
            
    //      const int   mm256_frame_size = ((int)frame_size + 7) / 8 * 8;
            
            auto gamma_ptr        = lock_gamma_const();
    //      auto beta_ptr         = lock_beta_const();
            auto dgamma_ptr       = lock_dgamma();
            auto dbeta_ptr        = lock_dbeta();

            auto mean_ptr         = m_mean.LockConst();
            auto rstd_ptr         = m_rstd.LockConst();
            
            auto x_ptr  = x_buf.LockConst<T>();
//          auto y_ptr  = y_buf.LockConst<T>();
            auto dx_ptr = dx_buf.Lock<T>();
            auto dy_ptr = dy_buf.LockConst<T>();

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                T   mean   = mean_ptr[node];
                T   rstd   = rstd_ptr[node];
                T   gamma  = gamma_ptr[node];
                T   dgamma = 0;
                T   dbeta  = 0;
                T   dmeanx = 0;
                T   dstd   = 0;

                for ( index_t frame = 0; frame < frame_size; ++frame) {
                    T x  = x_ptr.Get(frame, node);
                    T dy = dy_ptr.Get(frame, node);
                    T xc = x - mean;
                    T xn = xc * rstd;
                    dbeta  += dy;
                    dgamma += xn * dy;

                    T dxn = gamma * dy;
                    dstd += -(dxn * xc * (rstd * rstd));
                    dmeanx += -(dxn * rstd);
                }

                dgamma_ptr[node] += dgamma;
                dbeta_ptr[node]  += dbeta;

                T dvar  = dstd * rstd;
                T dmean = (dmeanx - (mean * dvar)) / (T)frame_size;

                for ( index_t frame = 0; frame < frame_size; ++frame) {
                    T dy = dy_ptr.Get(frame, node);
                    T x  = x_ptr.Get(frame, node);
                    T dxn = dy * gamma;
                    T dxc = dxn * rstd;
                    T dx  = dxc + dmean + (x * dvar / (T)frame_size);
                    dx_ptr.Set(frame, node, dx);
                }
            }

            return dx_buf;
        } 
    }
};

}
