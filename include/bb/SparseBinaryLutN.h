// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>
#include <random>

#include "bb/SparseLayer.h"
#include "bb/Tensor.h"


namespace bb {


template <int N = 6, typename BinType = Bit, typename RealType = float>
class SparseBinaryLutN : public SparseLayer
{
    using _super = SparseLayer;
    static int const NN = (1 << N);

protected:
    bool                    m_lut_binarize = true;
    bool                    m_host_only    = false;

    RealType                m_unbinarize_bias = (RealType)0.2;

    index_t                 m_max_tmp_mem_size = 256 * 1024 * 1024;

    std::string             m_connection;

    indices_t               m_input_shape;
    indices_t               m_output_shape;

    FrameBuffer             m_x_buf;

    Tensor_<std::int32_t>   m_input_index;
    Tensor_<std::int32_t>   m_reverse_index;
    bool                    m_reverse_index_dirty = true;

    std::shared_ptr<Tensor> m_W;
    std::shared_ptr<Tensor> m_dW;

    RealType                m_momentum = (RealType)0.0;

    RealType                m_gamma;
    RealType                m_beta;
    
    Tensor_<RealType>       m_mean;     // 平均値
    Tensor_<RealType>       m_rstd;     // 標準偏差の逆数

    Tensor_<RealType>       m_running_mean;
    Tensor_<RealType>       m_running_var;
    
    std::mt19937_64         m_mt;

public:
    struct create_t
    {
        indices_t       output_shape;               //< 出力形状
        std::string     connection;                 //< 結線ルール
        RealType        momentum  = (RealType)0.0;
        RealType        gamma     = (RealType)0.5;
        RealType        beta      = (RealType)0.5;
        std::uint64_t   seed      = 1;              //< 乱数シード
    };

protected:
    SparseBinaryLutN(create_t const &create)
    {
        BB_ASSERT(!create.output_shape.empty());

        m_output_shape = create.output_shape;
        m_connection   = create.connection;
        m_momentum     = create.momentum;
        m_gamma        = create.gamma;
        m_beta         = create.beta;

        m_mt.seed(create.seed);

        m_W  = std::make_shared<Tensor>();
        m_dW = std::make_shared<Tensor>();
    }

    void CommandProc(std::vector<std::string> args)
    {
        // LUTバイナライズ設定
        if ( args.size() == 2 && args[0] == "lut_binarize" )
        {
            m_lut_binarize = EvalBool(args[1]);
        }

        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }
    }

public:
    ~SparseBinaryLutN() {}


    static std::shared_ptr<SparseBinaryLutN> Create(create_t const &create)
    {
        return std::shared_ptr<SparseBinaryLutN>(new SparseBinaryLutN(create));
    }

    static std::shared_ptr<SparseBinaryLutN> Create(indices_t const &output_shape, std::string connection = "", std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.connection   = connection;
        create.seed         = seed;
        return Create(create);
    }

    static std::shared_ptr<SparseBinaryLutN> Create(index_t output_node_size, std::string connection = "", std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape.resize(1);
        create.output_shape[0] = output_node_size;
        create.connection      = connection;
        create.seed            = seed;
        return Create(create);
    }

    std::string GetClassName(void) const { return "SparseBinaryLutN"; }


public:
    // Serialize
    void Save(std::ostream &os) const 
    {
        SaveIndices(os, m_input_shape);
        SaveIndices(os, m_output_shape);
        m_input_index.Save(os);
        m_W->Save(os);
        bb::SaveValue(os, m_momentum);
        bb::SaveValue(os, m_gamma);
        bb::SaveValue(os, m_beta);
        m_running_mean.Save(os);
        m_running_var.Save(os);
    }

    void Load(std::istream &is)
    {
        m_input_shape  = LoadIndices(is);
        m_output_shape = LoadIndices(is);
        m_input_index.Load(is);
        m_W->Load(is);
        bb::LoadValue(is, m_momentum);
        bb::LoadValue(is, m_gamma);
        bb::LoadValue(is, m_beta);
        m_running_mean.Load(is);
        m_running_var.Load(is);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
        archive(cereal::make_nvp("input_shape",  m_input_shape));
        archive(cereal::make_nvp("output_shape", m_output_shape));
        archive(cereal::make_nvp("input_index",  m_input_index));
        archive(cereal::make_nvp("W",            *m_W));
        archive(cereal::make_nvp("gamma",        m_gamma));
        archive(cereal::make_nvp("beta",         m_beta));
        archive(cereal::make_nvp("running_mean", m_running_mean));
        archive(cereal::make_nvp("running_var",  m_running_var));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("input_shape",  m_input_shape));
        archive(cereal::make_nvp("output_shape", m_output_shape));
        archive(cereal::make_nvp("input_index",  m_input_index));
        archive(cereal::make_nvp("W",            *m_W));
        archive(cereal::make_nvp("gamma",        m_gamma));
        archive(cereal::make_nvp("beta",         m_beta));
        archive(cereal::make_nvp("running_mean", m_running_mean));
        archive(cereal::make_nvp("running_var",  m_running_var));
    }

    void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("SparseBinaryLutN", *this));
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("SparseBinaryLutN", *this));
    }
#endif


    Tensor       &W(void)       { return *m_W; }
    Tensor const &W(void) const { return *m_W; }
    
    Tensor       &dW(void)       { return *m_dW; }
    Tensor const &dW(void) const { return *m_dW; }

    auto lock_InputIndex(void)             { return m_input_index.Lock(); }
    auto lock_InputIndex_const(void) const { return m_input_index.LockConst(); }

    auto lock_W(void)              { return m_W->Lock<RealType>(); }
    auto lock_W_const(void) const  { return m_W->LockConst<RealType>(); }
    auto lock_dW(void)             { return m_dW->Lock<RealType>(); }
    auto lock_dW_const(void) const { return m_dW->LockConst<RealType>(); }

    auto lock_mean(void)               { return m_running_mean.Lock(); }
    auto lock_mean_const(void)   const { return m_running_mean.LockConst(); }
    auto lock_var(void)                { return m_running_var.Lock(); }
    auto lock_var_const(void)    const { return m_running_var.LockConst(); }
    
    // debug
    auto lock_tmp_mean_const(void)   const { return m_mean.LockConst(); }
    auto lock_tmp_rstd_const(void)   const { return m_rstd.LockConst(); }

    index_t GetNodeInputSize(index_t node) const
    {
        return N;
    }

    void SetNodeInput(index_t node, index_t input_index, index_t input_node)
    {
        auto ptr = lock_InputIndex();
        ptr(node, input_index) = (std::int32_t)input_node;
        m_reverse_index_dirty = true;
    }

    index_t GetNodeInput(index_t node, index_t input_index) const
    {
        auto ptr = lock_InputIndex_const();
        return (index_t)ptr(node, input_index);
    }


   /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 形状設定
        m_input_shape = shape;
        
        // 接続初期化
        auto output_node_size = GetShapeSize(m_output_shape);
        m_input_index.Resize(output_node_size, N);
        this->InitializeNodeInput(m_mt(), m_connection);
        m_reverse_index_dirty = true;

        // パラメータ初期化(結局初期値は何が良いのかまだよくわからない)
        m_W->Resize(DataType<RealType>::type, GetShapeSize(m_output_shape), NN);  m_W->InitNormalDistribution(0.5, 0.01, m_mt());
        m_dW->Resize(DataType<RealType>::type, GetShapeSize(m_output_shape), NN); m_dW->FillZero();

        m_mean.Resize(m_output_shape);
        m_rstd.Resize(m_output_shape);

        m_running_mean.Resize(m_output_shape); m_running_mean = (RealType)0.0;
        m_running_var.Resize(m_output_shape);  m_running_var  = (RealType)1.0;

        return m_output_shape;
    }


    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_input_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_output_shape;
    }
    
    
    
    Variables GetParameters(void)
    {
        Variables parameters;
        parameters.PushBack(m_W);
        return parameters;
    }

    Variables GetGradients(void)
    {
        Variables gradients;
        gradients.PushBack(m_dW);
        return gradients;
    }
    
    void        SetFrameBufferX(FrameBuffer x) { m_x_buf = x; }
    FrameBuffer GetFrameBufferX(void)          { return m_x_buf; }

    // ノード単位でのForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> input_value) const
    {
        BB_ASSERT(input_value.size() == N);

        // パラメータクリップ
        m_W->Clamp((RealType)0.0, (RealType)1.0);

        auto W_ptr = lock_W_const();
        RealType W[NN];
        for ( int i = 0; i < NN; ++i) {
            W[i] = W_ptr(node, i);
            if ( m_lut_binarize ) {
                W[i] = W[i] > (RealType)0.5 ? (RealType)1.0 : (RealType)0.0;
            }
        }

        RealType   x[N][2];
        for ( int i = 0; i < N; ++i) {
            RealType in_sig = (RealType)input_value[i];
            in_sig = std::min((RealType)1.0, std::max((RealType)0.0, in_sig));  // clip
            x[i][0] = (RealType)1.0 - in_sig;
            x[i][1] = in_sig;
        }

        RealType y = (RealType)0;
        for (int i = 0; i < NN; ++i) {
            RealType w = W[i];
            for (int j = 0; j < N; ++j) {
                w *= x[j][(i >> j) & 1];
            }
            y += w;
        }

        // clip
        y = std::max((RealType)0.0, y);
        y = std::min((RealType)1.0, y);
        
        // batch_noerm
        auto running_mean_ptr = m_running_mean.LockConst();
        auto running_var_ptr  = m_running_var.LockConst();
        y -= running_mean_ptr(node);
        y /= (RealType)sqrt(running_var_ptr(node)) + (RealType)1.0e-7;
        y  = y * m_gamma + m_beta;

        // binary
        y = (y > (RealType)0.5) ? (RealType)1.0 : (RealType)0.0;

        std::vector<double> result;
        result.push_back((double)y);

        return result;
    }


    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        BB_ASSERT(x_buf.GetType() == DataType<BinType>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_input_shape) {
            SetInputShape(x_buf.GetShape());
        }

        // 出力を設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), m_output_shape, DataType<BinType>::type);

        // backwardの為に保存
        if ( train ) {
            m_x_buf = x_buf;
        }

        // パラメータクリップ
        m_W->Clamp((RealType)0.0, (RealType)1.0);

#ifdef BB_WITH_CUDA
        // LUT6 Bit CUDA
        if ( N == 6 && DataType<BinType>::type == BB_TYPE_BIT && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            if ( train ) {
                auto x_ptr            = x_buf.LockDeviceMemoryConst();
                auto y_ptr            = y_buf.LockDeviceMemory(true);
                auto input_index_ptr  = m_input_index.LockDeviceMemoryConst();
                auto W_ptr            = m_W->LockDeviceMemoryConst();
                auto mean_ptr         = m_mean.LockDeviceMemory(true);
                auto rstd_ptr         = m_rstd.LockDeviceMemory(true);
                auto running_mean_ptr = m_running_mean.LockDeviceMemory();
                auto running_var_ptr  = m_running_var.LockDeviceMemory();

                bbcu_bit_fp32_SparseBinaryLut6_ForwardTraining
                    (
                        (int   const *)x_ptr.GetAddr(),
                        (int         *)y_ptr.GetAddr(),
                        (int   const *)input_index_ptr.GetAddr(),
                        (float const *)W_ptr.GetAddr(),
                        (float       *)mean_ptr.GetAddr(),
                        (float       *)rstd_ptr.GetAddr(),
                        (float       *)running_mean_ptr.GetAddr(),
                        (float       *)running_var_ptr.GetAddr(),
                        (float        )m_gamma,
                        (float        )m_beta,
                        (float        )m_momentum,
                        (float        )m_unbinarize_bias, 
                        (int          )y_buf.GetNodeSize(),
                        (int          )y_buf.GetFrameSize(),
                        (int          )(y_buf.GetFrameStride() / sizeof(int)),
                        (int          )(m_lut_binarize ? 1 : 0)
                    );
            }
            else {
                auto x_ptr            = x_buf.LockDeviceMemoryConst();
                auto y_ptr            = y_buf.LockDeviceMemory(true);
                auto input_index_ptr  = m_input_index.LockDeviceMemoryConst();
                auto W_ptr            = m_W->LockDeviceMemoryConst();
                auto running_mean_ptr = m_running_mean.LockDeviceMemory();
                auto running_var_ptr  = m_running_var.LockDeviceMemory();

                bbcu_bit_fp32_SparseBinaryLut6_ForwardInference
                    (
                        (int   const *)x_ptr.GetAddr(),
                        (int         *)y_ptr.GetAddr(),
                        (int   const *)input_index_ptr.GetAddr(),
                        (float const *)W_ptr.GetAddr(),
                        (float       *)running_mean_ptr.GetAddr(),
                        (float       *)running_var_ptr.GetAddr(),
                        (float        )m_gamma,
                        (float        )m_beta,
                        (float        )m_unbinarize_bias,
                        (int          )y_buf.GetNodeSize(),
                        (int          )y_buf.GetFrameSize(),
                        (int          )(y_buf.GetFrameStride() / sizeof(int)),
                        (int          )(m_lut_binarize ? 1 : 0)
                    );
            }

            return y_buf;
        }
#endif

        {
            // Generic
            auto node_size  = y_buf.GetNodeSize();
            auto frame_size = y_buf.GetFrameSize();

            auto x_ptr           = x_buf.LockConst<BinType>();
            auto y_ptr           = y_buf.Lock<RealType>();
            auto input_index_ptr = m_input_index.LockConst();
            auto W_ptr           = lock_W_const();

            #pragma omp parallel for
            for ( index_t node = 0; node < node_size; ++node ) {
                RealType W[NN];
                for ( int i = 0; i < NN; ++i) {
                    W[i] = W_ptr(node, i);
                    if ( m_lut_binarize ) {
                        W[i] = W[i] > (RealType)0.5 ? (RealType)1.0 : (RealType)0.0;
                    }
                }

                for ( index_t frame = 0; frame < frame_size; ++frame ) {
                    RealType   x[N][2];
                    for ( int i = 0; i < N; ++i) {
                        RealType in_sig = (RealType)x_ptr.Get(frame, input_index_ptr(node, i));
                        in_sig = in_sig > (RealType)0.5 ? (RealType)0.7 : (RealType)0.3;

                        x[i][0] = (RealType)1.0 - in_sig;
                        x[i][1] = in_sig;
                    }

                    RealType y = (RealType)0;
                    for (int i = 0; i < NN; ++i) {
                        RealType w = W[i];
                        for (int j = 0; j < N; ++j) {
                            w *= x[j][(i >> j) & 1];
                        }
                        y += w;
                    }

                    // clip
                    y = std::max((RealType)0.0, y);
                    y = std::min((RealType)1.0, y);

                    y_ptr.Set(frame, node, y);
                }
            }

            return y_buf;
        }
    }


    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        FrameBuffer x_buf = m_x_buf;
        m_x_buf = FrameBuffer();

        FrameBuffer dx_buf(dy_buf.GetFrameSize(), m_input_shape, DataType<RealType>::type);

        // make reverse index table
        if ( m_reverse_index_dirty ) {
            m_reverse_index = this->MakeReverseIndexTable(m_input_index, GetShapeSize(m_input_shape));
            m_reverse_index_dirty = false;
        }

        // tmp buffer
        index_t tmp_frame_size = m_max_tmp_mem_size / (sizeof(float) * GetShapeSize(m_output_shape)*N);
        tmp_frame_size = ((tmp_frame_size + 31) & ~0x1f);
        tmp_frame_size = std::min(tmp_frame_size, dy_buf.GetFrameSize());
        FrameBuffer tmp_buf(tmp_frame_size, GetShapeSize(m_output_shape)*N, DataType<RealType>::type);


#ifdef BB_WITH_CUDA
        if ( N == 6 && DataType<BinType>::type == BB_TYPE_BIT && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
                && x_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && tmp_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {

            Tensor_<RealType>   dmean(m_output_shape);
            Tensor_<RealType>   dvar(m_output_shape);

            auto x_ptr             = x_buf.LockDeviceMemoryConst();
            auto dy_ptr            = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr            = dx_buf.LockDeviceMemory(true);
            auto tmp_ptr           = tmp_buf.LockDeviceMemory(true);
            auto input_index_ptr   = m_input_index.LockDeviceMemoryConst();
            auto reverse_index_ptr = m_reverse_index.LockDeviceMemoryConst();
            auto W_ptr             = m_W->LockDeviceMemoryConst();
            auto dW_ptr            = m_dW->LockDeviceMemory();
            auto mean_ptr          = m_mean.LockDeviceMemoryConst();
            auto rstd_ptr          = m_rstd.LockDeviceMemoryConst();
            auto dmean_ptr         = dmean.LockDeviceMemory(true);
            auto dvar_ptr          = dvar.LockDeviceMemory(true);
            
            bbcu_bit_fp32_SparseBinaryLut6_Backward
                (
                    (int   const *)x_ptr.GetAddr(),
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (float       *)tmp_ptr.GetAddr(),
                    (int   const *)input_index_ptr.GetAddr(),
                    (int   const *)reverse_index_ptr.GetAddr(),
                    (float const *)W_ptr.GetAddr(),
                    (float       *)dW_ptr.GetAddr(),
                    (float const *)mean_ptr.GetAddr(),
                    (float const *)rstd_ptr.GetAddr(),
                    (float       *)dmean_ptr.GetAddr(),
                    (float       *)dvar_ptr.GetAddr(),
                    (float        )m_gamma,
                    (float        )m_beta,
                    (float        )m_unbinarize_bias,
                    (int          )m_reverse_index.GetShape()[0],
                    (int          )dx_buf.GetNodeSize(),
                    (int          )dy_buf.GetNodeSize(),
                    (int          )dy_buf.GetFrameSize(),
                    (int          )(dy_buf.GetFrameStride() / sizeof(float)),
                    (int          )(x_buf.GetFrameStride() / sizeof(int)),
                    (int          )tmp_buf.GetFrameSize(),
                    (int          )(tmp_buf.GetFrameStride() / sizeof(int)),
                    (int          )m_lut_binarize
                );
            
            return dx_buf;
        }
#endif

        BB_ASSERT(0);

        return dx_buf;
    }
};


}


// end of file