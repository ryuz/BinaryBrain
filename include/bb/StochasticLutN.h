// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include "bb/LutLayer.h"


namespace bb {


// 確率的LUTの抽象レイヤー
template <int N = 6, typename T = float>
class StochasticLutN : public SparseLayer<T, T>
{
    using _super = SparseLayer<T, T>;
    static int const NN = (1 << N);

protected:
    bool                    m_lut_binarize = true;
    bool                    m_y_binarize = false;
    bool                    m_host_only = false;
    bool                    m_host_simd = true;

    std::string             m_connection;

    indices_t               m_input_shape;
    indices_t               m_output_shape;

    FrameBuffer             m_x_buf;
    FrameBuffer             m_y_buf;
    FrameBuffer             m_dx_buf;

    FrameBuffer             m_tmp_buf;


    Tensor_<std::int32_t>   m_input_index;

    std::shared_ptr<Tensor> m_W;
    std::shared_ptr<Tensor> m_dW;

    std::mt19937_64         m_mt;

public:
    struct create_t
    {
        indices_t       output_shape;   //< 出力形状
        std::string     connection;     //< 結線ルール
        std::uint64_t   seed = 1;       //< 乱数シード
    };

protected:
    StochasticLutN(create_t const &create)
    {
        BB_ASSERT(!create.output_shape.empty());

        m_output_shape     = create.output_shape;
        m_connection       = create.connection;
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

        // Y出力バイナライズ設定
        if ( args.size() == 2 && args[0] == "y_binarize" )
        {
            m_y_binarize = EvalBool(args[1]);
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
    }

public:
    ~StochasticLutN() {}


    static std::shared_ptr<StochasticLutN> Create(create_t const &create)
    {
        return std::shared_ptr<StochasticLutN>(new StochasticLutN(create));
    }

    static std::shared_ptr<StochasticLutN> Create(indices_t const &output_shape, std::string connection = "", std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.connection   = connection;
        create.seed         = seed;
        return Create(create);
    }

    static std::shared_ptr<StochasticLutN> Create(index_t output_node_size, std::string connection = "", std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape.resize(1);
        create.output_shape[0] = output_node_size;
        create.connection      = connection;
        create.seed            = seed;
        return Create(create);
    }

    std::string GetClassName(void) const { return "StochasticLutN"; }


public:
    // Serialize
    void Save(std::ostream &os) const 
    {
        SaveIndices(os, m_input_shape);
        SaveIndices(os, m_output_shape);
        m_input_index.Save(os);
        m_W->Save(os);
    }

    void Load(std::istream &is)
    {
        m_input_shape  = LoadIndices(is);
        m_output_shape = LoadIndices(is);
        m_input_index.Load(is);
        m_W->Load(is);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
        archive(cereal::make_nvp("input_shape",      m_input_shape));
        archive(cereal::make_nvp("output_shape",     m_output_shape));
        archive(cereal::make_nvp("input_index",      m_input_index));
        archive(cereal::make_nvp("W",                *m_W));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("input_shape",      m_input_shape));
        archive(cereal::make_nvp("output_shape",     m_output_shape));
        archive(cereal::make_nvp("input_index",      m_input_index));
        archive(cereal::make_nvp("W",                *m_W));
    }

    void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("StochasticLutN", *this));
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("StochasticLutN", *this));
    }
#endif


    Tensor       &W(void)       { return *m_W; }
    Tensor const &W(void) const { return *m_W; }
    
    Tensor       &dW(void)       { return *m_dW; }
    Tensor const &dW(void) const { return *m_dW; }

    auto lock_InputIndex(void)             { return m_input_index.Lock(); }
    auto lock_InputIndex_const(void) const { return m_input_index.LockConst(); }

    auto lock_W(void)              { return m_W->Lock<T>(); }
    auto lock_W_const(void) const  { return m_W->LockConst<T>(); }
    auto lock_dW(void)             { return m_dW->Lock<T>(); }
    auto lock_dW_const(void) const { return m_dW->LockConst<T>(); }


    index_t GetNodeInputSize(index_t node) const
    {
        return N;
    }

    void SetNodeInput(index_t node, index_t input_index, index_t input_node)
    {
        auto ptr = lock_InputIndex();
        ptr(node, input_index) = (std::int32_t)input_node;
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

        // パラメータ初期化(結局初期値は何が良いのかまだよくわからない)
//      m_W->Resize(DataType<T>::type, m_output_node_size, NN);  m_W->InitUniformDistribution(0.4, 0.6, m_mt());
//      m_W->Resize(DataType<T>::type, m_output_node_size, NN);  m_W->InitUniformDistribution(0.0, 1.0, m_mt());
//      m_W->Resize(DataType<T>::type, m_output_node_size, NN);  m_W->InitNormalDistribution(0.5, 0.001, m_mt());
        m_W->Resize(DataType<T>::type, GetShapeSize(m_output_shape), NN);  m_W->InitNormalDistribution(0.5, 0.01, m_mt());

        m_dW->Resize(DataType<T>::type, GetShapeSize(m_output_shape), NN); m_dW->FillZero();

        return m_output_shape;
    }

    /**
     * @brief  出力のshape設定
     * @detail 出力のshape設定
     *         出力ノード数が変わらない限りshpeは自由
     * @param shape 新しいshape
     * @return なし
     */
    void SetOutputShape(indices_t const &shape)
    {
        BB_ASSERT(GetShapeSize(shape) == m_output_node_size);
        m_output_shape = shape;
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
    

    // ノード単位でのForward計算
    std::vector<T> ForwardNode(index_t node, std::vector<T> input_value) const
    {
        BB_ASSERT(input_value.size() == N);

        // パラメータクリップ
        m_W->Clamp((T)0.0, (T)1.0);

        auto W_ptr = lock_W_const();
        T W[NN];
        for ( int i = 0; i < NN; ++i) {
            W[i] = W_ptr(node, i);
            if ( m_lut_binarize ) {
                W[i] = W[i] > (T)0.5 ? (T)1.0 : (T)0.0;
            }
        }

        T   x[N][2];
        for ( int i = 0; i < N; ++i) {
            T in_sig = input_value[i];
            in_sig = std::min((T)1.0, std::max((T)0.0, in_sig));  // clip
            x[i][0] = (T)1.0 - in_sig;
            x[i][1] = in_sig;
        }

        T y = (T)0;
        for (int i = 0; i < NN; ++i) {
            T w = W[i];
            for (int j = 0; j < N; ++j) {
                w *= x[j][(i >> j) & 1];
            }
            y += w;
        }

        // clip
        y = std::max((T)0.0, y);
        y = std::min((T)1.0, y);
        
        std::vector<T> result;
        result.push_back(y);

        return result;
    }


    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        BB_ASSERT(x_buf.GetType() == DataType<T>::type);

        // backwardの為に保存
        m_x_buf = x_buf;

        // SetInputShpaeされていなければ初回に設定
        if (m_x_buf.GetShape() != m_input_shape) {
            SetInputShape(m_x_buf.GetShape());
        }

        // 出力を設定
        m_y_buf.Resize(DataType<T>::type, m_x_buf.GetFrameSize(), m_output_shape);

        // パラメータクリップ
        m_W->Clamp((T)0.0, (T)1.0);


#ifdef BB_WITH_CUDA
        // LUT6 CUDA版
        if ( N == 6 && DataType<T>::type == BB_TYPE_FP32 && !m_host_only
                && m_x_buf.IsDeviceAvailable() && m_y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            auto x_ptr           = x_buf.LockDeviceMemoryConst();
            auto y_ptr           = m_y_buf.LockDeviceMemory(true);
            auto input_index_ptr = m_input_index.LockDeviceMemoryConst();
            auto W_ptr           = m_W->LockDeviceMemoryConst();
               
            bbcu_fp32_StochasticLut6_Forward
                (
                    (const float *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int   const *)input_index_ptr.GetAddr(),
                    (float const *)W_ptr.GetAddr(),
                    (int          )m_y_buf.GetNodeSize(),
                    (int          )m_y_buf.GetFrameSize(),
                    (int          )(m_y_buf.GetFrameStride() / sizeof(float)),
                    (int          )(m_lut_binarize ? 1 : 0)
                );

            return m_y_buf;
        }
#endif

        {
            // 汎用版
            auto node_size  = m_y_buf.GetNodeSize();
            auto frame_size = m_y_buf.GetFrameSize();

            auto x_ptr           = x_buf.LockConst<T>();
            auto y_ptr           = m_y_buf.Lock<T>();
            auto input_index_ptr = m_input_index.LockConst();
            auto W_ptr           = lock_W_const();

            #pragma omp parallel for
            for ( index_t node = 0; node < node_size; ++node ) {
                T W[NN];
                for ( int i = 0; i < NN; ++i) {
                    W[i] = W_ptr(node, i);
                    if ( m_lut_binarize ) {
                        W[i] = W[i] > (T)0.5 ? (T)1.0 : (T)0.0;
                    }
                }

                for ( index_t frame = 0; frame < frame_size; ++frame ) {
                    T   x[N][2];
                    for ( int i = 0; i < N; ++i) {
                        T in_sig = x_ptr.Get(frame, input_index_ptr(node, i));
                        in_sig = std::min((T)1.0, std::max((T)0.0, in_sig));  // clip

                        x[i][0] = (T)1.0 - in_sig;
                        x[i][1] = in_sig;
                    }

                    T y = (T)0;
                    for (int i = 0; i < NN; ++i) {
                        T w = W[i];
                        for (int j = 0; j < N; ++j) {
                            w *= x[j][(i >> j) & 1];
                        }
                        y += w;
                    }

                    // clip
                    y = std::max((T)0.0, y);
                    y = std::min((T)1.0, y);

                    y_ptr.Set(frame, node, y);
                }
            }

            return m_y_buf;
        }
    }


    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(dy_buf.GetType() == DataType<T>::type);

        m_dx_buf.Resize(DataType<T>::type, dy_buf.GetFrameSize(), m_input_shape);
        m_tmp_buf.Resize(DataType<T>::type, dy_buf.GetFrameSize(), GetShapeSize(m_output_shape)*N);
        

#ifdef BB_WITH_CUDA
        if ( N == 6, DataType<T>::type == BB_TYPE_FP32 && !m_host_only
                && dy_buf.IsDeviceAvailable() && m_y_buf.IsDeviceAvailable() && m_dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            auto x_ptr           = m_x_buf.LockDeviceMemoryConst();
            auto dy_ptr          = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr          = m_dx_buf.LockDeviceMemory(true);
            auto input_index_ptr = m_input_index.LockDeviceMemoryConst();
            auto W_ptr           = m_W->LockDeviceMemoryConst();
            auto dW_ptr          = m_dW->LockDeviceMemory();
            auto tmp_ptr         = m_tmp_buf.LockDeviceMemory();
            
            bbcu_fp32_StochasticLut6_Backward(
                    (float const *)x_ptr.GetAddr(),
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (float       *)tmp_ptr.GetAddr(),
                    (int   const *)input_index_ptr.GetAddr(),
                    (float const *)W_ptr.GetAddr(),
                    (float       *)dW_ptr.GetAddr(),
                    (int          )m_dx_buf.GetNodeSize(),
                    (int          )m_y_buf.GetNodeSize(),
                    (int          )m_dx_buf.GetFrameSize(),
                    (int          )(m_dx_buf.GetFrameStride() / sizeof(float)),
                    (int          )(m_lut_binarize ? 1 : 0)
                );
            
            return m_dx_buf;
        }
#endif

        {
            // 汎用版
            m_dx_buf.FillZero();

            auto node_size  = dy_buf.GetNodeSize();
            auto frame_size = dy_buf.GetFrameSize();

            auto x_ptr           = m_x_buf.LockConst<T>();
            auto dy_ptr          = dy_buf.LockConst<T>();
            auto tmp_ptr         = m_tmp_buf.Lock<T>(true);
            auto input_index_ptr = m_input_index.LockConst();
            auto W_ptr           = lock_W_const();
            auto dW_ptr          = lock_dW();
            
            #pragma omp parallel for
            for ( index_t node = 0; node < node_size; ++node ) {
                T W[NN][2];
                for ( int i = 0; i < NN; ++i) {
                    T tmp = W_ptr(node, i);
                    if ( m_lut_binarize ) {
                        tmp = (tmp > (T)0.5) ? (T)1.0 : (T)0.0;
                    }
                    W[i][0] = -tmp;
                    W[i][1] = +tmp;
                }

                T dW[NN] = {0};
                for ( index_t frame = 0; frame < frame_size; ++frame ) {
                    T   x[N][2];
                    for ( int i = 0; i < N; ++i) {
                        T in_sig = x_ptr.Get(frame, input_index_ptr(node, i));
                        in_sig = std::min((T)1.0, std::max((T)0.0, in_sig));  // clip

                        x[i][0] = (T)1.0 - in_sig;
                        x[i][1] = in_sig;
                    }

                    T dy = dy_ptr.Get(frame, node);

                    for (int i = 0; i < NN; ++i) {
                        T dw = dy;
                        for (int j = 0; j < N; ++j) {
                            dw *= x[j][(i >> j) & 1];
                        }
                        dW[i] += dw;
                    }
                    
                    T dx0 = - dy * W[ 0][1] * x[5][0] * x[4][0] * x[3][0] * x[2][0] * x[1][0]
                            + dy * W[ 1][1] * x[5][0] * x[4][0] * x[3][0] * x[2][0] * x[1][0]
                            - dy * W[ 2][1] * x[5][0] * x[4][0] * x[3][0] * x[2][0] * x[1][1]
                            + dy * W[ 3][1] * x[5][0] * x[4][0] * x[3][0] * x[2][0] * x[1][1]
                            - dy * W[ 4][1] * x[5][0] * x[4][0] * x[3][0] * x[2][1] * x[1][0]
                            + dy * W[ 5][1] * x[5][0] * x[4][0] * x[3][0] * x[2][1] * x[1][0]
                            - dy * W[ 6][1] * x[5][0] * x[4][0] * x[3][0] * x[2][1] * x[1][1]
                            + dy * W[ 7][1] * x[5][0] * x[4][0] * x[3][0] * x[2][1] * x[1][1]
                            - dy * W[ 8][1] * x[5][0] * x[4][0] * x[3][1] * x[2][0] * x[1][0]
                            + dy * W[ 9][1] * x[5][0] * x[4][0] * x[3][1] * x[2][0] * x[1][0]
                            - dy * W[10][1] * x[5][0] * x[4][0] * x[3][1] * x[2][0] * x[1][1]
                            + dy * W[11][1] * x[5][0] * x[4][0] * x[3][1] * x[2][0] * x[1][1]
                            - dy * W[12][1] * x[5][0] * x[4][0] * x[3][1] * x[2][1] * x[1][0]
                            + dy * W[13][1] * x[5][0] * x[4][0] * x[3][1] * x[2][1] * x[1][0]
                            - dy * W[14][1] * x[5][0] * x[4][0] * x[3][1] * x[2][1] * x[1][1]
                            + dy * W[15][1] * x[5][0] * x[4][0] * x[3][1] * x[2][1] * x[1][1]
                            - dy * W[16][1] * x[5][0] * x[4][1] * x[3][0] * x[2][0] * x[1][0]
                            + dy * W[17][1] * x[5][0] * x[4][1] * x[3][0] * x[2][0] * x[1][0]
                            - dy * W[18][1] * x[5][0] * x[4][1] * x[3][0] * x[2][0] * x[1][1]
                            + dy * W[19][1] * x[5][0] * x[4][1] * x[3][0] * x[2][0] * x[1][1]
                            - dy * W[20][1] * x[5][0] * x[4][1] * x[3][0] * x[2][1] * x[1][0]
                            + dy * W[21][1] * x[5][0] * x[4][1] * x[3][0] * x[2][1] * x[1][0]
                            - dy * W[22][1] * x[5][0] * x[4][1] * x[3][0] * x[2][1] * x[1][1]
                            + dy * W[23][1] * x[5][0] * x[4][1] * x[3][0] * x[2][1] * x[1][1]
                            - dy * W[24][1] * x[5][0] * x[4][1] * x[3][1] * x[2][0] * x[1][0]
                            + dy * W[25][1] * x[5][0] * x[4][1] * x[3][1] * x[2][0] * x[1][0]
                            - dy * W[26][1] * x[5][0] * x[4][1] * x[3][1] * x[2][0] * x[1][1]
                            + dy * W[27][1] * x[5][0] * x[4][1] * x[3][1] * x[2][0] * x[1][1]
                            - dy * W[28][1] * x[5][0] * x[4][1] * x[3][1] * x[2][1] * x[1][0]
                            + dy * W[29][1] * x[5][0] * x[4][1] * x[3][1] * x[2][1] * x[1][0]
                            - dy * W[30][1] * x[5][0] * x[4][1] * x[3][1] * x[2][1] * x[1][1]
                            + dy * W[31][1] * x[5][0] * x[4][1] * x[3][1] * x[2][1] * x[1][1]
                            - dy * W[32][1] * x[5][1] * x[4][0] * x[3][0] * x[2][0] * x[1][0]
                            + dy * W[33][1] * x[5][1] * x[4][0] * x[3][0] * x[2][0] * x[1][0]
                            - dy * W[34][1] * x[5][1] * x[4][0] * x[3][0] * x[2][0] * x[1][1]
                            + dy * W[35][1] * x[5][1] * x[4][0] * x[3][0] * x[2][0] * x[1][1]
                            - dy * W[36][1] * x[5][1] * x[4][0] * x[3][0] * x[2][1] * x[1][0]
                            + dy * W[37][1] * x[5][1] * x[4][0] * x[3][0] * x[2][1] * x[1][0]
                            - dy * W[38][1] * x[5][1] * x[4][0] * x[3][0] * x[2][1] * x[1][1]
                            + dy * W[39][1] * x[5][1] * x[4][0] * x[3][0] * x[2][1] * x[1][1]
                            - dy * W[40][1] * x[5][1] * x[4][0] * x[3][1] * x[2][0] * x[1][0]
                            + dy * W[41][1] * x[5][1] * x[4][0] * x[3][1] * x[2][0] * x[1][0]
                            - dy * W[42][1] * x[5][1] * x[4][0] * x[3][1] * x[2][0] * x[1][1]
                            + dy * W[43][1] * x[5][1] * x[4][0] * x[3][1] * x[2][0] * x[1][1]
                            - dy * W[44][1] * x[5][1] * x[4][0] * x[3][1] * x[2][1] * x[1][0]
                            + dy * W[45][1] * x[5][1] * x[4][0] * x[3][1] * x[2][1] * x[1][0]
                            - dy * W[46][1] * x[5][1] * x[4][0] * x[3][1] * x[2][1] * x[1][1]
                            + dy * W[47][1] * x[5][1] * x[4][0] * x[3][1] * x[2][1] * x[1][1]
                            - dy * W[48][1] * x[5][1] * x[4][1] * x[3][0] * x[2][0] * x[1][0]
                            + dy * W[49][1] * x[5][1] * x[4][1] * x[3][0] * x[2][0] * x[1][0]
                            - dy * W[50][1] * x[5][1] * x[4][1] * x[3][0] * x[2][0] * x[1][1]
                            + dy * W[51][1] * x[5][1] * x[4][1] * x[3][0] * x[2][0] * x[1][1]
                            - dy * W[52][1] * x[5][1] * x[4][1] * x[3][0] * x[2][1] * x[1][0]
                            + dy * W[53][1] * x[5][1] * x[4][1] * x[3][0] * x[2][1] * x[1][0]
                            - dy * W[54][1] * x[5][1] * x[4][1] * x[3][0] * x[2][1] * x[1][1]
                            + dy * W[55][1] * x[5][1] * x[4][1] * x[3][0] * x[2][1] * x[1][1]
                            - dy * W[56][1] * x[5][1] * x[4][1] * x[3][1] * x[2][0] * x[1][0]
                            + dy * W[57][1] * x[5][1] * x[4][1] * x[3][1] * x[2][0] * x[1][0]
                            - dy * W[58][1] * x[5][1] * x[4][1] * x[3][1] * x[2][0] * x[1][1]
                            + dy * W[59][1] * x[5][1] * x[4][1] * x[3][1] * x[2][0] * x[1][1]
                            - dy * W[60][1] * x[5][1] * x[4][1] * x[3][1] * x[2][1] * x[1][0]
                            + dy * W[61][1] * x[5][1] * x[4][1] * x[3][1] * x[2][1] * x[1][0]
                            - dy * W[62][1] * x[5][1] * x[4][1] * x[3][1] * x[2][1] * x[1][1]
                            + dy * W[63][1] * x[5][1] * x[4][1] * x[3][1] * x[2][1] * x[1][1];

                    T dx1 = - dy * W[ 0][1] * x[5][0] * x[4][0] * x[3][0] * x[2][0] * x[0][0]
                            - dy * W[ 1][1] * x[5][0] * x[4][0] * x[3][0] * x[2][0] * x[0][1]
                            + dy * W[ 2][1] * x[5][0] * x[4][0] * x[3][0] * x[2][0] * x[0][0]
                            + dy * W[ 3][1] * x[5][0] * x[4][0] * x[3][0] * x[2][0] * x[0][1]
                            - dy * W[ 4][1] * x[5][0] * x[4][0] * x[3][0] * x[2][1] * x[0][0]
                            - dy * W[ 5][1] * x[5][0] * x[4][0] * x[3][0] * x[2][1] * x[0][1]
                            + dy * W[ 6][1] * x[5][0] * x[4][0] * x[3][0] * x[2][1] * x[0][0]
                            + dy * W[ 7][1] * x[5][0] * x[4][0] * x[3][0] * x[2][1] * x[0][1]
                            - dy * W[ 8][1] * x[5][0] * x[4][0] * x[3][1] * x[2][0] * x[0][0]
                            - dy * W[ 9][1] * x[5][0] * x[4][0] * x[3][1] * x[2][0] * x[0][1]
                            + dy * W[10][1] * x[5][0] * x[4][0] * x[3][1] * x[2][0] * x[0][0]
                            + dy * W[11][1] * x[5][0] * x[4][0] * x[3][1] * x[2][0] * x[0][1]
                            - dy * W[12][1] * x[5][0] * x[4][0] * x[3][1] * x[2][1] * x[0][0]
                            - dy * W[13][1] * x[5][0] * x[4][0] * x[3][1] * x[2][1] * x[0][1]
                            + dy * W[14][1] * x[5][0] * x[4][0] * x[3][1] * x[2][1] * x[0][0]
                            + dy * W[15][1] * x[5][0] * x[4][0] * x[3][1] * x[2][1] * x[0][1]
                            - dy * W[16][1] * x[5][0] * x[4][1] * x[3][0] * x[2][0] * x[0][0]
                            - dy * W[17][1] * x[5][0] * x[4][1] * x[3][0] * x[2][0] * x[0][1]
                            + dy * W[18][1] * x[5][0] * x[4][1] * x[3][0] * x[2][0] * x[0][0]
                            + dy * W[19][1] * x[5][0] * x[4][1] * x[3][0] * x[2][0] * x[0][1]
                            - dy * W[20][1] * x[5][0] * x[4][1] * x[3][0] * x[2][1] * x[0][0]
                            - dy * W[21][1] * x[5][0] * x[4][1] * x[3][0] * x[2][1] * x[0][1]
                            + dy * W[22][1] * x[5][0] * x[4][1] * x[3][0] * x[2][1] * x[0][0]
                            + dy * W[23][1] * x[5][0] * x[4][1] * x[3][0] * x[2][1] * x[0][1]
                            - dy * W[24][1] * x[5][0] * x[4][1] * x[3][1] * x[2][0] * x[0][0]
                            - dy * W[25][1] * x[5][0] * x[4][1] * x[3][1] * x[2][0] * x[0][1]
                            + dy * W[26][1] * x[5][0] * x[4][1] * x[3][1] * x[2][0] * x[0][0]
                            + dy * W[27][1] * x[5][0] * x[4][1] * x[3][1] * x[2][0] * x[0][1]
                            - dy * W[28][1] * x[5][0] * x[4][1] * x[3][1] * x[2][1] * x[0][0]
                            - dy * W[29][1] * x[5][0] * x[4][1] * x[3][1] * x[2][1] * x[0][1]
                            + dy * W[30][1] * x[5][0] * x[4][1] * x[3][1] * x[2][1] * x[0][0]
                            + dy * W[31][1] * x[5][0] * x[4][1] * x[3][1] * x[2][1] * x[0][1]
                            - dy * W[32][1] * x[5][1] * x[4][0] * x[3][0] * x[2][0] * x[0][0]
                            - dy * W[33][1] * x[5][1] * x[4][0] * x[3][0] * x[2][0] * x[0][1]
                            + dy * W[34][1] * x[5][1] * x[4][0] * x[3][0] * x[2][0] * x[0][0]
                            + dy * W[35][1] * x[5][1] * x[4][0] * x[3][0] * x[2][0] * x[0][1]
                            - dy * W[36][1] * x[5][1] * x[4][0] * x[3][0] * x[2][1] * x[0][0]
                            - dy * W[37][1] * x[5][1] * x[4][0] * x[3][0] * x[2][1] * x[0][1]
                            + dy * W[38][1] * x[5][1] * x[4][0] * x[3][0] * x[2][1] * x[0][0]
                            + dy * W[39][1] * x[5][1] * x[4][0] * x[3][0] * x[2][1] * x[0][1]
                            - dy * W[40][1] * x[5][1] * x[4][0] * x[3][1] * x[2][0] * x[0][0]
                            - dy * W[41][1] * x[5][1] * x[4][0] * x[3][1] * x[2][0] * x[0][1]
                            + dy * W[42][1] * x[5][1] * x[4][0] * x[3][1] * x[2][0] * x[0][0]
                            + dy * W[43][1] * x[5][1] * x[4][0] * x[3][1] * x[2][0] * x[0][1]
                            - dy * W[44][1] * x[5][1] * x[4][0] * x[3][1] * x[2][1] * x[0][0]
                            - dy * W[45][1] * x[5][1] * x[4][0] * x[3][1] * x[2][1] * x[0][1]
                            + dy * W[46][1] * x[5][1] * x[4][0] * x[3][1] * x[2][1] * x[0][0]
                            + dy * W[47][1] * x[5][1] * x[4][0] * x[3][1] * x[2][1] * x[0][1]
                            - dy * W[48][1] * x[5][1] * x[4][1] * x[3][0] * x[2][0] * x[0][0]
                            - dy * W[49][1] * x[5][1] * x[4][1] * x[3][0] * x[2][0] * x[0][1]
                            + dy * W[50][1] * x[5][1] * x[4][1] * x[3][0] * x[2][0] * x[0][0]
                            + dy * W[51][1] * x[5][1] * x[4][1] * x[3][0] * x[2][0] * x[0][1]
                            - dy * W[52][1] * x[5][1] * x[4][1] * x[3][0] * x[2][1] * x[0][0]
                            - dy * W[53][1] * x[5][1] * x[4][1] * x[3][0] * x[2][1] * x[0][1]
                            + dy * W[54][1] * x[5][1] * x[4][1] * x[3][0] * x[2][1] * x[0][0]
                            + dy * W[55][1] * x[5][1] * x[4][1] * x[3][0] * x[2][1] * x[0][1]
                            - dy * W[56][1] * x[5][1] * x[4][1] * x[3][1] * x[2][0] * x[0][0]
                            - dy * W[57][1] * x[5][1] * x[4][1] * x[3][1] * x[2][0] * x[0][1]
                            + dy * W[58][1] * x[5][1] * x[4][1] * x[3][1] * x[2][0] * x[0][0]
                            + dy * W[59][1] * x[5][1] * x[4][1] * x[3][1] * x[2][0] * x[0][1]
                            - dy * W[60][1] * x[5][1] * x[4][1] * x[3][1] * x[2][1] * x[0][0]
                            - dy * W[61][1] * x[5][1] * x[4][1] * x[3][1] * x[2][1] * x[0][1]
                            + dy * W[62][1] * x[5][1] * x[4][1] * x[3][1] * x[2][1] * x[0][0]
                            + dy * W[63][1] * x[5][1] * x[4][1] * x[3][1] * x[2][1] * x[0][1];

                    for (int i = 0; i < N; ++i) {
                        T dx = 0;
                        for (int j = 0; j < NN; ++j) {
                            T w = W[j][(j >> i) & 1];
                            for (int k = 0; k < N; ++k) {
                                if (i != k) {
                                    w *= x[k][(j >> k) & 1];
                                }
                            }
                            dx += w;
                        }
                        dx *= dy;
                        tmp_ptr.Set(frame, node * N + i, dx);
                    }
                }
                for ( int i = 0; i < NN; ++i) {
                    dW_ptr(node, i) += dW[i];
                }
            }

            auto dx_ptr = m_dx_buf.Lock<T>();

            #pragma omp parallel for
            for ( index_t frame = 0; frame < frame_size; ++frame ) {
                for ( index_t node = 0; node < node_size; ++node ) {
                    for (int i = 0; i < N; ++i) {
                        T dx = tmp_ptr.Get(frame, node * N + i);
                        auto input_node = input_index_ptr(node, i);
                        dx_ptr.Add(frame, input_node, dx);
                    }
                }
            }

            return m_dx_buf;
        }
    }
};


}
