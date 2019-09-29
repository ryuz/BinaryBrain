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
#include "bb/FixedSizeConnectionTable.h"
#include "bb/StochasticOperation.h"
#include "bb/StochasticLutSimd.h"


namespace bb {


// 確率的LUTの抽象レイヤー
template <int N = 6, typename BinType = float, typename RealType = float>
class StochasticLutN : public SparseLayer
{
    using _super = SparseLayer;
    static int const NN = (1 << N);

protected:
    bool                        m_binary_mode  = (DataType<BinType>::type == BB_TYPE_BIT);
    bool                        m_lut_binarize = true;

    bool                        m_y_binarize = false;
    bool                        m_host_only = false;
    bool                        m_host_simd = true;

    index_t                     m_max_tmp_mem_size = 256 * 1024 * 1024;

    std::string                 m_connection;

    RealType                    m_unbinarize_bias = (RealType)0.25;

    indices_t                   m_input_shape;
    indices_t                   m_output_shape;

    FrameBuffer                 m_x_buf;

    FixedSizeConnectionTable<N> m_connection_table;

    std::shared_ptr<Tensor>     m_W;
    std::shared_ptr<Tensor>     m_dW;

    std::mt19937_64             m_mt;

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
        // バイナリモード設定
        if (DataType<BinType>::type != BB_TYPE_BIT) {
            if ( args.size() == 2 && args[0] == "binary")
            {
                m_binary_mode = EvalBool(args[1]);
            }
        }

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

    std::string GetClassName(void) const { return "StochasticLut" + std::to_string(N); }


public:
    // Serialize
    void Save(std::ostream &os) const 
    {
        SaveIndices(os, m_input_shape);
        SaveIndices(os, m_output_shape);
        m_connection_table.Save(os);
        m_W->Save(os);
    }

    void Load(std::istream &is)
    {
        m_input_shape  = LoadIndices(is);
        m_output_shape = LoadIndices(is);
        m_connection_table.Load(is);
        m_W->Load(is);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
        archive(cereal::make_nvp("input_shape",      m_input_shape));
        archive(cereal::make_nvp("output_shape",     m_output_shape));
        archive(cereal::make_nvp("connection_table", m_connection_table));
        archive(cereal::make_nvp("W",                *m_W));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("input_shape",      m_input_shape));
        archive(cereal::make_nvp("output_shape",     m_output_shape));
        archive(cereal::make_nvp("connection_table", m_connection_table));
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

    auto lock_W(void)              { return m_W->Lock<RealType>(); }
    auto lock_W_const(void) const  { return m_W->LockConst<RealType>(); }
    auto lock_dW(void)             { return m_dW->Lock<RealType>(); }
    auto lock_dW_const(void) const { return m_dW->LockConst<RealType>(); }


    // 接続管理
    index_t GetNodeConnectionSize(index_t node) const
    {
        return m_connection_table.GetInputConnectionSize(node);
    }

    void SetNodeConnectionIndex(index_t node, index_t input_index, index_t input_node)
    {
        m_connection_table.SetInputConnection(node, input_index, input_node);
    }

    index_t GetNodeConnectionIndex(index_t node, index_t input_index) const
    {
        return m_connection_table.GetInputConnection(node, input_index);
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
        m_connection_table.SetShape(m_input_shape, m_output_shape);
        m_connection_table.InitializeConnection(m_mt(), m_connection);

//        auto output_node_size = GetShapeSize(m_output_shape);
//        m_input_index.Resize(output_node_size, N);
//        this->InitializeNodeInput(m_mt(), m_connection);

        // パラメータ初期化(結局初期値は何が良いのかまだよくわからない)
//      m_W->Resize({NN, m_output_node_size}, DataType<RealType>::type);  m_W->InitUniformDistribution(0.4, 0.6, m_mt());
//      m_W->Resize({NN, m_output_node_size}, DataType<RealType>::type);  m_W->InitUniformDistribution(0.0, 1.0, m_mt());
//      m_W->Resize({NN, m_output_node_size}, DataType<RealType>::type);  m_W->InitNormalDistribution(0.5, 0.001, m_mt());
        m_W->Resize({NN, GetShapeSize(m_output_shape)}, DataType<RealType>::type);  m_W->InitNormalDistribution(0.5, 0.01, m_mt());

        m_dW->Resize({NN, GetShapeSize(m_output_shape)}, DataType<RealType>::type); m_dW->FillZero();

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

        auto W_ptr = lock_W_const();
        RealType W[(1 << N)];
        for ( int i = 0; i < (1 << N); ++i) {
            W[i] = W_ptr(node, i);
            W[i] = std::min((RealType)1.0, std::max((RealType)0.0, W[i]));  // clip

            if ( m_lut_binarize ) {
                W[i] = W[i] > (RealType)0.5 ? (RealType)1.0 : (RealType)0.0;
            }
        }

        RealType   x[N][2];
        for ( int i = 0; i < N; ++i) {
            RealType x_tmp = (RealType)input_value[i];
            if ( m_binary_mode ) {
                x_tmp = (RealType)0.5 + ((x_tmp > (RealType)0.5) ? +m_unbinarize_bias : -m_unbinarize_bias);  // unbinarize
            }
            else {
                x_tmp = std::min((RealType)1.0, std::max((RealType)0.0, x_tmp));  // clip
            }
            x[i][0] = (RealType)1.0 - x_tmp;
            x[i][1] = x_tmp;
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
        FrameBuffer y_buf(x_buf.GetFrameSize(), m_output_shape, DataType<RealType>::type);

        // backwardの為に保存
        if ( train ) {
            m_x_buf = x_buf;
        }


        // パラメータクリップ
        m_W->Clamp((RealType)0.0, (RealType)1.0);


#ifdef BB_WITH_CUDA
        // LUT6 FP32 CUDA
        if ( N == 6 && DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            auto x_ptr           = x_buf.LockDeviceMemoryConst();
            auto y_ptr           = y_buf.LockDeviceMemory(true);
            auto input_table_ptr = m_connection_table.LockDeviceMemConst_InputTable();
            auto W_ptr           = m_W->LockDeviceMemoryConst();
               
            bbcu_fp32_StochasticLut6_Forward(
                    (float const *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int   const *)input_table_ptr.GetAddr(),
                    (float const *)W_ptr.GetAddr(),
                    (int          )y_buf.GetNodeSize(),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float)),
                    (int          )(m_binary_mode  ? 1 : 0),
                    (int          )(m_lut_binarize ? 1 : 0),
                    (float        )m_unbinarize_bias
                );

            return y_buf;
        }

        // LUT6 Bit CUDA
        if ( N == 6 && DataType<BinType>::type == BB_TYPE_BIT && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            auto x_ptr           = x_buf.LockDeviceMemoryConst();
            auto y_ptr           = y_buf.LockDeviceMemory(true);
            auto input_table_ptr = m_connection_table.LockDeviceMemConst_InputTable();
            auto W_ptr           = m_W->LockDeviceMemoryConst();
            
            bbcu_bit_fp32_StochasticLut6_Forward(
                    (int   const *)x_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int   const *)input_table_ptr.GetAddr(),
                    (float const *)W_ptr.GetAddr(),
                    (int          )y_buf.GetNodeSize(),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float)),
                    (int          )(x_buf.GetFrameStride() / sizeof(int)),
                    (int          )(m_lut_binarize ? 1 : 0),
                    (float        )m_unbinarize_bias
                );

            return y_buf;
        }
#endif

        // LUT6 SIMD
        if ( N == 6 && DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32 && m_host_simd
                && y_buf.GetFrameSize() % 8 == 0 ) {
            auto input_table_ptr = m_connection_table.LockConst_InputTable();
            simd_fp32_StochasticLut6_Forward(x_buf, y_buf, input_table_ptr.GetAddr(), m_W, m_binary_mode, m_lut_binarize, m_unbinarize_bias);
            return y_buf;
        }

        {
            // Generic
            auto node_size  = y_buf.GetNodeSize();
            auto frame_size = y_buf.GetFrameSize();

            auto x_ptr           = x_buf.LockConst<BinType>();
            auto y_ptr           = y_buf.Lock<RealType>();
            auto input_table_ptr = m_connection_table.LockConst_InputTable();
            auto W_ptr           = lock_W_const();

            #pragma omp parallel for
            for ( index_t node = 0; node < node_size; ++node ) {
                // read W
                RealType W[(1 << N)];
                for ( int i = 0; i < (1 << N); ++i) {
                    W[i] = W_ptr(node, i);
                    if ( m_lut_binarize ) {
                        W[i] = W[i] > (RealType)0.5 ? (RealType)1.0 : (RealType)0.0;   // binarize
                    }
                }

                for ( index_t frame = 0; frame < frame_size; ++frame ) {
                    // read x
                    RealType    x[N];
                    for ( int i = 0; i < N; ++i) {
                        RealType x_tmp = (RealType)x_ptr.Get(frame, input_table_ptr(node, i));
                        if ( m_binary_mode || DataType<BinType>::type == BB_TYPE_BIT ) {
                            x[i] = (RealType)0.5 + (x_tmp > (RealType)0.5 ? +m_unbinarize_bias : -m_unbinarize_bias);   // unbinarize
                        }
                        else {
                            x[i] = std::min((RealType)1.0, std::max((RealType)0.0, x_tmp));  // clip
                        }
                    }

                    // calculate
                    RealType    y;
                    StochasticOperation_Lut_Forward<RealType>(x, &y, W, N);

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
     

#ifdef BB_WITH_CUDA
        // LUT6 FP32 CUDA
        if ( N == 6 && DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
                && dy_buf.IsDeviceAvailable() && x_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {

            // tmp buffer
            index_t tmp_frame_size = m_max_tmp_mem_size / (sizeof(float) * this->GetOutputNodeSize()*N);
            tmp_frame_size = std::max(tmp_frame_size, (index_t)32);
            tmp_frame_size = ((tmp_frame_size + 31) & ~0x1f);
            tmp_frame_size = std::min(tmp_frame_size, dy_buf.GetFrameSize());
            FrameBuffer tmp_buf(tmp_frame_size, {this->GetOutputNodeSize()*N}, DataType<RealType>::type);

            auto x_ptr             = x_buf.LockDeviceMemoryConst();
            auto dy_ptr            = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr            = dx_buf.LockDeviceMemory(true);
            auto reverse_table_ptr = m_connection_table.LockDeviceMemConst_ReverseTable();
            auto input_table_ptr   = m_connection_table.LockDeviceMemConst_InputTable();
            auto W_ptr             = m_W->LockDeviceMemoryConst();
            auto dW_ptr            = m_dW->LockDeviceMemory();
            auto tmp_ptr           = tmp_buf.LockDeviceMemory();
            
            bbcu_fp32_StochasticLut6_Backward(
                    (float const *)x_ptr.GetAddr(),
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (float       *)tmp_ptr.GetAddr(),
                    (int   const *)input_table_ptr.GetAddr(),
                    (int   const *)reverse_table_ptr.GetAddr(),
                    (float const *)W_ptr.GetAddr(),
                    (float       *)dW_ptr.GetAddr(),
                    (int          )m_connection_table.GetReverseTableStride(),
                    (int          )dx_buf.GetNodeSize(),
                    (int          )dy_buf.GetNodeSize(),
                    (int          )dx_buf.GetFrameSize(),
                    (int          )(dx_buf.GetFrameStride() / sizeof(float)),
                    (int          )tmp_buf.GetFrameSize(),
                    (int          )(tmp_buf.GetFrameStride() / sizeof(float)),
                    (int          )(m_binary_mode  ? 1 : 0),
                    (int          )(m_lut_binarize ? 1 : 0),
                    (float        )m_unbinarize_bias
                );
            
            return dx_buf;
        }

        // LUT6 Bit CUDA
        if ( N == 6 && DataType<BinType>::type == BB_TYPE_BIT && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
                && dy_buf.IsDeviceAvailable() && x_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {

            // tmp buffer
            index_t tmp_frame_size = m_max_tmp_mem_size / (sizeof(float) * this->GetOutputNodeSize()*N);
            tmp_frame_size = std::max(tmp_frame_size, (index_t)32);
            tmp_frame_size = ((tmp_frame_size + 31) & ~0x1f);
            tmp_frame_size = std::min(tmp_frame_size, dy_buf.GetFrameSize());
            FrameBuffer tmp_buf(tmp_frame_size, {this->GetOutputNodeSize()*N}, DataType<RealType>::type);

            auto x_ptr             = x_buf.LockDeviceMemoryConst();
            auto dy_ptr            = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr            = dx_buf.LockDeviceMemory(true);
            auto reverse_table_ptr = m_connection_table.LockDeviceMemConst_ReverseTable();
            auto input_table_ptr   = m_connection_table.LockDeviceMemConst_InputTable();
            auto W_ptr             = m_W->LockDeviceMemoryConst();
            auto dW_ptr            = m_dW->LockDeviceMemory();
            auto tmp_ptr           = tmp_buf.LockDeviceMemory();
            
            bbcu_bit_fp32_StochasticLut6_Backward(
                    (int   const *)x_ptr.GetAddr(),
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)dx_ptr.GetAddr(),
                    (float       *)tmp_ptr.GetAddr(),
                    (int   const *)input_table_ptr.GetAddr(),
                    (int   const *)reverse_table_ptr.GetAddr(),
                    (float const *)W_ptr.GetAddr(),
                    (float       *)dW_ptr.GetAddr(),
                    (int          )m_connection_table.GetReverseTableStride(),
                    (int          )dx_buf.GetNodeSize(),
                    (int          )dy_buf.GetNodeSize(),
                    (int          )dx_buf.GetFrameSize(),
                    (int          )(dx_buf.GetFrameStride() / sizeof(float)),
                    (int          )(x_buf.GetFrameStride() / sizeof(int)),
                    (int          )tmp_buf.GetFrameSize(),
                    (int          )(tmp_buf.GetFrameStride() / sizeof(float)),
                    (int          )(m_lut_binarize ? 1 : 0),
                    (float        )m_unbinarize_bias
                );
            
            return dx_buf;
        }
#endif

        // LUT6 SIMD
        if ( N == 6 && DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32 && m_host_simd
                && dy_buf.GetFrameSize() % 8 == 0 ) {
            auto input_table_ptr = m_connection_table.LockConst_InputTable();
            simd_fp32_StochasticLut6_Backward(x_buf, dy_buf, dx_buf, input_table_ptr.GetAddr(), m_W, m_dW, m_unbinarize_bias, m_binary_mode, m_lut_binarize);
            return dx_buf;
        }

        {
            FrameBuffer tmp_buf(dy_buf.GetFrameSize(), {GetShapeSize(m_output_shape)*N}, DataType<RealType>::type);

            // generic
            dx_buf.FillZero();

            auto node_size  = dy_buf.GetNodeSize();
            auto frame_size = dy_buf.GetFrameSize();

            auto x_ptr           = x_buf.LockConst<BinType>();
            auto dy_ptr          = dy_buf.LockConst<RealType>();
            auto tmp_ptr         = tmp_buf.Lock<RealType>(true);
            auto input_table_ptr = m_connection_table.LockConst_InputTable();
            auto W_ptr           = lock_W_const();
            auto dW_ptr          = lock_dW();
            
            #pragma omp parallel for
            for ( index_t node = 0; node < node_size; ++node ) {
                // read W
                RealType W[1 << N];
                for ( int i = 0; i < NN; ++i) {
                    W[i] = W_ptr(node, i);
                    if ( m_lut_binarize ) {
                        W[i] = (W[i] > (RealType)0.5) ? (RealType)1.0 : (RealType)0.0;
                    }
                }

                // setup dW
                RealType dW[NN] = {0};

                for ( index_t frame = 0; frame < frame_size; ++frame ) {
                    // read x
                    RealType    x[N];
                    for ( int i = 0; i < N; ++i) {
                        RealType x_tmp = (RealType)x_ptr.Get(frame, input_table_ptr(node, i));
                        if ( m_binary_mode || DataType<BinType>::type == BB_TYPE_BIT ) {
                            x[i] = (RealType)0.5 + (x_tmp > (RealType)0.5 ? +m_unbinarize_bias : -m_unbinarize_bias);   // unbinarize
                        }
                        else {
                            x[i] = std::min((RealType)1.0, std::max((RealType)0.0, x_tmp));  // clip
                        }
                    }

                    // read dy
                    RealType dy = dy_ptr.Get(frame, node);

                    // calculate
                    RealType    dx[N];
                    StochasticOperation_Lut_Backward<RealType>(x, dx, &dy, W, dW, N);

                    // write dx
                    for (int i = 0; i < N; ++i) {
                        tmp_ptr.Set(frame, node * N + i, dx[i]);
                    }
                }

                // write dW
                for ( int i = 0; i < NN; ++i) {
                    dW_ptr(node, i) += dW[i];
                }
            }

            // integrate dx
            auto dx_ptr = dx_buf.Lock<RealType>();
            #pragma omp parallel for
            for ( index_t frame = 0; frame < frame_size; ++frame ) {
                for ( index_t node = 0; node < node_size; ++node ) {
                    for (int i = 0; i < N; ++i) {
                        RealType dx = tmp_ptr.Get(frame, node * N + i);
                        auto input_node = input_table_ptr(node, i);
                        dx_ptr.Add(frame, input_node, dx);
                    }
                }
            }

            return dx_buf;
        }
    }
};


}
