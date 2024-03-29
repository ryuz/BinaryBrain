﻿// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2022 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>

#include "bb/DataType.h"
#include "bb/Model.h"

#ifdef BB_WITH_CUDA
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "bbcu/bbcu.h"
#endif


namespace bb {


// DepthwiseAffineレイヤー (量子化対応版)
template <typename T = float>
class DepthwiseDenseAffineQuantize : public Model
{
    using _super = Model;

public:
    static inline std::string ModelName(void) { return "DepthwiseDenseAffineQuantize"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<T>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    bool                        m_host_only = false;
    bool                        m_binary_mode = false;

    T                           m_initialize_std = (T)0.01;
    std::string                 m_initializer = "";
    std::mt19937_64             m_mt;

    indices_t                   m_input_shape;
    index_t                     m_input_point_size = 0;
    index_t                     m_input_node_size = 0;
    indices_t                   m_output_shape;
    index_t                     m_output_point_size = 0;
    index_t                     m_output_node_size = 0;
    index_t                     m_depth_size = 0;

    std::shared_ptr<Tensor>     m_W;
    std::shared_ptr<Tensor>     m_b;
    std::shared_ptr<Tensor>     m_dW;
    std::shared_ptr<Tensor>     m_db;

    Tensor                      m_WQ;
    Tensor                      m_bQ;

    bool                        m_quantize = true;
    int                         m_weight_bits  = 8;
    int                         m_output_bits  = 16;
    int                         m_input_bits   = 0;
    T                           m_weight_scale = (T)1/(T)(1 << 8);
    T                           m_output_scale = (T)1/(T)(1 << 8);
    T                           m_input_scale  = (T)1/(T)(1 << 8);

    bool                        m_cublasEnable = false;
#ifdef BB_WITH_CUDA
    cublasHandle_t              m_cublasHandle;
#endif

public:
    struct create_t
    {
        indices_t       output_shape;
        index_t         input_point_size = 0;
        index_t         depth_size = 0;
        bool            quantize = true;
        int             weight_bits  = 8;
        int             output_bits  = 16;
        int             input_bits   = 0;
        T               weight_scale = (T)1/(T)(1 << 8);
        T               output_scale = (T)1/(T)(1 << 8);
        T               input_scale  = (T)1/(T)(1 << 8);
        T               initialize_std = (T)0.01;
        std::string     initializer = "";
        std::uint64_t   seed = 1;
    };

protected:
    DepthwiseDenseAffineQuantize(create_t const &create)
    {
        m_W = std::make_shared<Tensor>();
        m_b = std::make_shared<Tensor>();
        m_dW = std::make_shared<Tensor>();
        m_db = std::make_shared<Tensor>();

#ifdef BB_WITH_CUDA
        if ( cublasCreate(&m_cublasHandle) == CUBLAS_STATUS_SUCCESS ) {
            m_cublasEnable = true;
        }
#endif

//      BB_ASSERT(!create.output_shape.empty());

        m_quantize     = create.quantize;
        m_weight_bits  = create.weight_bits;
        m_output_bits  = create.output_bits;
        m_input_bits   = create.input_bits;
        m_weight_scale = create.weight_scale;
        m_output_scale = create.output_scale;
        m_input_scale  = create.input_scale;
        if ( m_weight_bits > 0 && m_weight_scale <= 0 ) { m_weight_scale = (T)1 / (T)(1 << (m_weight_bits-1)); }
        if ( m_output_bits > 0 && m_output_scale <= 0 ) { m_output_scale = (T)1 / (T)(1 << (m_output_bits-1)); }
        if ( m_input_bits  > 0 && m_input_scale  <= 0 ) { m_input_scale  = (T)1 / (T)(1 << (m_input_bits-1));  }

        m_initialize_std  = create.initialize_std;
        m_initializer     = create.initializer;
        m_mt.seed(create.seed);

        m_output_shape     = create.output_shape;
        m_output_node_size = CalcShapeSize(m_output_shape);
        m_depth_size       = create.depth_size;
        m_input_point_size = create.input_point_size;
    }

    void CommandProc(std::vector<std::string> args) override
    {
        _super::CommandProc(args);

        // 量子化モード設定
        if ( args.size() == 2 && args[0] == "quantize" )
        {
            m_quantize = EvalBool(args[1]);
        }

        // バイナリモード設定
        if ( args.size() == 2 && args[0] == "binary" )
        {
            m_binary_mode = EvalBool(args[1]);
        }

        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }
    }

    void PrintInfoText(std::ostream& os, std::string indent, int columns, int nest, int depth) const override
    {
        _super::PrintInfoText(os, indent, columns, nest, depth);
//      os << indent << " input  shape : " << GetInputShape();
//      os << indent << " output shape : " << GetOutputShape();
        os << indent << " input(" << m_input_point_size << ", " << m_depth_size << ")"
                     << " output(" << m_output_point_size << ", " << m_depth_size << ")" << std::endl;
    }

public:
    ~DepthwiseDenseAffineQuantize() {
#ifdef BB_WITH_CUDA
        if ( m_cublasEnable ) {
            BB_CUBLAS_SAFE_CALL(cublasDestroy(m_cublasHandle));
            m_cublasEnable = false;
        }
#endif
    }

    static std::shared_ptr<DepthwiseDenseAffineQuantize> Create(create_t const &create)
    {
        return std::shared_ptr<DepthwiseDenseAffineQuantize>(new DepthwiseDenseAffineQuantize(create));
    }

    static std::shared_ptr<DepthwiseDenseAffineQuantize> Create(indices_t const &output_shape, index_t input_point_size=0, index_t depth_size=0,
                                                                    bool quantize=true,
                                                                    int weight_bits =8, int output_bits=16, int input_bits=0,
                                                                    T weight_scale=(T)1/(T)(1 << 8), T output_scale=(T)1/(T)(1 << 8), T input_scale= (T)1/(T)(1 << 8))
    {
        create_t create;
        create.output_shape     = output_shape;
        create.input_point_size = input_point_size;
        create.depth_size       = depth_size;
        create.quantize         = quantize;
        create.weight_bits      = weight_bits;
        create.output_bits      = output_bits;
        create.input_bits       = input_bits;
        create.weight_scale     = weight_scale;
        create.output_scale     = output_scale;
        create.input_scale      = input_scale;
        return Create(create);
    }

    static std::shared_ptr<DepthwiseDenseAffineQuantize> Create(index_t output_node_size, index_t input_point_size=0, index_t depth_size=0,
                                                                    bool quantize=true,
                                                                    int weight_bits =8, int output_bits=16, int input_bits=0,
                                                                    T weight_scale=(T)1/(T)(1 << 8), T output_scale=(T)1/(T)(1 << 8), T input_scale= (T)1/(T)(1 << 8))
    {
        create_t create;
        create.output_shape.resize(1);
        create.output_shape[0] = output_node_size;
        create.quantize         = quantize;
        create.weight_bits      = weight_bits;
        create.output_bits      = output_bits;
        create.input_bits       = input_bits;
        create.weight_scale     = weight_scale;
        create.output_scale     = output_scale;
        create.input_scale      = input_scale;
        return Create(indices_t({output_node_size}), input_point_size, depth_size);
    }

    static std::shared_ptr<DepthwiseDenseAffineQuantize> Create(void)
    {
        return Create(create_t());
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<DepthwiseDenseAffineQuantize> CreatePy(
            indices_t       output_shape,
            index_t         input_point_size = 0,
            index_t         depth_size = 0,
            bool            quantize =true,
            int             weight_bits  = 8,
            int             output_bits  = 16,
            int             input_bits   = 0,
            T               weight_scale = (T)1/(T)(1 << 8),
            T               output_scale = (T)1/(T)(1 << 8),
            T               input_scale  = (T)1/(T)(1 << 8),
            T               initialize_std = (T)0.01,
            std::string     initializer = "he",
            std::uint64_t   seed = 1
        )
    {
        create_t create;
        create.output_shape      = output_shape;
        create.input_point_size  = input_point_size;
        create.depth_size        = depth_size;
        create.quantize          = quantize;
        create.weight_bits       = weight_bits;
        create.output_bits       = output_bits;
        create.input_bits        = input_bits;
        create.weight_scale      = weight_scale;
        create.output_scale      = output_scale;
        create.input_scale       = input_scale;
        create.initialize_std    = initialize_std;
        create.initializer       = initializer;
        create.seed              = seed;
        return Create(create);
    }
#endif


    Tensor       &W(void)       { return *m_W; }
    Tensor const &W(void) const { return *m_W; }
    Tensor       &b(void)       { return *m_b; }
    Tensor const &b(void) const { return *m_b; }
   
    Tensor       &dW(void)       { return *m_dW; }
    Tensor const &dW(void) const { return *m_dW; }
    Tensor       &db(void)       { return *m_db; }
    Tensor const &db(void) const { return *m_db; }

    auto lock_W(void)             { return m_W->Lock<T>(); }
    auto lock_W_const(void) const { return m_W->LockConst<T>(); }
    auto lock_b(void)             { return m_b->Lock<T>(); }
    auto lock_b_const(void) const { return m_b->LockConst<T>(); }

    auto lock_dW(void)             { return m_dW->Lock<T>(); }
    auto lock_dW_const(void) const { return m_dW->LockConst<T>(); }
    auto lock_db(void)             { return m_db->Lock<T>(); }
    auto lock_db_const(void) const { return m_db->LockConst<T>(); }

    Tensor WQ(void){
        if ( m_quantize && m_weight_bits > 0 ) {
            return m_W->Quantize(m_weight_bits, m_weight_scale);
        }
        else {
            return *m_W;
        }
    }
    Tensor bQ(void) {
        if ( m_quantize && m_weight_bits > 0 ) {
            return m_b->Quantize(m_weight_bits, m_weight_scale);
        }
        else {
            return *m_b;
        }
    }


   /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        BB_ASSERT(!shape.empty());

        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

        // 形状設定
        m_input_shape   = shape;
        m_input_node_size = CalcShapeSize(shape);

        if ( m_depth_size <= 0 ) {
            if ( m_input_point_size > 0 ) {
                m_depth_size = m_input_node_size / m_input_point_size;
            }
            else
            {
                m_depth_size = m_output_shape[0];
            }
        }

        BB_ASSERT(m_output_node_size > 0);
        BB_ASSERT(m_depth_size > 0);
        BB_ASSERT(m_output_node_size % m_depth_size == 0);
        BB_ASSERT(m_input_node_size % m_depth_size == 0);
        m_input_point_size  = m_input_node_size / m_depth_size;
        m_output_point_size = m_output_node_size / m_depth_size;


        // パラメータ初期化
        m_W->Resize ({m_depth_size, m_output_point_size, m_input_point_size}, DataType<T>::type);
        m_b->Resize ({m_depth_size, m_output_point_size},                     DataType<T>::type);
        m_dW->Resize({m_depth_size, m_output_point_size, m_input_point_size}, DataType<T>::type);
        m_db->Resize({m_depth_size, m_output_point_size},                     DataType<T>::type);

        if (m_initializer == "he" || m_initializer == "He") {
            m_initialize_std = (T)std::sqrt(2.0 / (double)m_input_node_size);
            m_W->InitNormalDistribution(0.0, m_initialize_std, m_mt());
            m_b->InitNormalDistribution(0.0, m_initialize_std, m_mt());
        }
        else if (m_initializer == "xavier" || m_initializer == "Xavier" ) {
            m_initialize_std = (T)std::sqrt(1.0 / (double)m_input_node_size);
            m_W->InitNormalDistribution(0.0, m_initialize_std, m_mt());
            m_b->InitNormalDistribution(0.0, m_initialize_std, m_mt());
        }
        else if (m_initializer == "normal" || m_initializer == "Normal" ) {
            m_W->InitNormalDistribution(0.0, m_initialize_std, m_mt());
            m_b->InitNormalDistribution(0.0, m_initialize_std, m_mt());
        }
        else if (m_initializer == "uniform" || m_initializer == "Uniform" ) {
            double k = m_initialize_std * std::sqrt(3);
            m_W->InitUniformDistribution(-k, +k, m_mt());
            m_b->InitUniformDistribution(-k, +k, m_mt());
        }
        else {
            double k = std::sqrt(1.0 / (double)m_input_node_size);
            m_W->InitUniformDistribution(-k, +k, m_mt());
            m_b->InitUniformDistribution(-k, +k, m_mt());
        }
        m_dW->FillZero();
        m_db->FillZero();

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
        BB_ASSERT(CalcShapeSize(shape) == CalcShapeSize(m_output_shape));
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
        if ( !this->m_parameter_lock ) {
            parameters.PushBack(m_W);
            parameters.PushBack(m_b);
        }
        return parameters;
    }

    Variables GetGradients(void)
    {
        Variables gradients;
        if ( !this->m_parameter_lock ) {
            gradients.PushBack(m_dW);
            gradients.PushBack(m_db);
        }
        return gradients;
    }


    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        // 量子化
        if ( m_quantize && m_input_bits > 0 ) {
            x_buf.Quantize_inplace(m_input_bits, m_input_scale);
        }

        if ( m_quantize && m_weight_bits > 0 ) {
            m_WQ = m_W->Quantize(m_weight_bits, m_weight_scale);
            m_bQ = m_b->Quantize(m_weight_bits, m_weight_scale);
        }
        else {
            m_WQ = *m_W;
            m_bQ = *m_b;
        }

        // backwardの為に保存
        if ( train ) {
            this->PushFrameBuffer(x_buf);
        }

        // 型合わせ
        if ( x_buf.GetType() != DataType<T>::type ) {
             x_buf = x_buf.ConvertTo(DataType<T>::type);
        }

        BB_ASSERT(x_buf.GetType() == DataType<T>::type);
        BB_ASSERT(x_buf.GetNodeSize() == m_input_node_size);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetNodeSize() != m_input_node_size) {
            SetInputShape(x_buf.GetShape());
        }

        // 出力を設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), m_output_shape, DataType<T>::type);

#ifdef BB_WITH_CUDA
        if (DataType<T>::type == BB_TYPE_FP32 && m_cublasEnable && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable())
        {
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);
            auto W_ptr = m_WQ.LockDeviceMemoryConst();
            auto b_ptr = m_bQ.LockDeviceMemoryConst();
            
            bbcu_fp32_MatrixRowwiseSetVector
                (
                    (float const *)b_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int          )y_buf.GetNodeSize(),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float))
                );

            int x_frame_stride = (int)(x_buf.GetFrameStride() / sizeof(float));
            int y_frame_stride = (int)(y_buf.GetFrameStride() / sizeof(float));
            float alpha = 1.0f;
            float beta  = 1.0f;
            for (index_t depth = 0; depth < m_depth_size; ++depth) {
                BB_CUBLAS_SAFE_CALL(cublasSgemm
                    (
                        m_cublasHandle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        (int)y_buf.GetFrameSize(),
                        (int)m_output_point_size, // y_buf.GetNodeSize(),
                        (int)m_input_point_size,  // x_buf.GetNodeSize(),
                        &alpha,
                        (const float *)x_ptr.GetAddr() + (depth * m_input_point_size * x_frame_stride),
                        (int)x_frame_stride,
                        (const float *)W_ptr.GetAddr() + (depth * m_input_point_size * m_output_point_size),
                        (int)m_input_point_size, // x_buf.GetNodeSize(),
                        &beta,
                        (float *)y_ptr.GetAddr() + (depth * m_output_point_size * y_frame_stride),
                        (int)y_frame_stride
                    ));
            }

            return y_buf;
        }
#endif

        {
            auto frame_size   = x_buf.GetFrameSize();

            auto x_ptr = x_buf.LockConst<T>();
            auto y_ptr = y_buf.Lock<T>();
            auto W_ptr = m_WQ.LockConst<T>();
            auto b_ptr = m_bQ.LockConst<T>();

            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t depth = 0; depth < m_depth_size; ++depth) {
                    for (index_t output_point = 0; output_point < m_output_point_size; ++output_point) {
                        index_t output_node = m_output_point_size * depth + output_point;
                        y_ptr.Set(frame, output_node, b_ptr(depth, output_point));
                        for (index_t input_point = 0; input_point < m_input_point_size; ++input_point) {
                            y_ptr.Add(frame, output_node, x_ptr.Get(frame, depth * m_input_point_size + input_point) * W_ptr(depth, output_point, input_point));
                        }
                    }
                }
            }

            return y_buf;
        }
    }


    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        if (dy_buf.Empty()) {
            m_dW = 0;
            m_db = 0;
            return dy_buf;
        }


        BB_ASSERT(dy_buf.GetType() == DataType<T>::type);

        // フレーム数
        auto frame_size = dy_buf.GetFrameSize();

        FrameBuffer x_buf = this->PopFrameBuffer();

        // 型合わせ
        if ( x_buf.GetType() != DataType<T>::type ) {
             x_buf = x_buf.ConvertTo(DataType<T>::type);
        }

        FrameBuffer dx_buf(dy_buf.GetFrameSize(), x_buf.GetShape(), DataType<T>::type);
        
        
#ifdef BB_WITH_CUDA
        if (DataType<T>::type == BB_TYPE_FP32 && m_cublasEnable && dy_buf.IsDeviceAvailable() && x_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable())
        {
            auto dy_ptr = dy_buf.LockDeviceMemoryConst();
            auto x_ptr  = x_buf.LockDeviceMemoryConst();
            auto dx_ptr = dx_buf.LockDeviceMemory(true);
            auto W_ptr  = m_WQ.LockDeviceMemoryConst();
            auto b_ptr  = m_bQ.LockDeviceMemoryConst();
            auto dW_ptr = m_dW->LockDeviceMemory();
            auto db_ptr = m_db->LockDeviceMemory();
            
            bbcu_fp32_MatrixColwiseSum
                (
                    (float const *)dy_ptr.GetAddr(),
                    (float       *)db_ptr.GetAddr(),
                    (int          )dy_buf.GetNodeSize(),
                    (int          )dy_buf.GetFrameSize(),
                    (int          )(dy_buf.GetFrameStride() / sizeof(float))
                );

            int dx_frame_stride = (int)(dx_buf.GetFrameStride() / sizeof(float));
            int dy_frame_stride = (int)(dy_buf.GetFrameStride() / sizeof(float));
            for (index_t depth = 0; depth < m_depth_size; ++depth) {
                float alpha = 1.0f;
                float beta = 0.0f;

                BB_CUBLAS_SAFE_CALL(cublasSgemm
                    (
                        m_cublasHandle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_T,
                        (int)dx_buf.GetFrameSize(),
                        (int)m_input_point_size, // dx_buf.GetNodeSize(),
                        (int)m_output_point_size, // dy_buf.GetNodeSize(),
                        &alpha,
                        (const float *)dy_ptr.GetAddr() + (depth * m_output_point_size * dy_frame_stride),
                        (int)dy_frame_stride,
                        (const float *)W_ptr.GetAddr() + (depth * m_output_point_size * m_input_point_size),
                        (int)m_input_point_size, // dx_buf.GetNodeSize(),
                        &beta,
                        (float *)dx_ptr.GetAddr() + (depth * m_input_point_size * dx_frame_stride),
                        (int)dx_frame_stride
                    ));
                
                beta = 1.0f;
                BB_CUBLAS_SAFE_CALL(cublasSgemm
                    (
                        m_cublasHandle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        (int)m_input_point_size, // dx_buf.GetNodeSize(),
                        (int)m_output_point_size, // dy_buf.GetNodeSize(),
                        (int)dx_buf.GetFrameSize(),
                        &alpha,
                        (const float *)x_ptr.GetAddr() + (depth * m_input_point_size * dx_frame_stride),
                        (int)dx_frame_stride,
                        (const float *)dy_ptr.GetAddr() + (depth * m_output_point_size * dy_frame_stride),
                        (int)dy_frame_stride,
                        &beta,
                        (float *)dW_ptr.GetAddr() + (depth * m_output_point_size * m_input_point_size),
                        (int)m_input_point_size // dx_buf.GetNodeSize()
                    ));
            }
            
            return dx_buf;
        }
#endif

        {
            dx_buf.FillZero();

            auto x_ptr  = x_buf.LockConst<T>();
            auto dy_ptr = dy_buf.LockConst<T>();
            auto dx_ptr = dx_buf.Lock<T>();
            auto W_ptr  = m_WQ.LockConst<T>();
            auto b_ptr  = m_bQ.LockConst<T>();
            auto dW_ptr = lock_dW();
            auto db_ptr = lock_db();

            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t depth = 0; depth < m_depth_size; ++depth) {
                    for (index_t output_point = 0; output_point < m_output_point_size; ++output_point) {
                        auto output_node = depth * m_output_point_size + output_point;
                        auto grad = dy_ptr.Get(frame, output_node);
                        db_ptr(depth, output_point) += grad;
                        for (index_t input_point = 0; input_point < m_input_point_size; ++input_point) {
                            dx_ptr.Add(frame, depth * m_input_point_size + input_point, grad * W_ptr(depth, output_point, input_point));
                            dW_ptr(depth, output_point, input_point) += grad * x_ptr.Get(frame, depth * m_input_point_size + input_point);
                        }
                    }
                }
            }

            return dx_buf;
        }
    }
    
        // シリアライズ
protected:
    void DumpObjectData(std::ostream &os) const override
    {
        // バージョン
        std::int64_t ver = 1;
        bb::SaveValue(os, ver);

        // 親クラス
        _super::DumpObjectData(os);

        // メンバ
        bb::SaveValue(os, m_host_only);
        bb::SaveValue(os, m_binary_mode);
        bb::SaveValue(os, m_initialize_std);
        bb::SaveValue(os, m_initializer);
        bb::SaveValue(os, m_input_shape);
        bb::SaveValue(os, m_output_shape);
        bb::SaveValue(os, m_input_point_size);
        bb::SaveValue(os, m_depth_size);
        bb::SaveValue(os, m_cublasEnable);
        bb::SaveValue(os, m_quantize);
        bb::SaveValue(os, m_weight_bits);
        bb::SaveValue(os, m_output_bits);
        bb::SaveValue(os, m_input_bits);
        bb::SaveValue(os, m_weight_scale);
        bb::SaveValue(os, m_output_scale);
        bb::SaveValue(os, m_input_scale);
        m_W->DumpObject(os);
        m_b->DumpObject(os);
    }

    void LoadObjectData(std::istream &is) override
    {
#ifdef BB_WITH_CUDA
        if ( m_cublasEnable ) {
            BB_CUBLAS_SAFE_CALL(cublasDestroy(m_cublasHandle));
            m_cublasEnable = false;
        }
#endif

        // バージョン
        std::int64_t ver;
        bb::LoadValue(is, ver);

        BB_ASSERT(ver == 1);

        // 親クラス
        _super::LoadObjectData(is);

        // メンバ
        bb::LoadValue(is, m_host_only);
        bb::LoadValue(is, m_binary_mode);
        bb::LoadValue(is, m_initialize_std);
        bb::LoadValue(is, m_initializer);
        bb::LoadValue(is, m_input_shape);
        bb::LoadValue(is, m_output_shape);
        bb::LoadValue(is, m_input_point_size);
        bb::LoadValue(is, m_depth_size);
        bb::LoadValue(is, m_cublasEnable);
        bb::LoadValue(is, m_quantize);
        bb::LoadValue(is, m_weight_bits);
        bb::LoadValue(is, m_output_bits);
        bb::LoadValue(is, m_input_bits);
        bb::LoadValue(is, m_weight_scale);
        bb::LoadValue(is, m_output_scale);
        bb::LoadValue(is, m_input_scale);
        m_W->LoadObject(is);
        m_b->LoadObject(is);
        
        // 再構築
#ifdef BB_WITH_CUDA
        if ( m_cublasEnable ) {
           if ( cublasCreate(&m_cublasHandle) != CUBLAS_STATUS_SUCCESS ) {
                m_cublasEnable = false;
            }
        }
#endif

        m_input_node_size = CalcShapeSize(m_input_shape);
        m_output_node_size = CalcShapeSize(m_output_shape);
        if ( !m_input_shape.empty() ) {
            if ( m_depth_size <= 0 ) {
                if ( m_input_point_size > 0 ) {
                    m_depth_size = m_input_node_size / m_input_point_size;
                }
                else
                {
                    m_depth_size = m_output_shape[0];
                }
            }

            m_input_point_size  = m_input_node_size / m_depth_size;
            m_output_point_size = m_output_node_size / m_depth_size;

            m_dW->Resize({m_depth_size, m_output_point_size, m_input_point_size}, DataType<T>::type);   m_dW->FillZero();
            m_db->Resize({m_depth_size, m_output_point_size},                     DataType<T>::type);   m_db->FillZero();
        }
    }


public:
    // Serialize(旧)
    void Save(std::ostream &os) const 
    {
        SaveValue(os, m_binary_mode);
        SaveIndices(os, m_input_shape);
        SaveIndices(os, m_output_shape);
        m_W->Save(os);
        m_b->Save(os);
    }

    void Load(std::istream &is)
    {
        bb::LoadValue(is, m_binary_mode);
        m_input_shape  = bb::LoadIndices(is);
        m_output_shape = bb::LoadIndices(is);
        m_W->Load(is);
        m_b->Load(is);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
        archive(cereal::make_nvp("binary_mode",      m_binary_mode));
        archive(cereal::make_nvp("input_shape",      m_input_shape));
        archive(cereal::make_nvp("output_shape",     m_output_shape));
        archive(cereal::make_nvp("cublasEnable",     m_cublasEnable));
        archive(cereal::make_nvp("quantize",         m_quantize));
        archive(cereal::make_nvp("weight_bits",      m_weight_bits));
        archive(cereal::make_nvp("output_bits",      m_output_bits));
        archive(cereal::make_nvp("input_bits",       m_input_bits));
        archive(cereal::make_nvp("weight_scale",     m_weight_scale));
        archive(cereal::make_nvp("output_scale",     m_output_scale));
        archive(cereal::make_nvp("input_scale",      m_input_scale));
        archive(cereal::make_nvp("W",                *m_W));
        archive(cereal::make_nvp("b",                *m_b));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("binary_mode",      m_binary_mode));
        archive(cereal::make_nvp("input_shape",      m_input_shape));
        archive(cereal::make_nvp("output_shape",     m_output_shape));
        archive(cereal::make_nvp("cublasEnable",     m_cublasEnable));
        archive(cereal::make_nvp("quantize",         m_quantize));
        archive(cereal::make_nvp("weight_bits",      m_weight_bits));
        archive(cereal::make_nvp("output_bits",      m_output_bits));
        archive(cereal::make_nvp("input_bits",       m_input_bits));
        archive(cereal::make_nvp("weight_scale",     m_weight_scale));
        archive(cereal::make_nvp("output_scale",     m_output_scale));
        archive(cereal::make_nvp("input_scale",      m_input_scale));

        m_input_node_size  = CalcShapeSize(m_input_shape);
        m_output_node_size = CalcShapeSize(m_output_shape);

        archive(cereal::make_nvp("W",                *m_W));
        archive(cereal::make_nvp("b",                *m_b));
    }

    void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("DepthwiseDenseAffineQuantize", *this));
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("DepthwiseDenseAffineQuantize", *this));
    }
#endif
};

}