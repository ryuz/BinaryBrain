﻿// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
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


// Affineレイヤー
template <typename T = float>
class DenseAffine : public Model
{
    using _super = Model;

public:
    static inline std::string ModelName(void) { return "DenseAffine"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<T>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }


protected:
    bool                        m_host_only = false;
    bool                        m_binary_mode = false;
    bool                        m_backward_break = false;

    T                           m_initialize_std = (T)0.01;
    std::string                 m_initializer = "";
    std::mt19937_64             m_mt;

    index_t                     m_input_node_size = 0;
    indices_t                   m_input_shape;
    index_t                     m_output_node_size = 0;
    indices_t                   m_output_shape;
    
    std::shared_ptr<Tensor>     m_W;
    std::shared_ptr<Tensor>     m_b;
    std::shared_ptr<Tensor>     m_dW;
    std::shared_ptr<Tensor>     m_db;
    
    bool                        m_cublasEnable = false;
#ifdef BB_WITH_CUDA
    cublasHandle_t              m_cublasHandle;
#endif


public:
    struct create_t
    {
        indices_t       output_shape;
        T               initialize_std = (T)0.01;
        std::string     initializer = "";
        std::uint64_t   seed = 1;
    };

protected:
    DenseAffine(create_t const &create)
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

        m_initialize_std  = create.initialize_std;
        m_initializer     = create.initializer;
        m_mt.seed(create.seed);

        m_output_shape     = create.output_shape;
        m_output_node_size = CalcShapeSize(m_output_shape);
    }

    void CommandProc(std::vector<std::string> args)
    {
        _super::CommandProc(args);

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

        if (args.size() == 2 && args[0] == "backward_break")
        {
            m_backward_break = EvalBool(args[1]);
        }
    }


public:
    ~DenseAffine() {
#ifdef BB_WITH_CUDA
        if ( m_cublasEnable ) {
            BB_CUBLAS_SAFE_CALL(cublasDestroy(m_cublasHandle));
            m_cublasEnable = false;
        }
#endif
    }

    static std::shared_ptr<DenseAffine> Create(create_t const &create)
    {
        return std::shared_ptr<DenseAffine>(new DenseAffine(create));
    }

    static std::shared_ptr<DenseAffine> Create(indices_t const &output_shape)
    {
        create_t create;
        create.output_shape = output_shape;
        return Create(create);
    }

    static std::shared_ptr<DenseAffine> Create(index_t output_node_size)
    {
        create_t create;
        create.output_shape.resize(1);
        create.output_shape[0] = output_node_size;
        return Create(create);
    }

    static std::shared_ptr<DenseAffine> Create(void)
    {
        return Create(create_t());
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<DenseAffine> CreatePy(
            indices_t       output_shape,
            T               initialize_std = (T)0.01,
            std::string     initializer = "",
            std::uint64_t   seed = 1
        )
    {
        create_t create;
        create.output_shape   = output_shape;
        create.initialize_std = initialize_std;
        create.initializer    = initializer;
        create.seed           = seed;
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


   /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

        // 形状設定
        m_input_shape = shape;
        m_input_node_size = CalcShapeSize(shape);

        // パラメータ初期化
        m_W->Resize ({m_output_node_size, m_input_node_size}, DataType<T>::type);
        m_b->Resize ({m_output_node_size},                    DataType<T>::type);
        m_dW->Resize({m_output_node_size, m_input_node_size}, DataType<T>::type);
        m_db->Resize({m_output_node_size},                    DataType<T>::type);

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
            double k = (double)m_initialize_std * std::sqrt(3.0);
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
        BB_ASSERT(CalcShapeSize(shape) == m_input_node_size);
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
    

    Variables GetParameters(void) override
    {
        Variables parameters;
        if ( !this->m_parameter_lock ) {
            parameters.PushBack(m_W);
            parameters.PushBack(m_b);
        }
        return parameters;
    }

    Variables GetGradients(void) override
    {
        Variables gradients;
        if ( !this->m_parameter_lock ) {
            gradients.PushBack(m_dW);
            gradients.PushBack(m_db);
        }
        return gradients;
    }

    FrameBuffer Forward(FrameBuffer x_buf, bool train = true) override
    {
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
            auto W_ptr = m_W->LockDeviceMemoryConst();
            auto b_ptr = m_b->LockDeviceMemoryConst();
            
            bbcu_fp32_MatrixRowwiseSetVector
                (
                    (float const *)b_ptr.GetAddr(),
                    (float       *)y_ptr.GetAddr(),
                    (int          )y_buf.GetNodeSize(),
                    (int          )y_buf.GetFrameSize(),
                    (int          )(y_buf.GetFrameStride() / sizeof(float))
                );

            float alpha = 1.0f;
            float beta = 1.0f;
            BB_CUBLAS_SAFE_CALL(cublasSgemm
                (
                    m_cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    (int)y_buf.GetFrameSize(),
                    (int)y_buf.GetNodeSize(),
                    (int)x_buf.GetNodeSize(),
                    &alpha,
                    (const float *)x_ptr.GetAddr(),
                    (int)(x_buf.GetFrameStride() / sizeof(float)),
                    (const float *)W_ptr.GetAddr(),
                    (int)x_buf.GetNodeSize(),
                    &beta,
                    (float *)y_ptr.GetAddr(),
                    (int)(y_buf.GetFrameStride() / sizeof(float))
                ));
            
            return y_buf;
        }
#endif

        {
            auto frame_size   = x_buf.GetFrameSize();

            auto x_ptr = x_buf.LockConst<T>();
            auto y_ptr = y_buf.Lock<T>();
            auto W_ptr = lock_W_const();
            auto b_ptr = lock_b_const();

            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t output_node = 0; output_node < m_output_node_size; ++output_node) {
                    y_ptr.Set(frame, output_node, b_ptr(output_node));
                    for (index_t input_node = 0; input_node < m_input_node_size; ++input_node) {
                        y_ptr.Add(frame, output_node, x_ptr.Get(frame, input_node) * W_ptr(output_node, input_node));
                    }
                }
            }

            return y_buf;
        }
    }


    FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        if ( dy_buf.Empty() || m_backward_break ) {
            m_dW = 0;
            m_db = 0;
            return FrameBuffer();
        }

        BB_ASSERT(dy_buf.GetType() == DataType<T>::type);

        // フレーム数
        auto frame_size = dy_buf.GetFrameSize();

        // forward時保存復帰
        FrameBuffer x_buf = PopFrameBuffer();
        BB_ASSERT(x_buf.GetFrameSize() == dy_buf.GetFrameSize());

        // 型合わせ
        if ( x_buf.GetType() != DataType<T>::type ) {
             x_buf = x_buf.ConvertTo(DataType<T>::type);
        }
        
        FrameBuffer dx_buf(x_buf.GetFrameSize(), x_buf.GetShape(), DataType<T>::type);


        #ifdef BB_WITH_CUDA
        if (DataType<T>::type == BB_TYPE_FP32 && m_cublasEnable && dy_buf.IsDeviceAvailable() && x_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable())
        {
            auto dy_ptr = dy_buf.LockDeviceMemoryConst();
            auto x_ptr  = x_buf.LockDeviceMemoryConst();
            auto dx_ptr = dx_buf.LockDeviceMemory(true);
            auto W_ptr  = m_W->LockDeviceMemoryConst();
            auto b_ptr  = m_b->LockDeviceMemoryConst();
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

            float alpha = 1.0f;
            float beta = 0.0f;
            BB_CUBLAS_SAFE_CALL(cublasSgemm
                (
                    m_cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    (int)dx_buf.GetFrameSize(),
                    (int)dx_buf.GetNodeSize(),
                    (int)dy_buf.GetNodeSize(),
                    &alpha,
                    (const float *)dy_ptr.GetAddr(),
                    (int)(dy_buf.GetFrameStride() / sizeof(float)),
                    (const float *)W_ptr.GetAddr(),
                    (int)dx_buf.GetNodeSize(),
                    &beta,
                    (float *)dx_ptr.GetAddr(),
                    (int)(dx_buf.GetFrameStride() / sizeof(float))
                ));
            
            beta = 1.0f;
            BB_CUBLAS_SAFE_CALL(cublasSgemm
                (
                    m_cublasHandle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    (int)dx_buf.GetNodeSize(),
                    (int)dy_buf.GetNodeSize(),
                    (int)dx_buf.GetFrameSize(),
                    &alpha,
                    (const float *)x_ptr.GetAddr(),
                    (int)(x_buf.GetFrameStride() / sizeof(float)),
                    (const float *)dy_ptr.GetAddr(),
                    (int)(dy_buf.GetFrameStride() / sizeof(float)),
                    &beta,
                    (float *)dW_ptr.GetAddr(),
                    (int)dx_buf.GetNodeSize()
                ));
            
            return dx_buf;
        }
#endif

        {
            dx_buf.FillZero();

            auto x_ptr  = x_buf.LockConst<T>();
            auto dy_ptr = dy_buf.LockConst<T>();
            auto dx_ptr = dx_buf.Lock<T>();
            auto W_ptr  = lock_W_const();
            auto b_ptr  = lock_b_const();
            auto dW_ptr = lock_dW();
            auto db_ptr = lock_db();

            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t output_node = 0; output_node < m_output_node_size; ++output_node) {
                    auto grad = dy_ptr.Get(frame, output_node);
                    db_ptr(output_node) += grad;
                    for (index_t input_node = 0; input_node < m_input_node_size; ++input_node) {
                        dx_ptr.Add(frame, input_node, grad * W_ptr(output_node, input_node));
                        dW_ptr(output_node, input_node) += grad * x_ptr.Get(frame, input_node);
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
        bb::SaveValue(os, m_cublasEnable);
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
        bb::LoadValue(is, m_cublasEnable);
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
        m_dW->Resize({m_output_node_size, m_input_node_size}, DataType<T>::type);   m_dW->FillZero();
        m_db->Resize({m_output_node_size},                    DataType<T>::type);   m_db->FillZero();
    }


public:
    // Serialize(旧)
    void Save(std::ostream &os) const 
    {
//      SaveValue(os, m_binary_mode);
        os.write((const char*)&m_binary_mode, sizeof(m_binary_mode));   // バグに対する後方互換性

        SaveIndices(os, m_input_shape);
        SaveIndices(os, m_output_shape);
        m_W->Save(os);
        m_b->Save(os);
    }

    void Load(std::istream &is)
    {
//      bb::LoadValue(is, m_binary_mode);
        is.read((char*)&m_binary_mode, sizeof(m_binary_mode));   // バグに対する後方互換性

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

        m_input_node_size  = CalcShapeSize(m_input_shape);
        m_output_node_size = CalcShapeSize(m_output_shape);

        archive(cereal::make_nvp("W",                *m_W));
        archive(cereal::make_nvp("b",                *m_b));
    }

    void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("DenseAffine", *this));
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("DenseAffine", *this));
    }
#endif
};

}