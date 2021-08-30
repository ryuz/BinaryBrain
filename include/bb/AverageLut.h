// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <array>
#include <vector>
#include "bb/BinaryLutModel.h"


namespace bb {


// LUT版popcount
template <typename BinType = float, typename RealType = float>
class AverageLut : public BinaryLutModel
{
    using _super = BinaryLutModel;

public:
    static inline std::string ClassName(void) { return "AverageLut"; }
    static inline std::string ObjectName(void){ return ClassName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const override { return ClassName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    bool                    m_host_only       = false;
    bool                    m_binarize_input  = false;
    bool                    m_binarize_output = true;

    std::string             m_connection;

    int                     m_n = 6;
    indices_t               m_input_shape;
    indices_t               m_output_shape;

    Tensor_<std::int32_t>   m_input_index;

    std::mt19937_64         m_mt;

public:
    struct create_t
    {
        int             n               = 6;
        indices_t       output_shape;
        std::string     connection      = "";
        bool            binarize_input  = false;
        bool            binarize_output = true;
        std::uint64_t   seed            = 1;
    };

protected:
    AverageLut(create_t const &create)
    {
        BB_ASSERT(!create.output_shape.empty());
        m_mt.seed(create.seed);
        m_n            = create.n;
        m_output_shape = create.output_shape;
        m_connection   = create.connection;
        m_input_index.Resize(CalcShapeSize(m_output_shape), (index_t)m_n);
    }

    void CommandProc(std::vector<std::string> args) override
    {
        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }

        // バイナリモード設定
        if ( args.size() == 2 && args[0] == "binary" )
        {
            m_binarize_input  = EvalBool(args[1]);
            m_binarize_output = EvalBool(args[1]);
        }

        if ( args.size() == 2 && args[0] == "binarize_input" )
        {
            m_binarize_input = EvalBool(args[1]);
        }

        if ( args.size() == 2 && args[0] == "binarize_output" )
        {
            m_binarize_output = EvalBool(args[1]);
        }
    }

public:
    ~AverageLut() {}

    static std::shared_ptr<AverageLut> Create(create_t const &create)
    {
        return std::shared_ptr<AverageLut>(new AverageLut(create));
    }

    static std::shared_ptr<AverageLut> Create(int n, indices_t const &output_shape, std::string connection = "", bool binarize = true, bool binarize_input = false, std::uint64_t seed = 1)
    {
        create_t create;
        create.n               = n;
        create.output_shape    = output_shape;
        create.connection      = connection;
        create.binarize_input  = binarize_input;
        create.binarize_output = binarize;
        create.seed            = seed;
        return Create(create);
    }

    static std::shared_ptr<AverageLut> Create(int n, index_t output_node_size, std::string connection = "", bool binarize = true, bool binarize_input = false, std::uint64_t seed = 1)
    {
        create_t create;
        create.n               = n;
        create.output_shape.resize(1);
        create.output_shape[0] = output_node_size;
        create.connection      = connection;
        create.binarize_input  = binarize_input;
        create.binarize_output = binarize;
        create.seed            = seed;
        return Create(create);
    }

    static std::shared_ptr<AverageLut> Create(void)
    {
        return Create(create_t());
    }


#ifdef BB_PYBIND11    // python用
    static std::shared_ptr<AverageLut> CreatePy(
                int             n,
                indices_t       output_shape,
                std::string     connection="",
                bool            binarize = true,
                bool            binarize_input = false,
                std::uint64_t   seed = 1)
    {
        create_t create;
        create.n               = n;
        create.output_shape    = output_shape;
        create.connection      = connection;
        create.binarize_input  = binarize_input;
        create.binarize_output = binarize;
        create.seed            = seed;
        return Create(create);
    }
#endif

    auto lock_InputIndex(void)             { return m_input_index.Lock(); }
    auto lock_InputIndex_const(void) const { return m_input_index.LockConst(); }

    // 疎結合の管理
    index_t GetNodeConnectionSize(index_t node) const override
    {
        return m_n;
    }

    void SetNodeConnectionIndex(index_t node, index_t input_index, index_t input_node) override
    {
        BB_ASSERT(node >= 0 && node < CalcShapeSize(m_output_shape));
        BB_ASSERT(input_index >= 0 && input_index < m_n);
        BB_DEBUG_ASSERT(input_node >= 0 && input_node < GetInputNodeSize());

        auto ptr = lock_InputIndex();
        ptr(node, input_index) = (std::int32_t)input_node;
    }

    index_t GetNodeConnectionIndex(index_t node, index_t input_index) const override
    {
        BB_ASSERT(node >= 0 && node < CalcShapeSize(m_output_shape));
        BB_ASSERT(input_index >= 0 && input_index < m_n);
        
        auto ptr = lock_InputIndex_const();
        return (index_t)ptr(node, input_index);
    }
    
    // LUT操作の定義
    int GetLutTableSize(index_t node) const
    {
        return (1 << m_n);
    }

    void SetLutTable(index_t node, int bitpos, bool value) override
    {
    }

    bool GetLutTable(index_t node, int bitpos) const override
    {
        int count = 0;
        for ( int i = 0; i < m_n; ++i ) {
            count += (bitpos & 1) ? +1 : -1;
            bitpos >>= 1;
        }
        return count > 0;
    }


   /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape) override
    {
        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

        // 形状設定
        m_input_shape = shape;

        // 接続初期化
        this->InitializeNodeInput(m_mt(), m_connection);

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
        BB_ASSERT(CalcShapeSize(shape) == this->m_output_node_size);
        m_output_shape = shape;
    }


    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const override
    {
        return m_input_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const override
    {
        return m_output_shape;
    }
    

public:
    FrameBuffer Forward(FrameBuffer x_buf, bool train = true) override
    {
        BB_ASSERT(x_buf.GetType() == DataType<BinType>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_input_shape) {
            SetInputShape(x_buf.GetShape());
        }
        
        // 出力を設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), m_output_shape, DataType<BinType>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<BinType>::type == BB_TYPE_FP32 && !m_host_only
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto x_ptr           = x_buf.LockDeviceMemoryConst();
            auto y_ptr           = y_buf.LockDeviceMemory(true);
            auto input_index_ptr = m_input_index.LockDeviceMemoryConst();

            bbcu_AverageLut_Forward<float>
                (
                    (float  const   *)x_ptr.GetAddr(),
                    (float          *)y_ptr.GetAddr(),
                    (int    const   *)input_index_ptr.GetAddr(),
                    (int             )m_n,
                    (int             )y_buf.GetNodeSize(),
                    (int             )y_buf.GetFrameSize(),
                    (int             )(y_buf.GetFrameStride() / sizeof(float)),
                    (bool            )m_binarize_input,
                    (bool            )m_binarize_output
                );
            return y_buf;
        }

        
        if ( DataType<BinType>::type == BB_TYPE_BIT && !m_host_only
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto x_ptr           = x_buf.LockDeviceMemoryConst();
            auto y_ptr           = y_buf.LockDeviceMemory(true);
            auto input_index_ptr = m_input_index.LockDeviceMemoryConst();

            bbcu_bit_AverageLut_Forward
                (
                    (int    const   *)x_ptr.GetAddr(),
                    (int            *)y_ptr.GetAddr(),
                    (int    const   *)input_index_ptr.GetAddr(),
                    (int             )m_n,
                    (int             )y_buf.GetNodeSize(),
                    (int             )y_buf.GetFrameSize(),
                    (int             )(y_buf.GetFrameStride() / sizeof(int))
                );
            return y_buf;
        }
        
#endif
        
        {
            // 汎用版
            auto x_ptr           = x_buf.LockConst<BinType>();
            auto y_ptr           = y_buf.Lock<BinType>();
            auto input_index_ptr = m_input_index.LockConst();

            index_t frame_size = x_buf.GetFrameSize();
            index_t node_size  = this->GetOutputNodeSize();

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    RealType sum = 0;
                    for (index_t i = 0; i < m_n; i++) {
                        index_t input_node = input_index_ptr(node, i);
                        RealType val = (RealType)x_ptr.Get(frame, input_node);
                        if (m_binarize_input) {
                            val =  (val > 0) ? (RealType)BB_BINARY_HI : (RealType)BB_BINARY_LO;
                        }
                        sum += val;
                    }
                    if (m_binarize_output) {
                        sum = (sum > 0) ? (RealType)BB_BINARY_HI : (RealType)BB_BINARY_LO;
                    }
                    y_ptr.Set(frame, node, (BinType)sum);
                }
            }

            return y_buf;
        }
    }

    // Backward
    FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        if (dy_buf.Empty()) {
            return dy_buf;
        }

        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        // 出力を設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), m_input_shape, DataType<RealType>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
                && dy_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto dy_ptr          = dy_buf.LockDeviceMemoryConst();
            auto dx_ptr          = dx_buf.LockDeviceMemory(true);
            auto input_index_ptr = m_input_index.LockDeviceMemoryConst();

            bbcu_AverageLut_Backward<float>
                (
                    (float  const   *)dy_ptr.GetAddr(),
                    (float          *)dx_ptr.GetAddr(),
                    (int    const   *)input_index_ptr.GetAddr(),
                    (int             )m_n,
                    (int             )dx_buf.GetNodeSize(),
                    (int             )dy_buf.GetNodeSize(),
                    (int             )dy_buf.GetFrameSize(),
                    (int             )(dy_buf.GetFrameStride() / sizeof(float))
                );
            return dx_buf;
        }
#endif

        {
            // 汎用版
            dx_buf.FillZero();

            auto dy_ptr          = dy_buf.LockConst<RealType>();
            auto dx_ptr          = dx_buf.Lock<RealType>();
            auto input_index_ptr = m_input_index.LockConst();

            index_t frame_size   = dy_buf.GetFrameSize();
            index_t node_size    = this->GetOutputNodeSize();

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto dx = dy_ptr.Get(frame, node) / m_n;
                    for (index_t i = 0; i < m_n; i++) {
                        index_t input_node = input_index_ptr(node, i);
                        dx_ptr.Add(frame, input_node, dx);
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
        bb::SaveValue(os, m_n);
        bb::SaveValue(os, m_host_only);
        bb::SaveValue(os, m_connection);
        bb::SaveValue(os, m_input_shape);
        bb::SaveValue(os, m_output_shape);
        bb::SaveValue(os, m_binarize_input);
        bb::SaveValue(os, m_binarize_output);
        m_input_index.DumpObject(os);
    }

    void LoadObjectData(std::istream &is) override
    {
        // バージョン
        std::int64_t ver;
        bb::LoadValue(is, ver);

        BB_ASSERT(ver == 1);

        // 親クラス
        _super::LoadObjectData(is);

        // メンバ
        bb::LoadValue(is, m_n);
        bb::LoadValue(is, m_host_only);
        bb::LoadValue(is, m_connection);
        bb::LoadValue(is, m_input_shape);
        bb::LoadValue(is, m_output_shape);
        bb::LoadValue(is, m_binarize_input);
        bb::LoadValue(is, m_binarize_output);
        m_input_index.LoadObject(is);
    }

};


}