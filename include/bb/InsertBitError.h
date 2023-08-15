// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>
#include <stack>
#include "bb/Model.h"


namespace bb {


/**
 * @brief   バイナリデータにビットエラーをのせる
 */
template <typename BinType = float, typename RealType = float>
class InsertBitError : public Model
{
    using _super = Model;

public:
    static inline std::string ClassName(void) { return "InsertBitError"; }
    static inline std::string ObjectName(void){ return ClassName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const { return ClassName(); }
    std::string GetObjectName(void) const { return ObjectName(); }

protected:
    bool                        m_host_only = false;
    indices_t                   m_shape;
    double                      m_error_rate;
    std::mt19937                m_rand_engine;
    std::stack<std::uint32_t>   m_seed_stack;

public:
    struct create_t
    {
        double  error_rate;     // エラー率


        void ObjectDump(std::ostream& os) const
        {
            bb::SaveValue(os, error_rate);
        }

        void ObjectLoad(std::istream& is)
        {
            bb::LoadValue(is, error_rate);
        }
    };

protected:
    InsertBitError(create_t const &create)
    {
        m_error_rate = create.error_rate;
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args)
    {
        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }

        // error_rate設定
        if (args.size() == 2 && args[0] == "error_rate")
        {
            m_error_rate = (double)EvalReal(args[1]);
        }
    }

public:
    ~InsertBitError() {}


    static std::shared_ptr<InsertBitError> Create(create_t const &create)
    {
        return std::shared_ptr<InsertBitError>(new InsertBitError(create));
    }

    static std::shared_ptr<InsertBitError> Create(double error_rate)
    {
        create_t create;
        create.error_rate = error_rate;
        return Create(create);
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<InsertBitError> CreatePy(double error_rate)
    {
        create_t create;
        create.error_rate = error_rate;
        return Create(create);
    }
#endif



    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape) override
    {
        // 設定済みなら何もしない
        if (shape == this->GetInputShape()) {
            return this->GetOutputShape();
        }

        // 形状設定
        m_shape = shape;
        return m_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const override
    {
        return m_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_shape;
    }
    
private:
    int GenRand(int seed, index_t node, index_t frame, index_t node_size, index_t frame_size) {
        seed += frame_size * node + frame;
        return ((1103515245 * seed + 12345) & 0xffff);
    }

public:
    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        if (!train) {
            return x_buf;
        }

        BB_ASSERT(x_buf.GetType() == DataType<BinType>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_shape) {
            (void)SetInputShape(x_buf.GetShape());
        }

        index_t frame_size = x_buf.GetFrameSize();
        index_t node_size = x_buf.GetNodeSize();

        int seed = m_rand_engine();
        m_seed_stack.push(seed);


#ifdef BB_WITH_CUDA
        if (!m_host_only && DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32
            && x_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {

            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemory();
            bbcu_fp32_BitError_Forward(
                (float *)ptr_x.GetAddr(),
                (int)seed,
                (double)m_error_rate,
                (int)x_buf.GetNodeSize(),
                (int)x_buf.GetFrameSize(),
                (int)(x_buf.GetFrameStride() / sizeof(float))
            );
            return x_buf;
        }
        if (!m_host_only && DataType<BinType>::type == BB_TYPE_BIT && DataType<RealType>::type == BB_TYPE_FP32
            && x_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {

            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemory();
            bbcu_bit_BitError_Forward(
                (int*)ptr_x.GetAddr(),
                (int)seed,
                (double)m_error_rate,
                (int)x_buf.GetNodeSize(),
                (int)x_buf.GetFrameSize(),
                (int)(x_buf.GetFrameStride() / sizeof(float))
            );
            return x_buf;
        }
#endif

        {
            // 汎用版
            int error_th = (int)(m_error_rate * 0x10000);
            auto x_ptr   = x_buf.template Lock<BinType>();
#pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto x = x_ptr.Get(frame, node);
                    int rnd = GenRand(seed, node, frame, node_size, frame_size);
                    if ( rnd < error_th ) {
                        x_ptr.Set(frame, node, (BinType)1 - x);
                    }
                }
            }
            return x_buf;
        }
    }
    

    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        index_t frame_size = dy_buf.GetFrameSize();
        index_t node_size = dy_buf.GetNodeSize();

        auto seed = m_seed_stack.top();
        m_seed_stack.pop();

#ifdef BB_WITH_CUDA
        if (DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
            && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
            // GPU版
            auto ptr_dy = dy_buf.LockDeviceMemory();
            bbcu_fp32_BitError_Backward(
                (float *)ptr_dy.GetAddr(),
                (int)seed,
                (double)m_error_rate,
                +1.0f,
                -1.0f,
                (int)dy_buf.GetNodeSize(),
                (int)dy_buf.GetFrameSize(),
                (int)(dy_buf.GetFrameStride() / sizeof(float))
            );
            return dy_buf;
        }
#endif


        {
            // 汎用版
            int error_th = (int)(m_error_rate * 0x10000);
            auto dy_ptr = dy_buf.template Lock<RealType>();
#pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t node = 0; node < node_size; ++node) {
                    auto dy = dy_ptr.Get(frame, node);
                    int rnd = GenRand(seed, node, frame, node_size, frame_size);
                    if ( rnd < error_th ) {
                        dy_ptr.Set(frame, node, -dy);
                    }
                }
            }

            return dy_buf;
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
        bb::SaveValue(os, m_error_rate);
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
        bb::LoadValue(is, m_error_rate);
    }
};


}


// end of file
