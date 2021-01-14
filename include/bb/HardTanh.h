// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Manager.h"
#include "bb/Binarize.h"


namespace bb {


// Hard-Tanh
template <typename BinType = float, typename RealType = float>
class HardTanh : public Binarize<BinType, RealType>
{
    using _super = Binarize<BinType, RealType>;

public:
    static inline std::string ModelName(void) { return "HardTanh"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:

    using _super::m_host_only;
    using _super::m_binary_th;
    using _super::m_hardtanh_min;
    using _super::m_hardtanh_max;
    using _super::m_x_buf;

    bool        m_binary_mode = false;

public:
    // 生成情報
    struct create_t
    {
        RealType   hardtanh_min = (RealType)-1.0;
        RealType   hardtanh_max = (RealType)+1.0;
    };

protected:
    HardTanh(create_t const &create)
    {
        m_hardtanh_min = create.hardtanh_min;
        m_hardtanh_max = create.hardtanh_max;
        m_binary_th    = (m_hardtanh_min + m_hardtanh_max) / (RealType)2;
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args) override
    {
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


public:
    ~HardTanh() {}

    static std::shared_ptr<HardTanh> Create(create_t const &create)
    {
        return std::shared_ptr<HardTanh>(new HardTanh(create));
    }

    static std::shared_ptr<HardTanh> Create(RealType hardtanh_min = (RealType)-1, RealType hardtanh_max = (RealType)+1)
    {
        create_t create;
        create.hardtanh_min = hardtanh_min;
        create.hardtanh_max = hardtanh_max;
        return Create(create);
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<HardTanh> CreatePy(double hardtanh_min = -1.0, double hardtanh_max = +1.0)
    {
        create_t create;
        create.hardtanh_min = (RealType)hardtanh_min;
        create.hardtanh_max = (RealType)hardtanh_max;
        return Create(create);
    }
#endif

    void        SetFrameBufferX(FrameBuffer x) { m_x_buf = x; }
    FrameBuffer GetFrameBufferX(void)          { return m_x_buf; }

    // 1ノードのみForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const override
    {
        if ( m_binary_mode ) {
            return _super::ForwardNode(node, x_vec);
        }

        for ( auto& x : x_vec ) {
            if ( x <= m_hardtanh_min ) { x = (double)m_hardtanh_min; }
            if ( x >= m_hardtanh_max ) { x = (double)m_hardtanh_max; }
        }
        return x_vec;
    }
    
    /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    inline FrameBuffer Forward(FrameBuffer x_buf, bool train = true) override
    {
        // binaryモード
        if ( DataType<BinType>::type == BB_TYPE_BIT || m_binary_mode ) {
            return _super::Forward(x_buf, train);
        }

        BB_ASSERT(x_buf.GetType() == DataType<RealType>::type);

        // backward用に保存
        if ( train ) {
            m_x_buf = x_buf;
        }

        // 戻り値の設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), x_buf.GetShape(), x_buf.GetType());

#ifdef BB_WITH_CUDA
        if ( DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory();
            bbcu_fp32_HardTanh_Forward(
                        (float const *)ptr_x.GetAddr(),
                        (float       *)ptr_y.GetAddr(),
                        (float        )m_hardtanh_min,
                        (float        )m_hardtanh_max,
                        (int          )y_buf.GetNodeSize(),
                        (int          )y_buf.GetFrameSize(),
                        (int          )(y_buf.GetFrameStride() / sizeof(float))
                    );
            
            return y_buf;
        }
#endif

        {
            // 汎用版
            index_t frame_size = x_buf.GetFrameSize();
            index_t node_size  = x_buf.GetNodeSize();

            auto x_ptr = x_buf.template LockConst<RealType>();
            auto y_ptr = y_buf.template Lock<BinType>();

            // Hard-Tanh
    #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto x = x_ptr.Get(frame, node);
                    if ( x <= m_hardtanh_min ) { x = m_hardtanh_min; }
                    if ( x >= m_hardtanh_max ) { x = m_hardtanh_max; }
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
    inline FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        // binaryモード
        if ( DataType<BinType>::type == BB_TYPE_BIT || m_binary_mode) {
            return _super::Backward(dy_buf);
        }

        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        // 戻り値のサイズ設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), dy_buf.GetShape(), dy_buf.GetType());

        auto x_buf = m_x_buf;
        m_x_buf = FrameBuffer();

#ifdef BB_WITH_CUDA
        if ( DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // GPU版
            auto ptr_x  = x_buf.LockDeviceMemoryConst();
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = dx_buf.LockDeviceMemory(true);
            bbcu_fp32_HardTanh_Backward(
                        (float const *)ptr_x.GetAddr(),
                        (float const *)ptr_dy.GetAddr(),
                        (float       *)ptr_dx.GetAddr(),
                        (float        )m_hardtanh_min,
                        (float        )m_hardtanh_max,
                        (int          )dx_buf.GetNodeSize(),
                        (int          )dx_buf.GetFrameSize(),
                        (int          )(dx_buf.GetFrameStride() / sizeof(float))
                    );
            return dx_buf;
        }
#endif

        {
            // 汎用版
            index_t frame_size = dx_buf.GetFrameSize();
            index_t node_size  = dx_buf.GetNodeSize();

            auto x_ptr  = x_buf.template LockConst<RealType>();
            auto dy_ptr = dy_buf.template LockConst<RealType>();
            auto dx_ptr = dx_buf.template Lock<RealType>();

            // Hard-Tanh
            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto x  = x_ptr.Get(frame, node);
                    auto dy = dy_ptr.Get(frame, node);
                    if ( x <= m_hardtanh_min ) { dy = (RealType)0; }
                    if ( x >= m_hardtanh_max ) { dy = (RealType)0; }
                    dx_ptr.Set(frame, node, dy);
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
        bb::SaveValue(os, m_binary_mode);

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
        bb::LoadValue(is, m_binary_mode);
    }
};


}


// end of file