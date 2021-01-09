// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Manager.h"
#include "bb/Activation.h"


namespace bb {


// Binarize(活性化層)
template <typename BinType = float, typename RealType = float>
class Binarize : public Activation
{
    using _super = Activation;

public:
    static inline std::string ClassName(void) { return "Binarize"; }
    static inline std::string ObjectName(void){ return ClassName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const { return ClassName(); }
    std::string GetObjectName(void) const { return ObjectName(); }

protected:
    bool        m_host_only = false;

    RealType    m_binary_th    = (RealType)0;
    RealType    m_hardtanh_min = (RealType)-1;
    RealType    m_hardtanh_max = (RealType)+1;

    FrameBuffer m_x_buf;


public:
    // 生成情報
    struct create_t
    {
        RealType    binary_th    = (RealType)0;
        RealType    hardtanh_min = (RealType)-1;
        RealType    hardtanh_max = (RealType)+1;
    };
    
protected:
    Binarize() {}

    Binarize(create_t const &create)
    {
        m_binary_th    = create.binary_th;
        m_hardtanh_min = create.hardtanh_min;
        m_hardtanh_max = create.hardtanh_max;
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args) override
    {
        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }
    }
    
public:
    ~Binarize() {}

    static std::shared_ptr<Binarize> Create(create_t const &create)
    {
        return std::shared_ptr<Binarize>(new Binarize(create));
    }

    static std::shared_ptr<Binarize> Create(RealType binary_th = (RealType)0, RealType hardtanh_min = (RealType)-1, RealType hardtanh_max = (RealType)+1)
    {
        create_t create;
        create.binary_th    = binary_th;
        create.hardtanh_min = hardtanh_min;
        create.hardtanh_max = hardtanh_max;
        return Create(create);
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<Binarize> CreatePy(RealType binary_th = (RealType)0, RealType hardtanh_min = (RealType)-1, RealType hardtanh_max = (RealType)+1)
    {
        create_t create;
        create.binary_th    = binary_th;
        create.hardtanh_min = hardtanh_min;
        create.hardtanh_max = hardtanh_max;
        return Create(create);
    }
#endif

    // ノード単位でのForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const override
    {
        std::vector<double> y_vec;
        for ( auto x : x_vec ) {
            y_vec.push_back((x > m_binary_th) ? m_hardtanh_max : m_hardtanh_min);
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
    inline FrameBuffer Forward(FrameBuffer x_buf, bool train = true) override
    {
        BB_ASSERT(x_buf.GetType() == DataType<RealType>::type);

        // backwardの為に保存
        if ( train ) {
            m_x_buf = x_buf;
        }

        // 戻り値のサイズ設定
        FrameBuffer y_buf( x_buf.GetFrameSize(), x_buf.GetShape(), DataType<BinType>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
            && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory(true);
            bbcu_fp32_Binarize_Forward(
                        (float const *)ptr_x.GetAddr(),
                        (float       *)ptr_y.GetAddr(),
                        (float        )m_binary_th,
                        (int          )y_buf.GetNodeSize(),
                        (int          )y_buf.GetFrameSize(),
                        (int          )(y_buf.GetFrameStride() / sizeof(float))
                    );
            return y_buf;
        }
#endif
        
#ifdef BB_WITH_CUDA
        if ( DataType<BinType>::type == BB_TYPE_BIT && DataType<RealType>::type == BB_TYPE_FP32 && !m_host_only
            && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory(true);
            bbcu_fp32_bit_Binarize_Forward
                    (
                        (float const *)ptr_x.GetAddr(),
                        (int         *)ptr_y.GetAddr(),
                        (float        )m_binary_th,
                        (int          )x_buf.GetNodeSize(),
                        (int          )x_buf.GetFrameSize(),
                        (int          )(x_buf.GetFrameStride() / sizeof(float)),
                        (int          )(y_buf.GetFrameStride() / sizeof(int))
                    );
            return y_buf;
        }
#endif

        {
            // 汎用版
            index_t frame_size = x_buf.GetFrameSize();
            index_t node_size = x_buf.GetNodeSize();

            auto x_ptr = x_buf.LockConst<RealType>();
            auto y_ptr = y_buf.Lock<BinType>();

            // Binarize
            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    y_ptr.Set(frame, node, x_ptr.Get(frame, node) > (RealType)0.0 ? (BinType)1.0 : (BinType)0.0);
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
        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        // 戻り値のサイズ設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), dy_buf.GetShape(), dy_buf.GetType());
        
        FrameBuffer x_buf = m_x_buf;
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
            index_t node_size = dx_buf.GetNodeSize();

            auto x_ptr  = x_buf.LockConst<RealType>();
            auto dy_ptr = dy_buf.LockConst<RealType>();
            auto dx_ptr = dx_buf.Lock<RealType>();
            
            // hard-tanh
    #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto dy = dy_ptr.Get(frame, node);
                    auto x  = x_ptr.Get(frame, node);
                    if ( x <= m_hardtanh_min || x >= m_hardtanh_max) { dy = (RealType)0.0; }
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
        bb::SaveValue(os, m_host_only);
        bb::SaveValue(os, m_binary_th);
        bb::SaveValue(os, m_hardtanh_min);
        bb::SaveValue(os, m_hardtanh_max);
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
        bb::LoadValue(is, m_host_only);
        bb::LoadValue(is, m_binary_th);
        bb::LoadValue(is, m_hardtanh_min);
        bb::LoadValue(is, m_hardtanh_max);
    }
};


};

