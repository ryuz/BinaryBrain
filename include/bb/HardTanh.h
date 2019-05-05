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


// ReLU(活性化層)
template <typename T = float>
class HardTanh : public Binarize<T>
{
protected:
    bool        m_binary_mode = false;

    using Binarize<T>::m_host_only;

    using Binarize<T>::m_binary_th;
    using Binarize<T>::m_hardtanh_min;
    using Binarize<T>::m_hardtanh_max;

    using Binarize<T>::m_x_buf;
    using Binarize<T>::m_y_buf;
    using Binarize<T>::m_dx_buf;

public:
    // 生成情報
    struct create_t
    {
        T   hardtanh_min = (T)-1;
        T   hardtanh_max = (T)+1;
    };

protected:
    HardTanh(create_t const &create)
    {
        m_hardtanh_min = create.hardtanh_min;
        m_hardtanh_max = create.hardtanh_max;
        m_binary_th    = (m_hardtanh_min + m_hardtanh_max) / (T)2;
    }

    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args)
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

    static std::shared_ptr<HardTanh> Create(T hardtanh_min = (T)-1, T hardtanh_max = (T)+1)
    {
        create_t create;
        create.hardtanh_min = hardtanh_min;
        create.hardtanh_max = hardtanh_max;
        return Create(create);
    }

    std::string GetClassName(void) const { return "HardTanh"; }


    // 1ノードのみForward計算
    std::vector<T> ForwardNode(index_t node, std::vector<T> x_vec) const
    {
        if ( m_binary_mode ) {
            return Binarize<T>::ForwardNode(node, x_vec);
        }

        for ( auto& x : x_vec ) {
            if ( x <= m_hardtanh_min ) { x = m_hardtanh_min; }
            if ( x >= m_hardtanh_max ) { x = m_hardtanh_max; }
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
    inline FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        // binaryモード
        if (m_binary_mode) {
            return Binarize<T>::Forward(x_buf, train);
        }

        BB_ASSERT(x_buf.GetType() == DataType<T>::type);

        // backward用に保存
        m_x_buf = x_buf;

        // 戻り値のサイズ設定
        m_y_buf.ResizeLike(x_buf);

#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && !m_host_only && m_x_buf.IsDeviceAvailable() && m_y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = m_y_buf.LockDeviceMemory();
            bbcu_fp32_HardTanh_Forward(
                        (float const *)ptr_x.GetAddr(),
                        (float       *)ptr_y.GetAddr(),
                        (float        )m_hardtanh_min,
                        (float        )m_hardtanh_max,
                        (int          )m_y_buf.GetNodeSize(),
                        (int          )m_y_buf.GetFrameSize(),
                        (int          )(m_y_buf.GetFrameStride() / sizeof(float))
                    );
            return m_y_buf;
        }
#endif

        {
            // 汎用版
            index_t frame_size = m_x_buf.GetFrameSize();
            index_t node_size = m_x_buf.GetNodeSize();

            auto x_ptr = m_x_buf.template LockConst<T>();
            auto y_ptr = m_y_buf.template Lock<T>();

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
            return m_y_buf;
        }
    }


   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    inline FrameBuffer Backward(FrameBuffer dy_buf)
    {
        // binaryモード
        if (m_binary_mode) {
            return Binarize<T>::Backward(dy_buf);
        }

        BB_ASSERT(dy_buf.GetType() == DataType<T>::type);

        // 戻り値のサイズ設定
        m_dx_buf.ResizeLike(dy_buf);

#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && !m_host_only && m_x_buf.IsDeviceAvailable() && m_dx_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // GPU版
            auto ptr_x  = m_x_buf.LockDeviceMemoryConst();
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = m_dx_buf.LockDeviceMemory(true);
            bbcu_fp32_HardTanh_Backward(
                        (float const *)ptr_x.GetAddr(),
                        (float const *)ptr_dy.GetAddr(),
                        (float       *)ptr_dx.GetAddr(),
                        (float        )m_hardtanh_min,
                        (float        )m_hardtanh_max,
                        (int          )m_dx_buf.GetNodeSize(),
                        (int          )m_dx_buf.GetFrameSize(),
                        (int          )(m_dx_buf.GetFrameStride() / sizeof(float))
                    );
            return m_dx_buf;
        }
#endif

        {
            // 汎用版
            index_t frame_size = m_dx_buf.GetFrameSize();
            index_t node_size = m_dx_buf.GetNodeSize();

            auto x_ptr  = m_x_buf.template LockConst<T>();
            auto dy_ptr = dy_buf.template LockConst<T>();
            auto dx_ptr = m_dx_buf.template Lock<T>();

            // Hard-Tanh
            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto x  = x_ptr.Get(frame, node);
                    auto dy = dy_ptr.Get(frame, node);
                    if ( x <= m_hardtanh_min ) { dy = (T)0; }
                    if ( x >= m_hardtanh_max ) { dy = (T)0; }
                    dx_ptr.Set(frame, node, dy);
                }
            }

            return m_dx_buf;
        }
    }
};


}


// end of file