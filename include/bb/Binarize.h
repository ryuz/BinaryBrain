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
template <typename T = float, typename YT = float>
class Binarize : public Activation<T, T>
{
protected:
    T           m_binary_th    = (T)0;
    T           m_hardtanh_min = (T)-1;
    T           m_hardtanh_max = (T)+1;

    FrameBuffer m_x_buf;

    bool        m_host_only = false;

public:
    // 生成情報
    struct create_t
    {
        T   binary_th    = (T)0;
        T   hardtanh_min = (T)-1;
        T   hardtanh_max = (T)+1;
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
    void CommandProc(std::vector<std::string> args)
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

    static std::shared_ptr<Binarize> Create(T binary_th = (T)0, T hardtanh_min = (T)-1, T hardtanh_max = (T)+1)
    {
        create_t create;
        create.binary_th    = binary_th;
        create.hardtanh_min = hardtanh_min;
        create.hardtanh_max = hardtanh_max;
        return Create(create);
    }


    std::string GetClassName(void) const { return "Binarize"; }
    
    
    // ノード単位でのForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {
        std::vector<double> y_vec;
        for ( auto x : x_vec ) {
            y_vec.push_back((x > 0.0) ? 1.0 : 0.0);
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
    inline FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        BB_ASSERT(x_buf.GetType() == DataType<T>::type);

        // backwardの為に保存
        if ( train ) {
            m_x_buf = x_buf;
        }

        // 戻り値のサイズ設定
        FrameBuffer y_buf(DataType<YT>::type, x_buf.GetFrameSize(), x_buf.GetShape());

#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && DataType<YT>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory();
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
        
        {
            // 汎用版
            index_t frame_size = x_buf.GetFrameSize();
            index_t node_size = x_buf.GetNodeSize();

            auto x_ptr = x_buf.LockConst<T>();
            auto y_ptr = y_buf.Lock<YT>();

            // Binarize
            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    y_ptr.Set(frame, node, x_ptr.Get(frame, node) > (T)0.0 ? (YT)1.0 : (YT)0.0);
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
    inline FrameBuffer Backward(FrameBuffer dy_buf)
    {
        BB_ASSERT(dy_buf.GetType() == DataType<T>::type);

        // 戻り値のサイズ設定
        FrameBuffer dx_buf(dy_buf.GetType(), dy_buf.GetFrameSize(), dy_buf.GetShape());
        
        FrameBuffer x_buf = m_x_buf;
        m_x_buf = FrameBuffer();

#ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && !m_host_only && x_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
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

            auto x_ptr  = x_buf.LockConst<T>();
            auto dy_ptr = dy_buf.LockConst<T>();
            auto dx_ptr = dx_buf.Lock<T>();
            
            // hard-tanh
    #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto dy = dy_ptr.Get(frame, node);
                    auto x  = x_ptr.Get(frame, node);
                    if ( x <= (T)-1.0 || x >= (T)1.0) { dy = (T)0.0; }
                    dx_ptr.Set(frame, node, dy);
                }
            }

            return dx_buf;
        }
    }
};


};

