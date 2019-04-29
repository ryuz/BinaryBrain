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
    bool        m_host_only   = false;

    using Binarize<T>::m_x_buf;
    using Binarize<T>::m_y_buf;
    using Binarize<T>::m_dx_buf;

protected:
    HardTanh() {}

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
    static std::shared_ptr<ReLU> Create(void)
    {
        auto self = std::shared_ptr<ReLU>(new ReLU);
        return self;
    }

    ~HardTanh() {}

    std::string GetClassName(void) const { return "HardTanh"; }


    // 1ノードのみForward計算
    std::vector<T> ForwardNode(index_t node, std::vector<T> x_vec) const
    {
        if ( m_binary_mode ) {
            return Binarize<T>::ForwardNode(node, x_vec);
        }

        for ( auto& x : x_vec ) {
            if ( x < (T)0.0 ) { x = (T)0.0; }
            if ( x > (T)1.0 ) { x = (T)1.0; }
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
            return Binarize<T>::Forward(x, train);
        }

        BB_ASSERT(x_buf.GetType() == DataType<T>::type);

        // backward用に保存
        m_x_buf = x_buf;

        // 戻り値のサイズ設定
        m_y_buf.ResizeLike(x);

        index_t frame_size = m_x.GetFrameSize();
        index_t node_size = m_x.GetNodeSize();

        auto x_ptr = m_x_buf.template LockConst<T>();
        auto y_ptr = m_y_buf.template Lock<T>();

        // Hard-Tanh
#pragma omp parallel for
        for (index_t node = 0; node < node_size; ++node) {
            for (index_t frame = 0; frame < frame_size; ++frame) {
                auto x = x_ptr.Get(frame, node);
                if ( x < (T)0.0 ) { x = (T)0.0; }
                if ( x > (T)1.0 ) { x = (T)1.0; }
                y_ptr.Set(frame, node, x);
            }
        }
        return m_y;
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

        BB_ASSERT(dy.GetType() == DataType<T>::type);

        // 戻り値のサイズ設定
        m_dx.ResizeLike(dy);

        index_t frame_size = m_dx.GetFrameSize();
        index_t node_size = m_dx.GetNodeSize();

        auto x_ptr  = m_x_buf.template LockConst<T>();
        auto dy_ptr = dy_buf.template LockConst<T>();
        auto dx_ptr = m_dx_buf.template Lock<T>();

        // Hard-Tanh
        #pragma omp parallel for
        for (index_t node = 0; node < node_size; ++node) {
            for (index_t frame = 0; frame < frame_size; ++frame) {
                auto x  = x_ptr.Get(frame, node);
                auto dy = dy_ptr.Get(frame, node);
                if ( x < (T)0.0 || x > (T)1.0 ) { dy = 0; }
                dx_ptr.Set(frame, node, dy);
            }
        }

        return m_dx;
    }
};


}


// end of file