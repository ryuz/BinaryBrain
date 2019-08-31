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


// Sigmoid(活性化層)
template <typename BinType = float, typename RealType = float>
class Sigmoid : public Binarize<BinType, RealType>
{
    using _super = Binarize<BinType, RealType>;

protected:
    bool        m_binary_mode = false;

    using _super::m_host_only;
    using _super::m_x_buf;

    FrameBuffer m_y_buf;

protected:
    Sigmoid() {}

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
    static std::shared_ptr<Sigmoid> Create(void)
    {
        auto self = std::shared_ptr<Sigmoid>(new Sigmoid);
        return self;
    }

    ~Sigmoid() {}

    std::string GetClassName(void) const { return "Sigmoid"; }


    /**
     * @brief  入力形状設定
     * @detail 入力形状を設定する
     *         内部変数を初期化し、以降、GetOutputShape()で値取得可能となることとする
     *         同一形状を指定しても内部変数は初期化されるものとする
     * @param  shape      1フレームのノードを構成するshape
     * @return 出力形状を返す
     */
    indices_t SetInputShape(indices_t shape)
    {
        return shape;
    }

    // 1ノードのみForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {
        if ( m_binary_mode ) {
            return _super::ForwardNode(node, x_vec);
        }

        std::vector<double> y_vec;
        for ( auto x : x_vec ) {
            y_vec.push_back((double)((RealType)1 / ((RealType)1 + std::exp(-(RealType)x)))); // Sigmoid
        }

        return y_vec;
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
            return _super::Forward(x_buf, train);
        }

        BB_ASSERT(x_buf.GetType() == DataType<RealType>::type);

        // 戻り値のサイズ設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), x_buf.GetShape(), x_buf.GetType());

        // ローカルに保存
        if ( train ) {
            m_y_buf = y_buf;
        }


#ifdef BB_WITH_CUDA
        if ( DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32 && !this->m_host_only
            && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory(true);
            bbcu_fp32_Sigmoid_Forward(
                        (float const *)ptr_x.GetAddr(),
                        (float       *)ptr_y.GetAddr(),
                        (int          )x_buf.GetNodeSize(),
                        (int          )x_buf.GetFrameSize(),
                        (int          )(x_buf.GetFrameStride() / sizeof(float))
                    );
            return y_buf;
        }
#endif

        {
            index_t frame_size = x_buf.GetFrameSize();
            index_t node_size = x_buf.GetNodeSize();

            auto x_ptr = x_buf.template LockConst<RealType>();
            auto y_ptr = y_buf.template Lock<BinType>();

            // Sigmoid
    #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    RealType sig = x_ptr.Get(frame, node);
                    y_ptr.Set(frame, node, (BinType)((RealType)1 / ((RealType)1 + std::exp(-sig))));
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
        // binaryモード
        if (m_binary_mode) {
            return _super::Backward(dy_buf);
        }

        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        // 戻り値のサイズ設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), dy_buf.GetShape(), dy_buf.GetType());

        FrameBuffer y_buf = m_y_buf;
        m_y_buf = FrameBuffer();

#ifdef BB_WITH_CUDA
        if (  DataType<BinType>::type == BB_TYPE_FP32 && DataType<RealType>::type == BB_TYPE_FP32  && !this->m_host_only
            && y_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // GPU版
            auto ptr_y  = y_buf.LockDeviceMemoryConst();
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = dx_buf.LockDeviceMemory(true);
            bbcu_fp32_Sigmoid_Backward(
                        (float const *)ptr_y.GetAddr(),
                        (float const *)ptr_dy.GetAddr(),
                        (float       *)ptr_dx.GetAddr(),
                        (int          )dy_buf.GetNodeSize(),
                        (int          )dy_buf.GetFrameSize(),
                        (int          )(dy_buf.GetFrameStride() / sizeof(float))
                    );
            return dx_buf;
        }
#endif
        {
            // 汎用版
            index_t frame_size = dx_buf.GetFrameSize();
            index_t node_size = dx_buf.GetNodeSize();

            auto y_ptr  = y_buf.template LockConst<RealType>();
            auto dy_ptr = dy_buf.template LockConst<RealType>();
            auto dx_ptr = dx_buf.template Lock<RealType>();

            // Sigmoid
    #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto sig  = y_ptr.Get(frame, node);
                    auto grad = dy_ptr.Get(frame, node);
                    dx_ptr.Set(frame, node, grad * (-sig + (RealType)1) * sig);
                }
            }
            return dx_buf;
        }
    }
};


}


