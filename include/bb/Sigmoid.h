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
template <typename T = float>
class Sigmoid : public Binarize<T>
{
protected:
    bool m_binary_mode = false;

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
    std::vector<T> ForwardNode(index_t node, std::vector<T> x_vec) const
    {
        if ( m_binary_mode ) {
            return Binarize<T>::ForwardNode(node, x_vec);
        }

        std::vector<T> y_vec;
        for ( auto x : x_vec ) {
		    y_vec.push_back((T)1 / ((T)1 + std::exp(-x))); // Sigmoid
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
    inline FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        // binaryモード
    	if (m_binary_mode) {
            return Binarize<T>::Forward(x, train);
        }

        BB_ASSERT(x.GetType() == DataType<T>::type);

        // backward用に保存
        m_x = x;

        // 戻り値のサイズ設定
        m_y.ResizeLike(x);

        index_t frame_size = m_x.GetFrameSize();
        index_t node_size = m_x.GetNodeSize();

		auto x_ptr = m_x.LockConst<T>();
		auto y_ptr = m_y.Lock<T>();

		// Sigmoid
#pragma omp parallel for
		for (index_t node = 0; node < node_size; ++node) {
			for (index_t frame = 0; frame < frame_size; ++frame) {
                auto sig = x_ptr.Get(frame, node);
				y_ptr.Set(frame, node, (T)1 / ((T)1 + std::exp(-sig)));
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
	inline FrameBuffer Backward(FrameBuffer dy)
    {
        // binaryモード
  		if (m_binary_mode) {
            return Binarize<T>::Backward(dy);
        }

        BB_ASSERT(dy.GetType() == DataType<T>::type);

        // 戻り値のサイズ設定
        m_dx.ResizeLike(dy);

        index_t frame_size = m_dx.GetFrameSize();
        index_t node_size = m_dx.GetNodeSize();

	    auto y_ptr  = m_y.LockConst<T>();
	    auto dy_ptr = dy.LockConst<T>();
	    auto dx_ptr = m_dx.Lock<T>();

        // Sigmoid
#pragma omp parallel for
		for (index_t node = 0; node < node_size; ++node) {
			for (index_t frame = 0; frame < frame_size; ++frame) {
                auto sig  = y_ptr.Get(frame, node);
                auto grad = dy_ptr.Get(frame, node);
				dx_ptr.Set(frame, node, grad * (-sig + (T)1) * sig);
			}
		}
        return m_dx;
    }
};


}


