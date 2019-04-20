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
class ReLU : public Binarize<T>
{
protected:
    bool        m_binary_mode = false;
    bool        m_host_only   = false;

    using Binarize<T>::m_x;
    using Binarize<T>::m_y;
    using Binarize<T>::m_dx;

protected:
	ReLU() {}

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

	~ReLU() {}

	std::string GetClassName(void) const { return "ReLU"; }


    // 1ノードのみForward計算
    std::vector<T> ForwardNode(index_t node, std::vector<T> x_vec) const
    {
        if ( m_binary_mode ) {
            return Binarize<T>::ForwardNode(node, x_vec);
        }

        std::vector<T> y_vec;
        for ( auto x : x_vec ) {
		    y_vec.push_back((x > (T)0.0) ? x : (T)0.0); // ReLU
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

		auto x_ptr = m_x.template LockConst<T>();
		auto y_ptr = m_y.template Lock<T>();

		// ReLU
#pragma omp parallel for
		for (index_t node = 0; node < node_size; ++node) {
			for (index_t frame = 0; frame < frame_size; ++frame) {
                auto sig = x_ptr.Get(frame, node);
				y_ptr.Set(frame, node, sig > (T)0.0 ? sig : (T)0.0);
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

		auto y_ptr  = m_y.template LockConst<T>();
		auto dy_ptr = dy.template LockConst<T>();
		auto dx_ptr = m_dx.template Lock<T>();

        // ReLU
#pragma omp parallel for
		for (index_t node = 0; node < node_size; ++node) {
			for (index_t frame = 0; frame < frame_size; ++frame) {
                auto sig  = y_ptr.Get(frame, node);
                auto grad = dy_ptr.Get(frame, node);
				dx_ptr.Set(frame, node, (sig > (T)0) ? grad : (T)0);
			}
		}

        return m_dx;
    }
};



/**
 * @brief  forward演算
 * @detail forward演算を行う
 * @param  x     入力データ
 * @param  train 学習時にtrueを指定
 * @return forward演算結果
 */
template<>
inline FrameBuffer ReLU<float>::Forward(FrameBuffer x, bool train)
{
    if ( m_binary_mode ) {
        return Binarize<float>::Forward(x, train);
    }

    BB_ASSERT(x.GetType() == BB_TYPE_FP32);

    // backward用に保存
    m_x = x;

    // 戻り値サイズ設定
    m_y.ResizeLike(x);

    // ReLU
#if BB_WITH_CUDA
    if ( !m_host_only && m_x.IsDeviceAvailable() && m_y.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        // CUDA版
        auto ptr_x = x.LockDeviceMemoryConst();
        auto ptr_y = m_y.LockDeviceMemory(true);
        bbcu_fp32_ReLU_Forward(
                    (float const *)ptr_x.GetAddr(),
                    (float       *)ptr_y.GetAddr(),
                    (int          )m_x.GetNodeSize(),
                    (int          )m_x.GetFrameSize(),
                    (int          )(m_x.GetFrameStride() / sizeof(float))
                );
        return m_y;
    }
#endif

    {
        // AVX版
        index_t frame_size = m_x.GetFrameSize();
        index_t node_size = m_x.GetNodeSize();

        auto x_ptr = m_x.LockConst<float>();
	    auto y_ptr = m_y.Lock<float>(true);

		index_t  m256_frame_size = (int)(((frame_size + 7) / 8) * 8);
		__m256 zero = _mm256_set1_ps(0);
		for (index_t node = 0; node < node_size; ++node) {
		    auto x_addr = (float const *)x_ptr.GetAddr(node);
		    auto y_addr = (float *)y_ptr.GetAddr(node);
		    for (index_t frame = 0; frame < m256_frame_size; frame += 8) {
			    __m256 in_sig = _mm256_load_ps(&x_addr[frame]);
			    in_sig = _mm256_max_ps(in_sig, zero);
			    _mm256_store_ps(&y_addr[frame], in_sig);
		    }
		}
        return m_y;
    }
}



/**
  * @brief  backward演算
  * @detail backward演算を行う
  *         
  * @return backward演算結果
  */
template<>
inline FrameBuffer ReLU<float>::Backward(FrameBuffer dy)
{
    if ( m_binary_mode ) {
        return Binarize<float>::Backward(dy);
    }

    BB_ASSERT(dy.GetType() == BB_TYPE_FP32);

    // 戻り値サイズ設定
    m_dx.ResizeLike(dy);

#if BB_WITH_CUDA
    if ( !m_host_only && m_x.IsDeviceAvailable() && m_dx.IsDeviceAvailable() && dy.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        // GPU版
        auto ptr_x  = m_x.LockDeviceMemoryConst();
        auto ptr_dy = dy.LockDeviceMemoryConst();
        auto ptr_dx = m_dx.LockDeviceMemory(true);
        bbcu_fp32_ReLU_Backward(
                    (float const *)ptr_x.GetAddr(),
                    (float const *)ptr_dy.GetAddr(),
                    (float       *)ptr_dx.GetAddr(),
                    (int          )dy.GetNodeSize(),
                    (int          )dy.GetFrameSize(),
                    (int          )(dy.GetFrameStride() / sizeof(float))
                );
        return m_dx;
    }
#endif

    {
        // AVX
        index_t frame_size = m_dx.GetFrameSize();
        index_t node_size = m_dx.GetNodeSize();

        auto x_ptr  = m_x.LockConst<float>();
	    auto y_ptr  = m_y.LockConst<float>();
	    auto dy_ptr = dy.LockConst<float>();
	    auto dx_ptr = m_dx.Lock<float>(true);

        index_t  m256_frame_size = (int)(((frame_size + 7) / 8) * 8);

		__m256 zero = _mm256_set1_ps(0);
		for (index_t node = 0; node < node_size; ++node) {
			auto y_addr  = (float *)y_ptr.GetAddr(node);
			auto dy_addr = (float *)dy_ptr.GetAddr(node);
			auto dx_addr = (float *)dx_ptr.GetAddr(node);
			for (index_t frame = 0; frame < m256_frame_size; frame += 8) {
				__m256 y    = _mm256_load_ps(&y_addr[frame]);
				__m256 dy   = _mm256_load_ps(&dy_addr[frame]);
				__m256 mask = _mm256_cmp_ps(y, zero, _CMP_GT_OS);
				__m256 dx   = _mm256_and_ps(dy, mask);
				_mm256_store_ps(&dx_addr[frame], dx);
			}
		}
        return m_dx;
    }
}


};


// end of file