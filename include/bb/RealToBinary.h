// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>

#include "bb/Layer.h"


namespace bb {



/**
 * @brief   バイナリ変調を行いながらバイナライズを行う
 * @details 閾値を乱数などで変調して実数をバイナリに変換する
 *          入力に対して出力は frame_mux_size 倍のフレーム数となる
 *          入力値に応じて 0と1 を確率的に発生させることを目的としている
 *          RealToBinary と組み合わせて使う想定
 * 
 * @tparam FXT  foward入力型 (x)
 * @tparam FXT  foward出力型 (y)
 * @tparam BT   backward型 (dy, dx)
 */
template <typename FXT = float, typename FYT = float, typename BT = float>
class RealToBinary : public Model
{
protected:
    FrameBuffer         m_y;
    FrameBuffer         m_dx;

	indices_t			m_node_shape;
    index_t             m_frame_mux_size;

	FXT					m_real_range_lo = (FXT)0.0;
	FXT					m_real_range_hi = (FXT)1.0;

	std::mt19937_64		m_mt;

protected:
	RealToBinary() {}
public:
	~RealToBinary() {}

    struct create_t
    {
        index_t         frame_mux_size = 1;
        FXT             real_range_lo = (FXT)0.0;
        FXT             real_range_hi = (FXT)1.0;
        std::uint64_t   seed = 1;
    };

    static std::shared_ptr<RealToBinary> Create(create_t const &create)
    {
        auto self = std::shared_ptr<RealToBinary>(new RealToBinary);

        self->m_frame_mux_size = create.frame_mux_size;
	    self->m_real_range_lo  = create.real_range_lo;
	    self->m_real_range_hi  = create.real_range_hi;

        self->m_mt.seed(create.seed);

        return self;
    }

    static std::shared_ptr<RealToBinary> Create(index_t frame_mux_size, std::uint64_t seed = 1)
    {
        create_t create;
        create.frame_mux_size = frame_mux_size;
        create.seed     = seed;
        return Create(create);
    }

	std::string GetClassName(void) const { return "RealToBinary"; }


    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 形状設定
        m_node_shape = shape;
        return m_node_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_node_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_node_shape;
    }
    

    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        BB_ASSERT(x.GetType() == DataType<FXT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x.GetShape() != m_node_shape) {
            SetInputShape(x.GetShape());
        }

        // 戻り値の型を設定
        m_y.Resize(DataType<FYT>::type, x.GetFrameSize() * m_frame_mux_size, m_node_shape);

		std::uniform_real_distribution<FXT>	dist_rand(m_real_range_lo, m_real_range_hi);

		index_t node_size         = m_y.GetNodeSize();
		index_t output_frame_size = m_y.GetFrameSize();

        auto x_ptr = x.GetConstPtr<FXT>();
        auto y_ptr = m_y.GetPtr<FYT>();

		for (index_t node = 0; node < node_size; ++node) {
    	    for (index_t output_frame = 0; output_frame < output_frame_size; ++output_frame) {
                index_t input_frame = output_frame / m_frame_mux_size;
            
			    FXT th = dist_rand(m_mt);
                FXT real_sig = x_ptr.Get(input_frame, node);
			    FYT bin_sig  = (real_sig > th) ? (FYT)1 : (FYT)0;
			    y_ptr.Set(output_frame, node, bin_sig);
		    }
		}

        return m_y;
	}


	FrameBuffer Backward(FrameBuffer dy)
	{
        BB_ASSERT(dy.GetType() == DataType<BT>::type);

        // 戻り値の型を設定
        m_dx.Resize(DataType<BT>::type, dy.GetFrameSize() / m_frame_mux_size, m_node_shape);

		index_t node_size         = dy.GetNodeSize();
		index_t output_frame_size = dy.GetFrameSize();

        m_dx.FillZero();

        auto dy_ptr = dy.GetConstPtr<BT>();
        auto dx_ptr = m_dx.GetPtr<BT>();

		for (index_t node = 0; node < node_size; node++) {
    		for (index_t output_frame = 0; output_frame < output_frame_size; ++output_frame) {
                index_t input_frame = output_frame / m_frame_mux_size;

                BT grad = dy_ptr.Get(output_frame, node);
				dx_ptr.Add(input_frame, node, grad);
			}
		}

        return m_dx;
	}
};


}


// end of file
