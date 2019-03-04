// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>

#include "bb/Model.h"


namespace bb {


template <typename FXT = float, typename FYT = float, typename BT = float>
class BinaryToReal : public Model
{
protected:
    FrameBuffer         m_y;
    FrameBuffer         m_dx;

	indices_t			m_input_shape;
	indices_t			m_output_shape;
    index_t             m_frame_mux_size;

protected:
	BinaryToReal() {}
public:
	~BinaryToReal() {}

    struct create_t
    {
        indices_t       output_shape;   
        index_t         frame_mux_size = 1;
    };

    static std::shared_ptr<BinaryToReal> Create(create_t const &create)
    {
        auto self = std::shared_ptr<BinaryToReal>(new BinaryToReal);

        self->m_output_shape   = create.output_shape;
        self->m_frame_mux_size = create.frame_mux_size;

        return self;
    }

    static std::shared_ptr<BinaryToReal> Create(indices_t output_shape, index_t frame_mux_size=1)
    {
        create_t create;
        create.output_shape   = output_shape;
        create.frame_mux_size = frame_mux_size;
        return Create(create);
    }

	std::string GetClassName(void) const { return "BinaryToReal"; }

    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 形状設定
        m_input_shape = shape;
        return m_output_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_input_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_output_shape;
    }
    

    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        BB_ASSERT(x.GetType() == DataType<FXT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x.GetShape() != m_input_shape) {
            SetInputShape(x.GetShape());
        }

        // 戻り値の型を設定
        BB_ASSERT(x.GetFrameSize() % m_frame_mux_size == 0);
        m_y.Resize(DataType<FYT>::type, x.GetFrameSize() / m_frame_mux_size, m_output_shape);

		auto x_ptr = x.GetConstPtr<FXT>();
		auto y_ptr = m_y.GetPtr<FYT>();

        index_t input_node_size   = GetInputNodeSize();
        index_t output_node_size  = GetOutputNodeSize();
        index_t output_frame_size = m_y.GetFrameSize();

		index_t node_size = std::max(input_node_size, output_node_size);

		std::vector<FYT>	vec_v(output_node_size, (FYT)0);
		std::vector<int>	vec_n(output_node_size, 0);
		for (index_t frame = 0; frame < output_frame_size; ++frame) {
			std::fill(vec_v.begin(), vec_v.end(), (FYT)0);
			std::fill(vec_n.begin(), vec_n.end(), 0);
			for (index_t node = 0; node < node_size; ++node) {
				for (index_t i = 0; i < m_frame_mux_size; ++i) {
					FYT bin_sig = (FYT)x_ptr.Get(frame*m_frame_mux_size + i, node);
					vec_v[node % output_node_size] += bin_sig;
					vec_n[node % output_node_size] += 1;
				}
			}

			for (index_t node = 0; node < output_node_size; ++node) {
				y_ptr.Set(frame, node, (FYT)vec_v[node] / vec_n[node]);
			}
		}

        return m_y;
	}
	

	FrameBuffer Backward(FrameBuffer dy)
	{
        BB_ASSERT(dy.GetType() == DataType<BT>::type);

        // 戻り値の型を設定
        m_dx.Resize(DataType<BT>::type, dy.GetFrameSize() * m_frame_mux_size, m_input_shape);

        index_t input_node_size   = GetInputNodeSize();
        index_t output_node_size  = GetOutputNodeSize();
		index_t output_frame_size = dy.GetFrameSize();

        auto dy_ptr = dy.GetConstPtr<BT>();
        auto dx_ptr = m_dx.GetPtr<BT>();

		BT	gain = (BT)output_node_size / (BT)input_node_size;
		for (index_t node = 0; node < input_node_size; node++) {
			for (index_t frame = 0; frame < output_frame_size; ++frame) {
				for (index_t i = 0; i < m_frame_mux_size; i++) {
					auto grad = dy_ptr.Get(frame, node % output_node_size);
					grad *= gain;
					dx_ptr.Set(frame*m_frame_mux_size + i, node, grad);
				}
			}
		}

        return m_dx;
	}
};

}