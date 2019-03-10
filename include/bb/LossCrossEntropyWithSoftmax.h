// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>

#include <Eigen/Core>

#include "bb/LossFunction.h"


namespace bb {

template <typename T = float>
class LossCrossEntropyWithSoftmax : public LossFunction
{
	using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using Stride = Eigen::Stride<Eigen::Dynamic, 1>;
	using MatMap = Eigen::Map<Matrix, 0, Stride>;
protected:
    FrameBuffer m_dy;
    double      m_loss;

protected:
	LossCrossEntropyWithSoftmax() {}

public:
	~LossCrossEntropyWithSoftmax() {}
	
    static std::shared_ptr<LossCrossEntropyWithSoftmax> Create(void)
    {
        auto self = std::shared_ptr<LossCrossEntropyWithSoftmax>(new LossCrossEntropyWithSoftmax);
        return self;
    }

    void Clear(void)
    {
        m_loss = 0;
    }

    double GetLoss(void) const 
    {
        return m_loss;
    }

    FrameBuffer CalculateLoss(FrameBuffer y, FrameBuffer t)
	{
		index_t frame_size  = y.GetFrameSize();
		index_t node_size   = y.GetNodeSize();
		index_t stride_size = y.GetFrameStride() / sizeof(T);

        m_dy.Resize(y.GetType(), y.GetFrameSize(), y.GetShape());

        auto y_ptr  = y.LockMemoryConst();
        auto t_ptr  = t.LockMemoryConst();
        auto dy_ptr = m_dy.LockMemory(true);

		MatMap y_mat ((T*)y_ptr.GetAddr(),  frame_size, node_size, Stride(stride_size, 1));
		MatMap t_mat ((T*)t_ptr.GetAddr(),  frame_size, node_size, Stride(stride_size, 1));
		MatMap dy_mat((T*)dy_ptr.GetAddr(), frame_size, node_size, Stride(stride_size, 1));

		auto x_exp = (y_mat.colwise() - y_mat.rowwise().maxCoeff()).array().exp();
		auto x_sum = x_exp.rowwise().sum();
		Matrix tmp = x_exp.array().colwise() / x_sum.array();

		dy_mat = (tmp - t_mat).array() * ((T)1 / frame_size);

        m_loss = -(tmp.array().log().array() * t_mat.array()).sum() / (double)frame_size;

        return m_dy;
	}
};


}

