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
class LossMeanSquaredError : public LossFunction
{
	using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using Stride = Eigen::Stride<Eigen::Dynamic, 1>;
	using MatMap = Eigen::Map<Matrix, 0, Stride>;
protected:
    FrameBuffer m_dy;
    double      m_loss;
	double		m_frames;

protected:
	LossMeanSquaredError() {}

public:
	~LossMeanSquaredError() {}
	
    static std::shared_ptr<LossMeanSquaredError> Create(void)
    {
        auto self = std::shared_ptr<LossMeanSquaredError>(new LossMeanSquaredError);
        return self;
    }

    void Clear(void)
    {
        m_loss = 0;
		m_frames = 0;
    }

    double GetLoss(void) const 
    {
        return m_loss / m_frames;
    }

    FrameBuffer CalculateLoss(FrameBuffer y, FrameBuffer t)
	{
		index_t frame_size  = y.GetFrameSize();
		index_t node_size   = y.GetNodeSize();

		m_dy.Resize(DataType<T>::type, y.GetFrameSize(), y.GetShape());

		auto y_ptr = y.GetConstPtr<T>();
		auto t_ptr = t.GetConstPtr<T>();
		auto dy_ptr = m_dy.GetPtr<T>();

		for (index_t frame = 0; frame < frame_size; ++frame) {
			for (index_t node = 0; node < node_size; ++node) {
				auto signal = y_ptr.Get(frame, node);
				auto target = t_ptr.Get(frame, node);
				auto grad = signal - target;
				auto error = grad * grad;

				dy_ptr.Set(frame, node, grad / (T)frame_size);
				m_loss += error;
			}
			m_frames++;
		}

        return m_dy;
	}
};


}

