// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>

#include "bb/AccuracyFunction.h"


namespace bb {


template <typename T = float>
class AccuracyMeanSquaredError : public AccuracyFunction
{
protected:
	double  m_accuracy = 0;
    index_t m_frames;


protected:
	AccuracyMeanSquaredError()	{}

public:
	~AccuracyMeanSquaredError() {}

    static std::shared_ptr<AccuracyMeanSquaredError> Create(void)
    {
        auto self = std::shared_ptr<AccuracyMeanSquaredError>(new AccuracyMeanSquaredError);
        return self;
    }
     
    void Clear(void)
    {
        m_accuracy = 0;
        m_frames   = 0;
    }


    double GetAccuracy(void) const
    {
        return m_accuracy / (double)m_frames;
    }

	void CalculateAccuracy(FrameBuffer y, FrameBuffer t)
	{
		BB_ASSERT(y.GetType() == DataType<T>::type);
		BB_ASSERT(t.GetType() == DataType<T>::type);

		index_t frame_size = y.GetFrameSize();
		index_t node_size = y.GetNodeSize();

		auto y_ptr = y.LockConst<T>();
		auto t_ptr = t.LockConst<T>();

		for (index_t frame = 0; frame < frame_size; ++frame) {
			for (index_t node = 0; node < node_size; ++node) {
				auto signal = y_ptr.Get(frame, node);
				auto target = t_ptr.Get(frame, node);
				auto grad = target - signal;
				auto error = grad * grad;
				m_accuracy += error;
			}
			m_frames++;
		}
	}

};


}

