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

#include "bb/NeuralNetLossFunction.h"


namespace bb {


template <typename T = float>
class NeuralNetLossMeanSquaredError : public NeuralNetLossFunction<T>
{
protected:
	using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using Stride = Eigen::Stride<Eigen::Dynamic, 1>;
	using MatMap = Eigen::Map<Matrix, 0, Stride>;

public:
	NeuralNetLossMeanSquaredError() {}
	~NeuralNetLossMeanSquaredError() {}
	
	double CalculateLoss(NeuralNetBuffer<T> buf_sig, NeuralNetBuffer<T> buf_err, typename std::vector< std::vector<T> >::const_iterator t_begin) const
	{
		INDEX frame_size = buf_sig.GetFrameSize();
		INDEX node_size = buf_sig.GetNodeSize();
		INDEX stride_size = buf_sig.GetFrameStride() / sizeof(T);

		MatMap x((T*)buf_sig.GetBuffer(), frame_size, node_size, Stride(stride_size, 1));
		MatMap dx((T*)buf_err.GetBuffer(), frame_size, node_size, Stride(stride_size, 1));

		Matrix t(frame_size, node_size);
		for (INDEX frame = 0; frame < frame_size; ++frame) {
			for (INDEX node = 0; node < node_size; ++node) {
				t(frame, node) = (*t_begin)[node];
			}
			++t_begin;
		}

		auto x_exp = (x.colwise() - x.rowwise().maxCoeff()).array().exp();
		auto x_sum = x_exp.rowwise().sum();
		Matrix y = x_exp.array().colwise() / x_sum.array();

		dx = y - t;

		return (dx.array() * dx.array()).mean();
	}
};


}

