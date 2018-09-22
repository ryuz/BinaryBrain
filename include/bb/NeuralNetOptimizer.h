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


namespace bb {


template <typename T = float, typename INDEX = size_t>
class ParamOptimizer
{
public:
	virtual ~ParamOptimizer() {}

protected:
	virtual void UpdateParam(INDEX index, T& param, const T grad) = 0;
	virtual void PreUpdate(void) {};
	virtual void PostUpdate(void) {};

public:
	virtual void Update(T& param, const T grad)
	{
		PreUpdate();
		UpdateParam(0, param, grad);
		PostUpdate();
	}

	virtual void Update(std::vector<T>& param, const std::vector<T>& grad)
	{
		BB_ASSERT(param.size() == grad.size());

		PreUpdate();
		for (INDEX i = 0; i < (INDEX)param.size(); ++i) {
			UpdateParam(i, param[i], grad[i]);
		}
		PostUpdate();
	}

	template<INDEX N>
	void Update(std::array<T, N>& param, const std::array<T, N>& grad)
	{
		PreUpdate();
		for (INDEX i = 0; i < N; ++i) {
			UpdateParam(i, param[i], grad[i]);
		}
		PostUpdate();
	}

	template<typename Matrix>
	void Update(Matrix& param, const Matrix& grad)
	{
		BB_ASSERT(param.size() == grad.size());
		PreUpdate();
		for (INDEX i = 0; i < (INDEX)param.size(); ++i) {
			UpdateParam(i, param.data()[i], grad.data()[i]);
		}
		PostUpdate();

	}
};


template <typename T = float, typename INDEX = size_t>
class NeuralNetOptimizer
{
public:
	virtual ParamOptimizer<T, INDEX>* Create(INDEX param_size) const = 0;
};


}