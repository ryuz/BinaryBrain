// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/NeuralNetLayerBuf.h"


namespace bb {


// Dropout
template <typename T = float>
class NeuralNetDropout : public NeuralNetLayerBuf<T>
{
protected:
	INDEX				m_frame_size = 1;
	INDEX				m_node_size = 0;
	bool				m_binary_mode = false;

	double				m_rate = 0.5;
	std::mt19937_64		m_mt;
	std::vector<bool>	m_mask;

public:
	NeuralNetDropout() {}

	NeuralNetDropout(INDEX node_size, double rate, std::int64_t seed=1)
	{
		m_rate = rate;
		m_mt.seed(seed);
		Resize(node_size);
	}

	~NeuralNetDropout() {}

	std::string GetClassName(void) const { return "NeuralNetDropout"; }
	
	void Resize(INDEX node_size)
	{
		m_node_size = node_size;
		m_mask.resize(m_node_size);
	}
	
	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }
	
	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetInputNodeSize(void) const { return m_node_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputNodeSize(void) const { return m_node_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }


	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		if (m_binary_mode) {
			return input_value;
		}
		else {
			for (auto& v : input_value) {
				v = m_mask[node] ? v : 0;
			}
		}
		return input_value;
	}

	void Forward(bool train = true)
	{
		if (m_binary_mode) {
			// Binarize
			auto x = this->GetInputSignalBuffer();
			auto y = this->GetOutputSignalBuffer();

#pragma omp parallel for
			for (int node = 0; node < (int)m_node_size; ++node) {
				for (INDEX frame = 0; frame < m_frame_size; ++frame) {
					y.template Set<T>(frame, node, x.template Get<T>(frame, node));
				}
			}
		}
		else {
			auto x = this->GetInputSignalBuffer();
			auto y = this->GetOutputSignalBuffer();

			if (train) {
				// generate mask
				std::uniform_real_distribution<double> dist(0.0, 1.0);
				for (INDEX node = 0; node < m_node_size; ++node) {
					m_mask[node] = (dist(m_mt) > m_rate);
				}

#pragma omp parallel for
				for (int node = 0; node < (int)m_node_size; ++node) {
					if (m_mask[node]) {
						for (INDEX frame = 0; frame < m_frame_size; ++frame) {
							y.template Set<T>(frame, node, x.template Get<T>(frame, node));
						}
					}
					else {
						for (INDEX frame = 0; frame < m_frame_size; ++frame) {
							y.template Set<T>(frame, node, 0);
						}
					}
				}
			}
			else {
#pragma omp parallel for
				for (int node = 0; node < (int)m_node_size; ++node) {
					for (INDEX frame = 0; frame < m_frame_size; ++frame) {
						y.template Set<T>(frame, node, (T)(x.template Get<T>(frame, node) * (1.0 - m_rate)));
					}
				}
			}
		}
	}

	void Backward(void)
	{
		if (m_binary_mode) {
			// Binarize
			auto dx = this->GetInputErrorBuffer();
			auto dy = this->GetOutputErrorBuffer();

#pragma omp parallel for
			for (int node = 0; node < (int)m_node_size; ++node) {
				for (INDEX frame = 0; frame < m_frame_size; ++frame) {
					dx.template Set<T>(frame, node, dy.template Get<T>(frame, node));
				}
			}
		}
		else {
			auto dx = this->GetInputErrorBuffer();
			auto dy = this->GetOutputErrorBuffer();

#pragma omp parallel for
			for (int node = 0; node < (int)m_node_size; ++node) {
				if (m_mask[node]) {
					for (INDEX frame = 0; frame < m_frame_size; ++frame) {
						dx.template Set<T>(frame, node, dy.template Get<T>(frame, node));
					}
				}
				else {
					for (INDEX frame = 0; frame < m_frame_size; ++frame) {
						dx.template Set<T>(frame, node, 0);
					}
				}
			}
		}
	}

	void Update(void)
	{
	}

};

}
