// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>

#include "bb/NeuralNetLayer.h"
#include "bb/NeuralNetRealToBinary.h"
#include "bb/NeuralNetBinaryToReal.h"


namespace bb {


// バイナリを多重化して評価
template <typename BT = bool, typename T = float, typename INDEX = size_t>
class NeuralNetBinaryMultiplex : public NeuralNetLayer<T, INDEX>
{
protected:
	// 3層で構成
	NeuralNetRealToBinary<BT, T, INDEX>		m_real2bin;
	NeuralNetLayer<T, INDEX>*				m_layer;
	NeuralNetBinaryToReal<BT, T, INDEX>		m_bin2real;
	
	INDEX	m_batch_size = 0;
	INDEX	m_mux_size   = 0;
	
public:
	NeuralNetBinaryMultiplex() {}
	
	NeuralNetBinaryMultiplex(NeuralNetLayer<T, INDEX>* layer, INDEX input_node_size, INDEX output_node_size, INDEX input_hmux_size=1, INDEX output_hmux_size=1)
		: m_real2bin(input_node_size, input_node_size*input_hmux_size), m_bin2real(output_node_size*output_hmux_size, output_node_size)
	{
		m_layer = layer;
	}
	
	~NeuralNetBinaryMultiplex() {}
	
	std::string GetClassName(void) const { return "NeuralNetBinaryMultiplex"; }


	void InitializeCoeff(std::uint64_t seed)
	{
		std::mt19937_64 mt(seed);
		m_real2bin.InitializeCoeff(mt());
		m_layer->InitializeCoeff(mt());
		m_bin2real.InitializeCoeff(mt());
	}
	
	void SetBinaryMode(bool enable)
	{
		m_real2bin.SetBinaryMode(enable);
		m_layer->SetBinaryMode(enable);
		m_bin2real.SetBinaryMode(enable);
	}
	
	void  SetMuxSize(INDEX mux_size)
	{
		if (m_mux_size == mux_size) {
			return;
		}
		m_mux_size = mux_size;
		m_real2bin.SetMuxSize(mux_size);
		m_bin2real.SetMuxSize(mux_size);

		m_batch_size = 0;
	}
	
	void SetOptimizer(const NeuralNetOptimizer<T, INDEX>* optimizer)
	{
		m_real2bin.SetOptimizer(optimizer);
		m_layer->SetOptimizer(optimizer);
		m_bin2real.SetOptimizer(optimizer);
	}

	void  SetBatchSize(INDEX batch_size)
	{
		m_real2bin.SetBatchSize(batch_size);
		m_layer->SetBatchSize(batch_size * m_mux_size);
		m_bin2real.SetBatchSize(batch_size);
		
		if (m_batch_size == batch_size) {
			return;
		}
		m_batch_size = batch_size;
		
		
		// チェック
		CheckConnection(m_real2bin, *m_layer);
		CheckConnection(*m_layer, m_bin2real);

		m_real2bin.SetOutputSignalBuffer(m_real2bin.CreateOutputSignalBuffer());
		m_real2bin.SetOutputErrorBuffer(m_real2bin.CreateOutputErrorBuffer());
		m_layer->SetInputSignalBuffer(m_real2bin.GetOutputSignalBuffer());
		m_layer->SetInputErrorBuffer(m_real2bin.GetOutputErrorBuffer());
		
		m_layer->SetOutputSignalBuffer(m_layer->CreateOutputSignalBuffer());
		m_layer->SetOutputErrorBuffer(m_layer->CreateOutputErrorBuffer());
		m_bin2real.SetInputSignalBuffer(m_layer->GetOutputSignalBuffer());
		m_bin2real.SetInputErrorBuffer(m_layer->GetOutputErrorBuffer());
	}

	
	// 入出力バッファ
	void  SetInputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_real2bin.SetInputSignalBuffer(buffer); }
	void  SetInputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_real2bin.SetInputErrorBuffer(buffer); }
	void  SetOutputSignalBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_bin2real.SetOutputSignalBuffer(buffer); }
	void  SetOutputErrorBuffer(NeuralNetBuffer<T, INDEX> buffer) { m_bin2real.SetOutputErrorBuffer(buffer); }

	const NeuralNetBuffer<T, INDEX>& GetInputSignalBuffer(void) const { return m_real2bin.GetInputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetInputErrorBuffer(void) const { return m_real2bin.GetInputErrorBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputSignalBuffer(void) const { return m_bin2real.GetOutputSignalBuffer(); }
	const NeuralNetBuffer<T, INDEX>& GetOutputErrorBuffer(void) const { return m_bin2real.GetOutputErrorBuffer(); }


	INDEX GetInputFrameSize(void) const { return m_real2bin.GetInputFrameSize(); }
	INDEX GetInputNodeSize(void) const { return m_real2bin.GetInputNodeSize(); }
	int   GetInputSignalDataType(void) const { return m_real2bin.GetInputSignalDataType(); }
	int   GetInputErrorDataType(void) const { return m_real2bin.GetInputErrorDataType(); }

	INDEX GetOutputFrameSize(void) const { return m_bin2real.GetOutputFrameSize(); }
	INDEX GetOutputNodeSize(void) const { return m_bin2real.GetOutputNodeSize(); }
	int   GetOutputSignalDataType(void) const { return m_bin2real.GetOutputSignalDataType(); }
	int   GetOutputErrorDataType(void) const { return m_bin2real.GetOutputErrorDataType(); }


public:

	void Forward(bool train = true)
	{
		m_real2bin.Forward(train);
		m_layer->Forward(train);
		m_bin2real.Forward(train);
	}

	void Backward(void)
	{
		m_bin2real.Backward();
		m_layer->Backward();
		m_real2bin.Backward();
	}

	void Update(void)
	{
		m_bin2real.Update();
		m_layer->Update();
		m_real2bin.Update();
	}
	
	bool Feedback(const std::vector<double>& loss)
	{
		return m_layer->Feedback(loss);
	}


public:

	// 出力の損失関数(ラベル指定)
	template <typename LT, int LABEL_SIZE>
	std::vector<double> GetOutputOnehotLoss(std::vector<LT> label, INDEX offset=0)
	{
		auto buf = m_layer->GetOutputSignalBuffer();
		INDEX frame_size = m_layer->GetOutputFrameSize();
		INDEX node_size = m_layer->GetOutputNodeSize();

		std::vector<double> vec_loss_x(frame_size);
		double* vec_loss = &vec_loss_x[0];

#pragma omp parallel for
		for (int frame = 0; frame < (int)frame_size; ++frame) {
			vec_loss[frame] = 0;
			for (size_t node = 0; node < node_size; ++node) {
				if (label[frame / m_mux_size + offset] == (node % LABEL_SIZE)) {
					vec_loss[frame] += (buf.template Get<BT>(frame, node) ? 0.0 : +1.0);
				}
				else {
					vec_loss[frame] += (buf.template Get<BT>(frame, node) ? +(1.0 / LABEL_SIZE) : -(0.0 / LABEL_SIZE));
				}
			}
		}

		return vec_loss_x;
	}

	// 出力の損失関数(ベクタ指定)
	std::vector<double> GetOutputOnehotLoss(std::vector< std::vector<T> > y, INDEX offset = 0)
	{
		auto buf = m_layer->GetOutputSignalBuffer();
		INDEX frame_size = m_layer->GetOutputFrameSize();
		INDEX node_size = m_layer->GetOutputNodeSize();
		INDEX num_class = (INDEX)y[0].size();

		std::vector<double> vec_loss_x(frame_size);
		double* vec_loss = &vec_loss_x[0];

//		double	loss_0 = 0.1;
//		double	loss_1 = 1.0;
		double	loss_0 = 1.0 / num_class;
		double	loss_1 = 1.0 - loss_0;

#pragma omp parallel for
		for (int frame = 0; frame < (int)frame_size; ++frame) {
			vec_loss[frame] = 0;
			for (size_t node = 0; node < node_size; ++node) {
				if (y[frame / m_mux_size + offset][node % num_class] != 0) {
					vec_loss[frame] += (buf.template Get<BT>(frame, node) ? 0.0 : loss_1);
				}
				else {
					vec_loss[frame] += (buf.template Get<BT>(frame, node) ? loss_0 : 0.0);
				}
			}
		}

		return vec_loss_x;
	}

	// 出力の損失関数
	std::vector<double> CalcLoss(std::vector< std::vector<T> > y, INDEX offset = 0)
	{
		auto out_sig_buf = m_layer->GetOutputSignalBuffer();
		INDEX frame_size = m_layer->GetOutputFrameSize();
		INDEX node_size = m_layer->GetOutputNodeSize();
		INDEX num_class = (INDEX)y[0].size();

		size_t y_size = frame_size / m_mux_size;
		std::vector<double>	averages(y_size);
#pragma omp parallel for
		for (int i = 0; i < (int)y_size; ++i) {
			T sum = 0;
			for (int j = 0; j < num_class; ++j) {
				sum += (T)y[i + offset][j];
			}
			averages[i] = sum / (T)y[i].size();
		}
		
		std::vector<double> vec_loss_x(frame_size);
		double* vec_loss = &vec_loss_x[0];

#pragma omp parallel for
		for (int frame = 0; frame < (int)frame_size; ++frame) {
			vec_loss[frame] = 0;
			auto ave = averages[frame / m_mux_size];
			for (size_t node = 0; node < node_size; ++node) {
				auto sig = out_sig_buf.GetReal(frame, node);
				auto exp = y[frame / m_mux_size + offset][node % num_class];
				vec_loss[frame] += (double)(abs(sig - exp) * abs(exp - ave));
			}
		}

		return vec_loss_x;
	}


public:
	// Serialize
	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
	}

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
	}
	
	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		m_layer->Save(archive);
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		m_layer->Load(archive);
	}

};


}
