// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <vector>
#include <intrin.h>
#include <assert.h>
#include "NeuralNetGroup.h"
#include "NeuralNetLossFunction.h"
#include "NeuralNetAccuracyFunction.h"


namespace bb {


// NeuralNet 最上位構成用クラス
template <typename T = float, typename INDEX = size_t>
class NeuralNet : public NeuralNetGroup<T, INDEX>
{
protected:
	INDEX						m_batch_size = 0;

	NeuralNetBuffer<T, INDEX>	m_input_signal_buffers;
	NeuralNetBuffer<T, INDEX>	m_output_signal_buffers;
	NeuralNetBuffer<T, INDEX>	m_input_error_buffers;
	NeuralNetBuffer<T, INDEX>	m_output_error_buffers;
	
public:
	// コンストラクタ
	NeuralNet()
	{
	}
	
	// デストラクタ
	~NeuralNet() {
	}

	std::string GetClassName(void) const { return "NeuralNet"; }

	void SetBatchSize(INDEX batch_size)
	{
		// 親クラス呼び出し
		NeuralNetGroup<T, INDEX>::SetBatchSize(batch_size);

		// サイズ変更が無ければそのまま
		if (m_batch_size == batch_size) {
			return;
		}
		m_batch_size = batch_size;

		// 入出力のバッファも準備
		m_input_signal_buffers = m_firstLayer->CreateInputSignalBuffer();
		m_input_error_buffers = m_firstLayer->CreateInputErrorBuffer();
		m_output_signal_buffers = m_lastLayer->CreateOutputSignalBuffer();
		m_output_error_buffers = m_lastLayer->CreateOutputErrorBuffer();
		m_firstLayer->SetInputSignalBuffer(m_input_signal_buffers);
		m_firstLayer->SetInputErrorBuffer(m_input_error_buffers);
		m_lastLayer->SetOutputSignalBuffer(m_output_signal_buffers);
		m_lastLayer->SetOutputErrorBuffer(m_output_error_buffers);
	}

	void Forward(bool train = true, INDEX start_layer = 0)
	{
		INDEX layer_size = m_layers.size();

		for (INDEX layer = start_layer; layer < layer_size; ++layer) {
			m_layers[layer]->Forward(train);
		}
	}

	void Backward(void)
	{
		for (auto layer = m_layers.rbegin(); layer != m_layers.rend(); ++layer) {
			(*layer)->Backward();
		}
	}

	void Update(void)
	{
		for (auto layer = m_layers.begin(); layer != m_layers.end(); ++layer) {
			(*layer)->Update();
		}
	}


	// 入出力データへのアクセス補助
	void SetInputSignal(INDEX frame, INDEX node, T signal) {
		return m_firstLayer->GetInputSignalBuffer().SetReal(frame, node, signal);
	}

	void SetInputSignal(INDEX frame, std::vector<T> signals) {
		for (INDEX node = 0; node < (INDEX)signals.size(); ++node) {
			SetInputSignal(frame, node, signals[node]);
		}
	}

	T GetOutputSignal(INDEX frame, INDEX node) {
		return m_lastLayer->GetOutputSignalBuffer().GetReal(frame, node);
	}

	std::vector<T> GetOutputSignal(INDEX frame) {
		std::vector<T> signals(m_lastLayer->GetOutputNodeSize());
		for (INDEX node = 0; node < (INDEX)signals.size(); ++node) {
			signals[node] = GetOutputSignal(frame, node);
		}
		return signals;
	}

	void SetOutputError(INDEX frame, INDEX node, T error) {
		m_lastLayer->GetOutputErrorBuffer().SetReal(frame, node, error);
	}

	void SetOutputError(INDEX frame, std::vector<T> errors) {
		for (INDEX node = 0; node < (INDEX)errors.size(); ++node) {
			SetOutputError(frame, node, errors[node]);
		}
	}


public:
	double RunCalculation(
		const std::vector< std::vector<T> >& x,
		const std::vector< std::vector<T> >& t,
		INDEX max_batch_size,
		const NeuralNetAccuracyFunction<T, INDEX>* accFunc = nullptr,
		const NeuralNetLossFunction<T, INDEX>* lossFunc = nullptr,
		bool train = false,
		bool print_progress = false)
	{
		auto it_t = t.cbegin();

		INDEX x_size = (INDEX)x.size();
		double accuracy = 0;

		for (INDEX x_index = 0; x_index < x_size; x_index += max_batch_size) {
			// 末尾のバッチサイズクリップ
			INDEX batch_size = std::min(max_batch_size, x.size() - x_index);
			INDEX node_size = x[0].size();

			// バッチサイズ設定
			SetBatchSize(batch_size);

			auto in_sig_buf = GetInputSignalBuffer();
			auto out_sig_buf = GetOutputSignalBuffer();

			// データ格納
			for (INDEX frame = 0; frame < batch_size; ++frame) {
				for (INDEX node = 0; node < node_size; ++node) {
					in_sig_buf.Set<T>(frame, node, x[x_index + frame][node]);
				}
			}

			// 予測
			Forward(train);

			// 進捗表示
			if (print_progress) {
				INDEX progress = x_index + batch_size;
				INDEX rate = progress * 100 / x_size;
				std::cout << "[" << rate << "% (" << progress << "/" << x_size << ")]";
			}

			// 誤差逆伝播
			if (lossFunc != nullptr) {
				auto out_err_buf = GetOutputErrorBuffer();
				auto loss = lossFunc->CalculateLoss(out_sig_buf, out_err_buf, it_t);

				// 進捗表示
				if (print_progress) {
					std::cout << "  loss : " << loss;
				}
			}

			if (accFunc != nullptr) {
				accuracy += accFunc->CalculateAccuracy(out_sig_buf, it_t);

				// 進捗表示
				if (print_progress) {
					std::cout << "  acc : " << accuracy / (x_index + batch_size);
				}
			}

			if ( train ) {
				// 逆伝播
				Backward();

				// 更新
				Update();
			}
			
			// 進捗表示
			if (print_progress) {
				std::cout << "\r" << std::flush;
			}

			// イテレータを進める
			it_t += batch_size;
		}

		// 進捗表示クリア
		if (print_progress) {
			std::cout << "                                                                    \r" << std::flush;
		}

		return accuracy / x_size;
	}

};


}