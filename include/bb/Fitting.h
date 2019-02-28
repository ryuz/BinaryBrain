// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <assert.h>

#include "bb/Layer.h"
#include "bb/LossFunction.h"
#include "bb/AccuracyFunction.h"
#include "bb/Optimizer.h"
#include "bb/Utility.h"
#include "bb/TrainData.h"


namespace bb {


template <typename T>
class Fitting
{
protected:
    std::mt19937_64                        m_mt;

//	std::shared_ptr<AccuracyFunction>      m_accFunc;
//	std::shared_ptr<LossFunction>          m_lossFunc;
    
public:

    struct run_calculation_t
    {
	    std::vector< std::vector<T> > const     &x;
	    std::vector< std::vector<T> > const     &t;
	    index_t                                 max_batch_size;
	    index_t                                 min_batch_size;
	    std::shared_ptr<AccuracyFunction>       accFunc;
	    std::shared_ptr<LossFunction>           lossFunc;
	    bool                                    train = false;
	    bool                                    print_progress = false;
    };
    
    double RunCalculation(
        indices_t                           x_shape,
        std::vector< std::vector<T> > const &x_vec,
        indices_t                           t_shape,
		std::vector< std::vector<T> > const &t_vec,
        index_t max_batch_size,
		index_t min_batch_size = 1,
	    std::shared_ptr<AccuracyFunction> accFunc = nullptr,
	    std::shared_ptr<LossFunction>     lossFunc = nullptr,
		std::shared_ptr<Optimizer>        optimizer = nullptr,
		bool train = false,
		bool print_progress = false)
    {
        BB_ASSERT(x.size() == y.size());

        if ( accFunc  != nullptr ) { accFunc.Clear(); }
        if ( lossFunc != nullptr ) { lossFunc.Clear(); }
        
        index_t frame_size = (index_t)x.size();
        
        FrameBuffer x_buf;
        FrameBuffer t_buf;

        bb::index_t index = 0;
        while ( index < frame_size )
        {
            // ミニバッチサイズ計算
            bb::index_t  mini_batch_size = std::min(max_batch_size, frame_size - index);

            // 残数が規定以下なら抜ける
            if ( mini_batch_size < min_batch_size ) {
                break;
            }

            // 学習データセット
            x_buf.Resize(DataType<T>::type, mini_batch_size, x_shape);
            x_buf.SetVector(x_vec, index);

            // Forward
            auto y_buf = net->Forward(x_buf, train);

            // 期待値データセット
            t_buf.Resize(DataType<T>::type, mini_batch_size, t_shape);
            t.SetVector(t_vec, index);

			// 進捗表示
			if ( print_progress ) {
				index_t progress = index + mini_batch_size;
				index_t rate = progress * 100 / frame_size;
				std::cout << "[" << rate << "% (" << progress << "/" << frame_size << ")]";
			}
            
            FrameBuffer dy_buf;
            if ( lossFunc != nullptr ) {
                dy_buf = lossFunc->CalculateLoss(y_buf, t_buf);
            }

            if ( accFunc != nullptr ) {
                accFunc.CalculateAccuracy(y_buf, t_buf);
            }

            if ( train && lossFunc != nullptr ) {
                auto dx = net->Backward(dy_buf);
                
                if ( optimizer != nullptr ) {
                    optimizer->Update();
                }
            }

            // 進捗表示
		    if ( print_progress ) {
                if ( lossFunc != nullptr ) {
	    		    std::cout << "  loss : " << lossFunc->GetLoss();
                }

                if ( accFunc != nullptr ) {
                    std::cout << "  acc : " << accFunc->GetAccuracy();
                }

				std::cout << "\r" << std::flush;
			}

            // インデックスを進める
            index += min_batch_size;
        }
    }


    double RunCalculation(
		const   std::vector< std::vector<T> >& x,
		const   std::vector< std::vector<T> >& y,
		index_t max_batch_size,
		index_t min_batch_size,
		const NeuralNetAccuracyFunction<T>* accFunc = nullptr,
		const NeuralNetLossFunction<T>* lossFunc = nullptr,
		bool train = false,
		bool print_progress = false)
	{
		auto it_y = y.cbegin();

		INDEX x_size = (INDEX)x.size();
		double accuracy = 0;

		for (INDEX x_index = 0; x_index < x_size; x_index += max_batch_size) {
			// 末尾のバッチサイズクリップ
			INDEX batch_size = std::min(max_batch_size, (INDEX)x.size() - x_index);
			if (batch_size < min_batch_size) { break; }

			INDEX node_size = x[0].size();

			// バッチサイズ設定
			SetBatchSize(batch_size);

			auto in_sig_buf = this->GetInputSignalBuffer();
			auto out_sig_buf = this->GetOutputSignalBuffer();

			// データ格納
			for (INDEX frame = 0; frame < batch_size; ++frame) {
				for (INDEX node = 0; node < node_size; ++node) {
					in_sig_buf.template Set<T>(frame, node, x[x_index + frame][node]);
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
				auto out_err_buf = this->GetOutputErrorBuffer();
				auto loss = lossFunc->CalculateLoss(out_sig_buf, out_err_buf, it_y);

				// 進捗表示
				if (print_progress) {
					std::cout << "  loss : " << loss;
				}
			}

			if (accFunc != nullptr) {
				accuracy += accFunc->CalculateAccuracy(out_sig_buf, it_y);

				// 進捗表示
				if (print_progress) {
					std::cout << "  acc : " << accuracy / (x_index + batch_size);
				}
			}

			if (train) {
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
			it_y += batch_size;
		}

		// 進捗表示クリア
		if (print_progress) {
			std::cout << "                                                                    \r" << std::flush;
		}

		return accuracy / x_size;
	}
	
	void Fitting(
		std::string name,
		std::vector< std::vector<T> >& x_train,
		std::vector< std::vector<T> >& y_train,
		std::vector< std::vector<T> >& x_test,
		std::vector< std::vector<T> >& y_test,
		INDEX epoch_size,
		INDEX max_batch_size,
		const NeuralNetAccuracyFunction<T>* accFunc,
		const NeuralNetLossFunction<T>* lossFunc,
		bool print_progress = true,
		bool file_write = true,
		bool over_write = false,
		bool initial_evaluation = false,
		std::uint64_t seed=1,
		void (*callback)(NeuralNet<T>* net, void* user) = 0,
		void* user = 0)
	{
		std::string csv_file_name = name + "_acc.txt";
		std::string log_file_name = name + "_log.txt";
		std::string net_file_name = name + "_net.json";
		std::mt19937_64 mt(seed);
		
		// ログファイルオープン
		std::ofstream ofs_log;
		if (file_write) {
			ofs_log.open(log_file_name, over_write ? std::ios::out : std::ios::app);
		}

		{
			// ログ出力先設定
			ostream_tee log_stream;
			log_stream.add(std::cout);
			if (ofs_log.is_open()) { log_stream.add(ofs_log); }
			m_log_stream = &log_stream;

			// 以前の計算があれば読み込み
			int prev_epoch = 0;
			if (file_write && !over_write) {
				std::ifstream ifs(net_file_name);
				if (ifs.is_open()) {
					cereal::JSONInputArchive ar(ifs);
					ar(cereal::make_nvp("epoch", prev_epoch));
					this->Load(ar);
					log_stream << "[load] " << net_file_name << std::endl;
				}
			}

			// 開始メッセージ
			log_stream << "fitting start : " << name << std::endl;

			// 初期評価
			if (initial_evaluation) {
				auto test_accuracy  = RunCalculation(x_test,  y_test,  max_batch_size, 0, accFunc);
				auto train_accuracy = RunCalculation(x_train, y_train, max_batch_size, 0, accFunc);
				log_stream << "[initial] "
					<< "test_accuracy : " << std::setw(6) << std::fixed << std::setprecision(4) << test_accuracy << " "
					<< "train_accuracy : " << std::setw(6) << std::fixed << std::setprecision(4) << train_accuracy << std::endl;
			}

			// 開始時間記録
			auto start_time = std::chrono::system_clock::now();

			for (int epoch = 0; epoch < epoch_size; ++epoch) {
				// 学習実施
				auto train_accuracy = RunCalculation(x_train, y_train, max_batch_size, max_batch_size, accFunc, lossFunc, true, print_progress);

				// ネット保存
				if (file_write) {
					int save_epoc = epoch + 1 + prev_epoch;

					if(1){
						std::stringstream fname;
						fname << name << "_net_" << save_epoc << ".json";
						std::ofstream ofs_net(fname.str());
						cereal::JSONOutputArchive ar(ofs_net);
						ar(cereal::make_nvp("epoch", save_epoc));
						this->Save(ar);
						std::cout << "[save] " << fname.str() << std::endl;
			//			log_streamt << "[save] " << fname.str() << std::endl;
					}

					{
						std::ofstream ofs_net(net_file_name);
						cereal::JSONOutputArchive ar(ofs_net);
						ar(cereal::make_nvp("epoch", save_epoc));
						this->Save(ar);
			//			log_stream << "[save] " << net_file_name << std::endl;
					}
				}

				// 学習状況評価
				double now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
				auto test_accuracy = RunCalculation(x_test, y_test, max_batch_size, 0, accFunc);
				log_stream	<< std::setw(10) << std::fixed << std::setprecision(2) << now_time << "s "
							<< "epoch[" << std::setw(3) << epoch + 1 + prev_epoch << "] "
							<< "test_accuracy : "  << std::setw(6) << std::fixed << std::setprecision(4) << test_accuracy  << " "
							<< "train_accuracy : " << std::setw(6) << std::fixed << std::setprecision(4) << train_accuracy << std::endl;

				// callback
				if (callback) {
					callback(this, user);
				}

				// Shuffle
				ShuffleDataSet(mt(), x_train, y_train);
			}

			// 終了メッセージ
			log_stream << "fitting end\n" << std::endl;
		}
	}

	void Fitting(
		std::string name,
		TrainData<T> td,
		INDEX epoch_size,
		INDEX mini_batch_size,
		const NeuralNetAccuracyFunction<T>* accFunc,
		const NeuralNetLossFunction<T>* lossFunc,
		bool print_progress = true,
		bool file_write = true,
		bool over_write = false,
		bool initial_evaluation = true,
		std::uint64_t seed = 1,
		void (*callback)(NeuralNet<T>* net, void* user) = 0,
		void* user = 0
		)
	{
		Fitting(
			name,
			td.x_train,
			td.y_train,
			td.x_test,
			td.y_test,
			epoch_size,
			mini_batch_size,
			accFunc,
			lossFunc,
			print_progress,
			file_write,
			over_write,
			initial_evaluation,
			seed,
			callback,
			user);
	}

};


}