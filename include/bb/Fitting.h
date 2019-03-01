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


namespace bb {


template <typename T>
class Fitting
{
protected:
    using callback_proc_t = void (*)(std::shared_ptr< Layer >, void*);

    std::string                         m_name;
    std::shared_ptr<Layer>              m_net;

  	std::mt19937_64                     m_mt;

	index_t                             m_epoch = 0;
	index_t                             m_max_batch_size = 16;

	std::shared_ptr<AccuracyFunction>   m_accFunc;
	std::shared_ptr<LossFunction>       m_lossFunc;
	std::shared_ptr<Optimizer>          m_optimizer;

	bool                                m_print_progress     = true;
	bool                                m_file_write         = true;
	bool                                m_over_write         = false;
	bool                                m_initial_evaluation = false;
	
    callback_proc_t                     m_callback_proc = nullptr;
	void                                *m_callback_user = 0;
    
public:
    // コンストラクタ
    Fitting(std::shared_ptr< Layer > net, std::string name="")
    {
        m_net  = net;
        m_name = name;
        if ( m_name.empty() ) {
            m_name = m_net->GetLayerName();
        }
    }

    void SetSeed(std::int64_t seed)     { m_mt.seed(seed); }
    void SetMaxBatchSize(index_t batch) { m_max_batch_size = batch; }

    void SetAccuracyFunction(std::shared_ptr<AccuracyFunction> accFunc) { m_accFunc = accFunc; }
    void SetLossFunction(std::shared_ptr<LossFunction > lossFunc)       { m_lossFunc = lossFunc; }
    void SetOptimizer(std::shared_ptr<Optimizer > optimizer)            { m_optimizer = optimizer; }

    void SetPrintProgress(bool print_progress) { m_print_progress = print_progress; }
    void SetFileWrite(bool file_write) { m_file_write = file_write; }
    void SetOverWrite(bool over_write) { m_over_write = over_write; }
    void SetInitialEvaluation(bool initial_evaluation) { m_initial_evaluation = false; }

    void SetCallback(callback_proc_t callback_proc, void *user)
    {
        m_callback_proc = callback_proc;
	    m_callback_user = user;
    }
    

	void Run(
            TrainData<T> &td,
		    index_t      epoch_size,
		    index_t      batch_size
        )
    {
		std::string csv_file_name = m_name + "_acc.txt";
		std::string log_file_name = m_name + "_log.txt";
		std::string net_file_name = m_name + "_net.json";
		
		// ログファイルオープン
		std::ofstream ofs_log;
		if ( m_file_write ) {
			ofs_log.open(log_file_name, m_over_write ? std::ios::out : std::ios::app);
		}

		{
			// ログ出力先設定
			ostream_tee log_stream;
			log_stream.add(std::cout);
			if (ofs_log.is_open()) { log_stream.add(ofs_log); }

			// 以前の計算があれば読み込み
			int prev_epoch = 0;
            /*
			if (file_write && !over_write) {
				std::ifstream ifs(net_file_name);
				if (ifs.is_open()) {
					cereal::JSONInputArchive ar(ifs);
					ar(cereal::make_nvp("epoch", prev_epoch));
					this->Load(ar);
					log_stream << "[load] " << net_file_name << std::endl;
				}
			}
            */

			// 開始メッセージ
			log_stream << "fitting start : " << m_name << std::endl;

			// 初期評価
			if (m_initial_evaluation) {
				auto test_accuracy  = Calculation(td.x_test,  td.x_shape, td.t_test,  td.t_shape, batch_size, 0, m_accFunc);
				auto train_accuracy = Calculation(td.x_train, td.x_shape, td.t_train, td.t_shape, batch_size, 0, m_accFunc);
				log_stream << "[initial] "
					<< "test_accuracy : " << std::setw(6) << std::fixed << std::setprecision(4) << test_accuracy << " "
					<< "train_accuracy : " << std::setw(6) << std::fixed << std::setprecision(4) << train_accuracy << std::endl;
			}

			// 開始時間記録
			auto start_time = std::chrono::system_clock::now();

			for (int epoch = 0; epoch < epoch_size; ++epoch) {
				// 学習実施
				auto train_accuracy = Calculation(td.x_train, td.x_shape, td.t_train, td.t_shape, batch_size, batch_size,
                                        m_accFunc, m_lossFunc, m_optimizer, true, m_print_progress);

				// ネット保存
#if 0
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
#endif

				// 学習状況評価
				double now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
				auto test_accuracy = Calculation(td.x_test,  td.x_shape, td.t_test,  td.t_shape, batch_size, 0, m_accFunc);
				log_stream	<< std::setw(10) << std::fixed << std::setprecision(2) << now_time << "s "
							<< "epoch[" << std::setw(3) << epoch + 1 + prev_epoch << "] "
							<< "test_accuracy : "  << std::setw(6) << std::fixed << std::setprecision(4) << test_accuracy  << " "
							<< "train_accuracy : " << std::setw(6) << std::fixed << std::setprecision(4) << train_accuracy << std::endl;

				// callback
				if (m_callback_proc != nullptr) {
					m_callback_proc(m_net, m_callback_user);
				}

				// Shuffle
				ShuffleDataSet(m_mt(), td.x_train, td.t_train);
			}

			// 終了メッセージ
			log_stream << "fitting end\n" << std::endl;
		}
	}


protected:
    double Calculation(
                std::vector< std::vector<T> > const &x,
                indices_t x_shape,
                std::vector< std::vector<T> > const &t,
                indices_t t_shape,
		        index_t max_batch_size,
		        index_t min_batch_size,
	            std::shared_ptr< AccuracyFunction > accFunc = nullptr,
	            std::shared_ptr< LossFunction > lossFunc = nullptr,	
                std::shared_ptr< Optimizer > optimizer = nullptr,
		        bool train = false,
		        bool print_progress = false)

    {
        BB_ASSERT(x.size() == t.size());

        if ( accFunc  != nullptr ) { accFunc->Clear(); }
        if ( lossFunc != nullptr ) { lossFunc->Clear(); }
        
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
            x_buf.SetVector(x, index);

            // Forward
            auto y_buf = m_net->Forward(x_buf, train);

            // 期待値データセット
            t_buf.Resize(DataType<T>::type, mini_batch_size, t_shape);
            t_buf.SetVector(t, index);

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
                accFunc->CalculateAccuracy(y_buf, t_buf);
            }

            if ( train && lossFunc != nullptr ) {
                auto dx = m_net->Backward(dy_buf);
                
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
            index += mini_batch_size;
        }

        return accFunc->GetAccuracy();
    }

};


}