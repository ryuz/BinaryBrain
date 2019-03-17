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

#include "bb/Model.h"
#include "bb/LossFunction.h"
#include "bb/AccuracyFunction.h"
#include "bb/Optimizer.h"
#include "bb/Utility.h"


namespace bb {


// 実行アシストクラス
template <typename T>
class Runner
{
protected:
    using callback_proc_t = void (*)(std::shared_ptr< Model >, void*);

    std::string                         m_name;
    std::shared_ptr<Model>              m_net;

  	std::mt19937_64                     m_mt;

	index_t                             m_epoch = 0;
	index_t                             m_max_batch_size = 16;

	std::shared_ptr<AccuracyFunction>   m_accFunc;
	std::shared_ptr<LossFunction>       m_lossFunc;
	std::shared_ptr<Optimizer>          m_optimizer;

	bool                                m_print_progress     = true;
    bool                                m_file_read          = false;
    bool                                m_file_write         = false;
    bool                                m_write_serial       = false;
	bool                                m_initial_evaluation = false;
	
    callback_proc_t                     m_callback_proc = nullptr;
	void                                *m_callback_user = 0;
    
protected:
    // コンストラクタ
    Runner() {}
    

public:
    struct create_t
    {
        std::string                         name;
        std::shared_ptr<Model>              net;
	    std::shared_ptr<AccuracyFunction>   accFunc;
	    std::shared_ptr<LossFunction>       lossFunc;
	    std::shared_ptr<Optimizer>          optimizer;
	    bool                                print_progress = true;
        bool                                file_read = false;
        bool                                file_write = false;
        bool                                write_serial = false;
	    bool                                initial_evaluation = false;
        std::int64_t                        seed = 1;
	    callback_proc_t                     callback_proc = nullptr;
	    void*                               callback_user = 0;
    };

    static std::shared_ptr<Runner> Create(create_t const &create)
    {
        auto self = std::shared_ptr<Runner>(new Runner);

        BB_ASSERT(create.net != nullptr);

        self->m_name               = create.name;
        self->m_net                = create.net;
	    self->m_accFunc            = create.accFunc;
	    self->m_lossFunc           = create.lossFunc;
	    self->m_optimizer          = create.optimizer;
	    self->m_print_progress     = create.print_progress;
        self->m_file_read          = create.file_read;
        self->m_file_write         = create.file_write;
        self->m_write_serial       = create.write_serial;
	    self->m_initial_evaluation = create.initial_evaluation;
	    self->m_callback_proc      = create.callback_proc;
	    self->m_callback_user      = create.callback_user;
        
        self->m_mt.seed(create.seed);

        if ( self->m_name.empty() ) {
            self->m_name = self->m_net->GetName();
        }

        return self;
    }
  
    static std::shared_ptr<Runner> Create(
                std::string                         name,
                std::shared_ptr<Model>              net,
	            index_t                             epoch_size,
	            index_t                             batch_size,
	            std::shared_ptr<AccuracyFunction>   accFunc,
	            std::shared_ptr<LossFunction>       lossFunc,
	            std::shared_ptr<Optimizer>          optimizer,
	            bool                                print_progress = false,
                bool                                file_read  = false,
                bool                                file_write = false,
                bool                                write_serial = false,
	            bool                                initial_evaluation = false,
                std::int64_t                        seed = 1,
	            callback_proc_t                     callback_proc = nullptr,
	            void*                               callback_user = 0
        )
    {
        create_t create;

        create.name               = name;
        create.net                = net;
        create.accFunc            = accFunc;
        create.lossFunc           = lossFunc;
        create.optimizer          = optimizer;
        create.print_progress     = print_progress;
        create.file_read          = file_read;
        create.file_write         = file_write;
        create.write_serial       = write_serial;
        create.initial_evaluation = initial_evaluation;
        create.seed               = seed;
        create.callback_proc      = callback_proc;
        create.callback_user      = callback_user;

        return Create(create);
    }


    // アクセサ
    void        SetName(std::string name) { m_name = name; }
    std::string GetName(void) const { return m_name; }

    void SetSeed(std::int64_t seed)     { m_mt.seed(seed); }

    void    SetMaxBatchSize(index_t batch) { m_max_batch_size = batch; }
    index_t GetMaxBatchSize(void) const  { return m_max_batch_size; }

    void                              SetAccuracyFunction(std::shared_ptr<AccuracyFunction> accFunc) { m_accFunc = accFunc; }
    std::shared_ptr<AccuracyFunction> GetAccuracyFunction(void) const { return m_accFunc; }

    void SetLossFunction(std::shared_ptr<LossFunction > lossFunc)       { m_lossFunc = lossFunc; }
    void SetOptimizer(std::shared_ptr<Optimizer > optimizer)            { m_optimizer = optimizer; }

    void SetPrintProgress(bool print_progress) { m_print_progress = print_progress; }
    void SetFileRead(bool file_read) { m_file_read = file_read; }
    void SetFileWrite(bool file_write) { m_file_write = file_write; }
    void SetInitialEvaluation(bool initial_evaluation) { m_initial_evaluation = false; }

    void SetCallback(callback_proc_t callback_proc, void *user)
    {
        m_callback_proc = callback_proc;
	    m_callback_user = user;
    }
    

    // Serialize
  	void Save(std::ostream &os) const
	{
		/*
        Save(os, m_name);
        SaveIndex(os, m_epoch);
        SaveIndex(os, m_max_batch_size);
	    Save(os, m_print_progress);
	    Save(os, m_file_write);
	    Save(os, m_over_write);
	    Save(os, m_initial_evaluation);
        m_net->Save(os);
		*/
	}

	void Load(std::istream &is)
	{
		/*
        Load(is, m_name);
        m_epoch = LoadIndex(is);
        m_max_batch_size = LoadIndex(is);
	    Load(is, m_print_progress);
	    Load(is, m_file_write);
	    Load(is, m_over_write);
	    Load(is, m_initial_evaluation);
        m_net->Load(is);
		*/
	}

   	void SaveBinary(std::string filename) const
	{
		std::ofstream ofs(filename, std::ios::binary);
		Save(ofs);
	}

   	void LoadBinary(std::string filename)
	{
		std::ifstream ifs(filename, std::ios::binary);
		Load(ifs);
	}
    

#ifdef BB_WITH_CEREAL
	template <class Archive>
	void save(Archive& archive, std::uint32_t const version) const
	{
		archive(cereal::make_nvp("name", m_name));
		archive(cereal::make_nvp("epoch", m_epoch));
//		archive(cereal::make_nvp("max_batch_size", m_max_batch_size));
//		archive(cereal::make_nvp("print_progress", m_print_progress));
//		archive(cereal::make_nvp("file_write", m_file_write));
        m_net->Save(archive);
	}

	template <class Archive>
	void load(Archive& archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("name", m_name));
		archive(cereal::make_nvp("epoch", m_epoch));
//		archive(cereal::make_nvp("max_batch_size", m_max_batch_size));
//		archive(cereal::make_nvp("print_progress", m_print_progress));
//		archive(cereal::make_nvp("file_write", m_file_write));
        m_net->Load(archive);
	}

   	void SaveJson(std::string filename) const
    {
        std::ofstream ofs(filename);
        cereal::JSONOutputArchive archive(ofs);
		archive(cereal::make_nvp("runner", *this));
	}

   	void SaveJson(std::ostream &os) const
    {
        cereal::JSONOutputArchive archive(os);
		archive(cereal::make_nvp("runner", *this));
	}

	void LoadJson(std::string filename)
    {
        std::ifstream ifs(filename);
        cereal::JSONInputArchive archive(ifs);
		archive(cereal::make_nvp("runner", *this));
	}

	void LoadJson(std::istream &is)
    {
        cereal::JSONInputArchive archive(is);
		archive(cereal::make_nvp("runner", *this));
	}
#endif


	void Fitting(
            TrainData<T> &td,
		    index_t      epoch_size,
		    index_t      batch_size
        )
    {
		std::string csv_file_name = m_name + "_acc.txt";
		std::string log_file_name = m_name + "_log.txt";
#ifdef BB_WITH_CEREAL
		std::string net_file_name = m_name + "_net.json";
#else
		std::string net_file_name = m_name + "_net.bin";
#endif

		// ログファイルオープン
		std::ofstream ofs_log;
		if ( m_file_write ) {
			ofs_log.open(log_file_name, m_file_read ? std::ios::app : std::ios::out);
		}

		{
			// ログ出力先設定
			ostream_tee log_stream;
			log_stream.add(std::cout);
			if (ofs_log.is_open()) { log_stream.add(ofs_log); }

			int prev_epoch = 0;
            
            // 以前の計算があれば読み込み
            if ( m_file_read ) {
#ifdef BB_WITH_CEREAL
                std::ifstream ifs(net_file_name);
				if (ifs.is_open()) {
                    LoadJson(ifs);
                    std::cout << "[load] " << net_file_name << std::endl;
				}
#else
                std::ifstream ifs(net_file_name, std::ios::binary);
				if (ifs.is_open()) {
                    Load(ifs);
                    std::cout << "[load] " << net_file_name << std::endl;
				}
#endif
            }
            

			// 開始メッセージ
			log_stream << "fitting start : " << m_name << std::endl;

            // オプティマイザ設定
            m_optimizer->SetVariables(m_net->GetParameters(), m_net->GetGradients());

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
				if (m_file_write) {
					int save_epoc = epoch + 1 + prev_epoch;

#ifdef BB_WITH_CEREAL
					if ( m_write_serial ) {
						std::stringstream fname;
						fname << m_name << "_net_" << save_epoc << ".json";
						SaveJson(fname.str());
						std::cout << "[save] " << fname.str() << std::endl;
			//			log_streamt << "[save] " << fname.str() << std::endl;
					}

					{
						SaveJson(net_file_name);
			//			log_stream << "[save] " << net_file_name << std::endl;
					}
#else
					if ( m_write_serial ) {
						std::stringstream fname;
						fname << m_name << "_net_" << save_epoc << ".bin";
						SaveBinary(fname.str());
						std::cout << "[save] " << fname.str() << std::endl;
			//			log_streamt << "[save] " << fname.str() << std::endl;
					}

					{
						SaveBinary(net_file_name);
			//			log_stream << "[save] " << net_file_name << std::endl;
					}
#endif
                }

				// 学習状況評価
                {
				    double now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
				    auto test_accuracy  = Calculation(td.x_test,  td.x_shape, td.t_test,  td.t_shape, batch_size, 0, m_accFunc);
				    auto train_accuracy = Calculation(td.x_train, td.x_shape, td.t_train, td.t_shape, batch_size, 0, m_accFunc);
				    log_stream	<< std::setw(10) << std::fixed << std::setprecision(2) << now_time << "s "
							    << "epoch[" << std::setw(3) << epoch + 1 + prev_epoch << "] "
							    << "test_accuracy : "  << std::setw(6) << std::fixed << std::setprecision(4) << test_accuracy  << " "
							    << "train_accuracy : " << std::setw(6) << std::fixed << std::setprecision(4) << train_accuracy << std::endl;
                }

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


	double Evaluation(
            TrainData<T> &td,
		    index_t      batch_size
        )
    {
        return Calculation(td.x_test,  td.x_shape, td.t_test,  td.t_shape, batch_size, 0, m_accFunc);
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

        if ( accFunc  != nullptr ) {
            accFunc->Clear();
        }
        if ( lossFunc != nullptr ) {
            lossFunc->Clear();
        }
        
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

        std::cout << "                                                                               \r" << std::flush;

        return accFunc->GetAccuracy();
    }

};


}