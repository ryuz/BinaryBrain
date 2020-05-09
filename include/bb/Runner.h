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
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <string>

#include "bb/Model.h"
#include "bb/LossFunction.h"
#include "bb/MetricsFunction.h"
#include "bb/Optimizer.h"
#include "bb/Utility.h"


namespace bb {



// Pythonからの状態保存用にここだけ機能を切り出す
struct RunStatus
{
    std::string             name;
    index_t                 epoch = 0;
    std::shared_ptr<Model>  net;

#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("name", name));
        archive(cereal::make_nvp("epoch", epoch));
        net->Save(archive);
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("name", name));
        archive(cereal::make_nvp("epoch", epoch));
        net->Load(archive);
    }

    bool SaveJson(std::ostream &os) const
    {
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp("runner", *this));
        return true;
    }

    bool SaveJson(std::string filename)
    {
        std::ofstream ofs(filename);
        if ( !ofs.is_open() ) {
            return false;
        }
        return SaveJson(ofs);
    }
    

    bool LoadJson(std::istream &is)
    {
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp("runner", *this));
        return true;
    }

    bool LoadJson(std::string filename)
    {
        std::ifstream ifs(filename);
        if ( !ifs.is_open() ) {
            return false;
        }
        return LoadJson(ifs);
    }
    
    static bool WriteJson(std::string filename, std::shared_ptr<Model> net, std::string name, index_t epoch)
    {
        RunStatus rs;
        rs.name  = name;
        rs.epoch = epoch;
        rs.net   = net;
        return rs.SaveJson(filename);
    }

    static bool ReadJson(std::string filename, std::shared_ptr<Model> net, std::string &name, index_t &epoch)
    {
        RunStatus rs;
        rs.net   = net;
        if ( !rs.LoadJson(filename) ) {
            return false;
        }
        name  = rs.name;
        epoch = rs.epoch;
        return true;
    }
#endif
};




// 実行アシストクラス
template <typename T>
class Runner
{
protected:
    using callback_proc_t          = void (*)(std::shared_ptr< Model >, void*);
    using data_augmentation_proc_t = void (*)(TrainData<T>&, std::uint64_t, void*);

    std::string                         m_name;
    std::shared_ptr<Model>              m_net;

    std::mt19937_64                     m_mt;

    index_t                             m_epoch = 0;
    index_t                             m_max_run_size = 0;

    std::shared_ptr<MetricsFunction>    m_metricsFunc;
    std::shared_ptr<LossFunction>       m_lossFunc;
    std::shared_ptr<Optimizer>          m_optimizer;

    bool                                m_print_progress          = true;
    bool                                m_print_progress_loss     = true;     //< 途中経過で損失を表示するか
    bool                                m_print_progress_accuracy = true;     //< 途中経過で精度を表示するか
    bool                                m_log_write               = true;     //< ログを書き込むか
    bool                                m_log_append              = true;     //< ログを追記モードにするか
    bool                                m_file_read               = false;
    bool                                m_file_write              = false;
    bool                                m_write_serial            = false;
    bool                                m_initial_evaluation      = false;
    
    callback_proc_t                     m_callback_proc = nullptr;
    void                                *m_callback_user = 0;

    data_augmentation_proc_t            m_data_augmentation_proc = nullptr;
    void                                *m_data_augmentation_user = 0;
    
public:
    struct create_t
    {
        std::string                         name;                               //< ネット名
        std::shared_ptr<Model>              net;                                //< ネット
        std::shared_ptr<LossFunction>       lossFunc;                           //< 損失関数オブジェクト
        std::shared_ptr<MetricsFunction>    metricsFunc;                        //< 評価関数オブジェクト
        std::shared_ptr<Optimizer>          optimizer;                          //< オプティマイザ
        index_t                             max_run_size = 0;                   //< 最大実行バッチ数
        bool                                print_progress = true;              //< 途中経過を表示するか
        bool                                print_progress_loss = true;         //< 途中経過で損失を表示するか
        bool                                print_progress_accuracy = true;     //< 途中経過で精度を表示するか
        bool                                log_write               = true;     //< ログを書き込むか
        bool                                log_append              = true;     //< ログを追記モードにするか
        bool                                file_read = false;                  //< 以前の計算があれば読み込むか
        bool                                file_write = false;                 //< 計算結果を保存するか
        bool                                write_serial = false;               //< EPOC単位で計算結果を連番で保存するか
        bool                                initial_evaluation = false;         //< 初期評価を行うか
        std::int64_t                        seed = 1;                           //< 乱数初期値
        callback_proc_t                     callback_proc = nullptr;            //< コールバック関数
        void*                               callback_user = 0;                  //< コールバック関数のユーザーパラメータ
        data_augmentation_proc_t            data_augmentation_proc = nullptr;   //< Data Augmentation用処理挿入
        void*                               data_augmentation_user = 0;         //< コールバック関数のユーザーパラメータ
    };

protected:
    // コンストラクタ
    Runner(create_t const &create)
    {
        BB_ASSERT(create.net != nullptr);

        m_name                    = create.name;
        m_net                     = create.net;
        m_metricsFunc             = create.metricsFunc;
        m_lossFunc                = create.lossFunc;
        m_optimizer               = create.optimizer;
        m_max_run_size            = create.max_run_size;
        m_print_progress          = create.print_progress;
        m_print_progress_loss     = create.print_progress_loss;
        m_print_progress_accuracy = create.print_progress_accuracy;
        m_log_write               = create.log_write;
        m_log_append              = create.log_append;
        m_file_read               = create.file_read;
        m_file_write              = create.file_write;
        m_write_serial            = create.write_serial;
        m_initial_evaluation      = create.initial_evaluation;
        m_callback_proc           = create.callback_proc;
        m_callback_user           = create.callback_user;
        m_data_augmentation_proc  = create.data_augmentation_proc;
        m_data_augmentation_user  = create.data_augmentation_user;
        
        m_mt.seed(create.seed);

        if ( m_name.empty() ) {
            m_name = m_net->GetName();
        }
    }
    

public:
    ~Runner() {}

    static std::shared_ptr<Runner> Create(create_t const &create)
    {
        return std::shared_ptr<Runner>(new Runner(create));
    }
  
    static std::shared_ptr<Runner> Create(
                std::string                         name,
                std::shared_ptr<Model>              net,
                index_t                             epoch_size,
                index_t                             batch_size,
                std::shared_ptr<MetricsFunction>    metricsFunc,
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
        create.metricsFunc        = metricsFunc;
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


    static std::shared_ptr<Runner> CreateEx
            (
                std::string                         name,
                std::shared_ptr<Model>              net,
                std::shared_ptr<LossFunction>       lossFunc,
                std::shared_ptr<MetricsFunction>    metricsFunc,
                std::shared_ptr<Optimizer>          optimizer,
                index_t                             max_run_size = 0,
                bool                                print_progress = true,
                bool                                print_progress_loss = true,
                bool                                print_progress_accuracy = true,
                bool                                log_write = true,
                bool                                log_append = true,
                bool                                file_read = false,
                bool                                file_write = false,
                bool                                write_serial = false,
                bool                                initial_evaluation = false,
                std::int64_t                        seed = 1
            )
    {
        create_t create;
        create.name                    = name;
        create.net                     = net;
        create.lossFunc                = lossFunc;
        create.metricsFunc             = metricsFunc;
        create.optimizer               = optimizer;
        create.max_run_size            = max_run_size;
        create.print_progress          = print_progress;
        create.print_progress_loss     = print_progress_loss;
        create.print_progress_accuracy = print_progress_accuracy;
        create.log_write               = log_write;
        create.log_append              = log_append;
        create.file_read               = file_read;
        create.file_write              = file_write;
        create.write_serial            = write_serial;
        create.initial_evaluation      = initial_evaluation;
        create.seed                    = seed;
        return Create(create);
    }


    // アクセサ
    void        SetName(std::string name) { m_name = name; }
    std::string GetName(void) const { return m_name; }

    void SetSeed(std::int64_t seed)     { m_mt.seed(seed); }

    void                                SetMetricsFunction(std::shared_ptr<MetricsFunction> metricsFunc) { m_metricsFunc = metricsFunc; }
    std::shared_ptr<MetricsFunction>    GetMetricsFunction(void) const { return m_metricsFunc; }

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
        SaveValue(os, m_name);
        SaveIndex(os, m_epoch);
        m_net->Save(os);
    }

    void Load(std::istream &is)
    {
        LoadValue(is, m_name);
        m_epoch = LoadIndex(is);
        m_net->Load(is);
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
        m_net->Save(archive);
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("name", m_name));
        archive(cereal::make_nvp("epoch", m_epoch));
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
        std::string csv_file_name = m_name + "_metrics.txt";
        std::string log_file_name = m_name + "_log.txt";
#ifdef BB_WITH_CEREAL
        std::string net_file_name = m_name + "_net.json";
#else
        std::string net_file_name = m_name + "_net.bin";
#endif

        // ログファイルオープン
        std::ofstream ofs_log;
        if ( m_log_write ) {
            ofs_log.open(log_file_name, m_log_append ? std::ios::app : std::ios::out);
        }

        {
            // ログ出力先設定
            ostream_tee log_stream;
            log_stream.add(std::cout);
            if (ofs_log.is_open()) { log_stream.add(ofs_log); }
            
            if (ofs_log.is_open()) {
                m_net->PrintInfo(0,  ofs_log);
                ofs_log << "-----------------------------------"    << std::endl;
                ofs_log << "epoch_size      : " << epoch_size       << std::endl;
                ofs_log << "mini_batch_size : " << batch_size       << std::endl;
                ofs_log << "-----------------------------------"    << std::endl;
            }
            
            // 以前の計算があれば読み込み
            if ( m_file_read ) {
#ifdef BB_WITH_CEREAL
                if ( RunStatus::ReadJson(net_file_name, m_net, m_name, m_epoch) ) {
                    std::cout << "[load] " << net_file_name << std::endl;
                }
                else {
                    std::cout << "[file not found] " << net_file_name << std::endl;
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
                auto test_metrics  = Calculation(td.x_test,  td.x_shape, td.t_test,  td.t_shape, batch_size, 0, m_metricsFunc, m_lossFunc, nullptr, false, m_print_progress);
                auto test_loss     = m_lossFunc->GetLoss();
                auto train_metrics = Calculation(td.x_train, td.x_shape, td.t_train, td.t_shape, batch_size, 0, m_metricsFunc, m_lossFunc, nullptr, false, m_print_progress);
                auto train_loss    = m_lossFunc->GetLoss();
                log_stream << "[initial] "
                    << "test "  << m_metricsFunc->GetMetricsString() << " : " << std::setw(6) << std::fixed << std::setprecision(4) << test_metrics  << "  "
                    << "test loss : "                                         << std::setw(6) << std::fixed << std::setprecision(4) << test_loss     << "  "
                    << "train " << m_metricsFunc->GetMetricsString() << " : " << std::setw(6) << std::fixed << std::setprecision(4) << train_metrics << "  "
                    << "train loss : "                                        << std::setw(6) << std::fixed << std::setprecision(4) << train_loss    << std::endl;
            }

            // 開始時間記録
            auto start_time = std::chrono::system_clock::now();

            for (int epoch = 0; epoch < epoch_size; ++epoch) {
                auto td_work = td;
                if ( m_data_augmentation_proc != nullptr ) {
                    m_data_augmentation_proc(td_work, m_mt(), m_data_augmentation_user);
                }

                // 学習実施
                m_epoch++;
                Calculation(td_work.x_train, td_work.x_shape, td_work.t_train, td_work.t_shape, batch_size, batch_size,
                                        m_metricsFunc, m_lossFunc, m_optimizer, true, m_print_progress, m_print_progress_loss, m_print_progress_accuracy);

                // ネット保存
                if (m_file_write) {
#ifdef BB_WITH_CEREAL
                    if ( m_write_serial ) {
                        std::stringstream fname;
                        fname << m_name << "_net_" << m_epoch << ".json";
                        if ( RunStatus::WriteJson(fname.str(), m_net, m_name, m_epoch) ) {
                            std::cout << "[save] " << fname.str() << std::endl;
                        }
                        else {
                            std::cout << "[write error] " << fname.str() << std::endl;
                        }
                    }

                    {
                        if ( RunStatus::WriteJson(net_file_name, m_net, m_name, m_epoch) ) {
                        //  std::cout << "[save] " << net_file_name << std::endl;
                        }
                        else {
                            std::cout << "[write error] " << net_file_name << std::endl;
                        }
                    }
#else
                    if ( m_write_serial ) {
                        std::stringstream fname;
                        fname << m_name << "_net_" << m_epoch << ".bin";
                        SaveBinary(fname.str());
                        std::cout << "[save] " << fname.str() << std::endl;
            //          log_streamt << "[save] " << fname.str() << std::endl;
                    }

                    {
                        SaveBinary(net_file_name);
            //          log_stream << "[save] " << net_file_name << std::endl;
                    }
#endif
                }

                // 学習状況評価
                {
                    double now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
                    auto test_metrics  = Calculation(td_work.x_test,  td_work.x_shape, td_work.t_test,  td_work.t_shape, batch_size, 0, m_metricsFunc, m_lossFunc, nullptr, false, m_print_progress);
                    auto test_loss     = m_lossFunc->GetLoss();
                    auto train_metrics = Calculation(td_work.x_train, td_work.x_shape, td_work.t_train, td_work.t_shape, batch_size, 0, m_metricsFunc, m_lossFunc, nullptr, false, m_print_progress);
                    auto train_loss    = m_lossFunc->GetLoss();
                    log_stream  << std::setw(10) << std::fixed << std::setprecision(2) << now_time << "s "
                                << "epoch[" << std::setw(3) << m_epoch << "] "
                                << "test "  << m_metricsFunc->GetMetricsString() << " : " << std::setw(6) << std::fixed << std::setprecision(4) << test_metrics  << "  "
                                << "test loss : "                                         << std::setw(6) << std::fixed << std::setprecision(4) << test_loss     << "  "
                                << "train " << m_metricsFunc->GetMetricsString() << " : " << std::setw(6) << std::fixed << std::setprecision(4) << train_metrics << "  "
                                << "train loss : "                                        << std::setw(6) << std::fixed << std::setprecision(4) << train_loss    << std::endl;
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
        return Calculation(td.x_test,  td.x_shape, td.t_test,  td.t_shape, batch_size, 0, m_metricsFunc, nullptr, nullptr, false, m_print_progress);
    }

    
protected:
    double Calculation(
                std::vector< std::vector<T> > const &x,
                indices_t x_shape,
                std::vector< std::vector<T> > const &t,
                indices_t t_shape,
                index_t max_batch_size,
                index_t min_batch_size,
                std::shared_ptr< MetricsFunction > metricsFunc = nullptr,
                std::shared_ptr< LossFunction >    lossFunc = nullptr,  
                std::shared_ptr< Optimizer >       optimizer = nullptr,
                bool train = false,
                bool print_progress = false,
                bool print_progress_loss = true,
                bool print_progress_metrics = true
            )

    {
        BB_ASSERT(x.size() == t.size());

        if ( metricsFunc != nullptr ) {
            metricsFunc->Clear();
        }
        if ( lossFunc != nullptr ) {
            lossFunc->Clear();
        }
        
        index_t frame_size = (index_t)x.size();
        
        FrameBuffer x_buf;
        FrameBuffer t_buf;

        index_t index = 0;
        while ( index < frame_size )
        {
            // ミニバッチサイズ計算
            index_t  mini_batch_size = std::min(max_batch_size, frame_size - index);

            // 残数が規定以下なら抜ける
            if ( mini_batch_size < min_batch_size ) {
                break;
            }

            index_t i = 0;
            while ( i < mini_batch_size ) {
                index_t  run_size = mini_batch_size - i;
                if (m_max_run_size > 0 && run_size > m_max_run_size) {
                    run_size = m_max_run_size;
                }

                // 学習データセット
                x_buf.Resize(run_size, x_shape, DataType<T>::type);
                x_buf.SetVector(x, index + i);

                // Forward
                auto y_buf = m_net->Forward(x_buf, train);

                // 期待値データセット
                t_buf.Resize(run_size, t_shape, DataType<T>::type);
                t_buf.SetVector(t, index + i);
                
                FrameBuffer dy_buf;
                if ( lossFunc != nullptr ) {
                    dy_buf = lossFunc->CalculateLoss(y_buf, t_buf, mini_batch_size);
                }

                if ( metricsFunc != nullptr ) {
                    metricsFunc->CalculateMetrics(y_buf, t_buf);
                }

                if ( train && lossFunc != nullptr ) {
                    auto dx = m_net->Backward(dy_buf);
                }

                i += run_size;
            }

            if ( train && lossFunc != nullptr ) {
                if ( optimizer != nullptr ) {
                    optimizer->Update();
                }
            }

            // print progress
            if ( print_progress ) {
                std::stringstream ss;

                index_t progress = index + mini_batch_size;
                index_t rate = progress * 100 / frame_size;
                ss << "\r[" << rate << "% (" << progress << "/" << frame_size << ")]";

                if ( print_progress_loss && lossFunc != nullptr ) {
                    ss << "  loss : " << lossFunc->GetLoss();
                }

                if ( print_progress_metrics && metricsFunc != nullptr ) {
                    ss << "  " << metricsFunc->GetMetricsString() << " : " << metricsFunc->GetMetrics();
                }
                ss << "        ";

                std::cerr << ss.str() << std::flush;
            }

            // インデックスを進める
            index += mini_batch_size;
        }

        // clear progress
        if ( print_progress ) {
            std::cerr << "\r                                                                               \r" << std::flush;
        }

        return metricsFunc->GetMetrics();
    }

};


}