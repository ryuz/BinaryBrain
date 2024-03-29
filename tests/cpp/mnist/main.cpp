﻿// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <omp.h>
#include <iostream>
#include <string.h>

#include "bb/Version.h"

#ifdef BB_WITH_CUDA
#include "bbcu/bbcu.h"
#endif


//  サンプルごとにプログラムを分けると特にVisualStudioでメンテナンスが面倒なので
// 1個のプロジェクトに複数サンプルを統合して引数で呼び分けています。


void MnistStochasticLutSimple      (int epoch_size, int mini_batch_size,                            int test_modulation_size, bool binary_mode, bool file_read);
void MnistStochasticLutCnn         (int epoch_size, int mini_batch_size,                            int test_modulation_size, bool binary_mode, bool file_read);
void MnistDifferentiableLutSimple  (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistDifferentiableLutCnn     (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistMicroMlpLutSimple        (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistMicroMlpLutCnn           (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistDenseSimple              (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistDenseCnn                 (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistDenseAffineQuantize      (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistAeDifferentiableLutSimple(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistAeDifferentiableLutCnn   (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistCustomModel              (int epoch_size, int mini_batch_size,                                                      bool binary_mode, bool file_read);
void MnistLoadNet                  (int epoch_size, int mini_batch_size, std::string filename);

void MnistDenseBinary(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);

void MnistLoopDenseCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);


// メイン関数
int main(int argc, char *argv[])
{
    // set default parameter
    std::string netname               = "All";
    int         epoch_size            = 8;
    int         mini_batch_size       = 32;
    int         train_modulation_size = 7;
    int         test_modulation_size  = 0;
    bool        file_read             = false;
    bool        binary_mode           = true;
    bool        print_device          = false;
    std::string load_net_file         = "MnistDifferentiableLutSimple.bb_net";

    std::cout << "BinaryBrain version " << bb::GetVersionString();
    std::cout << "  MNIST sample\n" << std::endl;

    if ( argc < 2 ) {
        std::cout << "usage:" << std::endl;
        std::cout << argv[0] << " [options] <sample name>" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "options" << std::endl;
        std::cout << "  -epoch <epoch size>                     set epoch size" << std::endl;
        std::cout << "  -mini_batch <mini_batch size>           set mini batch size" << std::endl;
        std::cout << "  -modulation_size <modulation_size>      set train modulation size" << std::endl;
        std::cout << "  -test_modulation_size <modulation_size> set test modulation size" << std::endl;
        std::cout << "  -binary <0|1>                           set binary mode" << std::endl;
        std::cout << "  -read <0|1>                             file read" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "<sample name>" << std::endl;
        std::cout << "  StochasticLutSimple       Stochastic LUT Network Simple DNN" << std::endl;
        std::cout << "  StochasticLutCnn          Stochastic LUT Network CNN" << std::endl;
        std::cout << "  DifferentiableLutSimple   Differentiable LUT Network Simple DNN" << std::endl;
        std::cout << "  DifferentiableLutCnn      Differentiable LUT Network CNN" << std::endl;
        std::cout << "  DenseSimple               Dense Simple DNN" << std::endl;
        std::cout << "  DenseCnn                  Dense CNN" << std::endl;
        std::cout << "  AeDifferentiableLutSimple AutoEncoder Simple DNN" << std::endl;
        std::cout << "  AeDifferentiableLutCnn    AutoEncoder CNN" << std::endl;
        std::cout << "  Custom                    Custum mode" << std::endl;
        std::cout << "  All                       run all" << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-num_threads") == 0 && i + 1 < argc) {
            ++i;
            int num_threads = (int)strtoul(argv[i], NULL, 0);
#ifdef _OPENMP
            omp_set_num_threads(num_threads);
#endif
        }
        else if (strcmp(argv[i], "-epoch") == 0 && i + 1 < argc) {
            ++i;
            epoch_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-mini_batch") == 0 && i + 1 < argc) {
            ++i;
            mini_batch_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-modulation_size") == 0 && i + 1 < argc) {
            ++i;
            train_modulation_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-test_modulation_size") == 0 && i + 1 < argc) {
            ++i;
            test_modulation_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-binary") == 0 && i + 1 < argc) {
            ++i;
            binary_mode = (strtoul(argv[i], NULL, 0) != 0);
        }
        else if (strcmp(argv[i], "-read") == 0 && i + 1 < argc) {
            ++i;
            file_read = (strtoul(argv[i], NULL, 0) != 0);
        }
        else if (strcmp(argv[i], "-print_device") == 0 ) {
            print_device = true;
        }
        else if (strcmp(argv[i], "-load_net") == 0 && i + 1 < argc) {
            ++i;
            load_net_file = argv[i];
        }
        else {
            netname = argv[i];
        }
    }

    if (test_modulation_size <= 0) {
        test_modulation_size = train_modulation_size;
    }


#ifdef BB_WITH_CUDA
    int device = 0;
    cudaSetDevice(device);

    if ( print_device ) {
        bbcu::PrintDeviceProperties(device);
    }
#endif

   

    if ( netname == "All" || netname == "StochasticLutSimple" ) {
        MnistStochasticLutSimple(epoch_size, mini_batch_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "StochasticLutCnn" ) {
        MnistStochasticLutCnn(epoch_size, mini_batch_size,test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DifferentiableLutSimple" ) {
        MnistDifferentiableLutSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DifferentiableLutCnn" ) {
        MnistDifferentiableLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    
    if ( netname == "All" || netname == "MicroMlpLutSimple" ) {
        MnistMicroMlpLutSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "MicroMlpLuCnn" ) {
        MnistMicroMlpLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DenseSimple" ) {
        MnistDenseSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DenseAffineQuantize" ) {
        MnistDenseAffineQuantize(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DenseCnn" ) {
        MnistDenseCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DenseBinary" ) {
       MnistDenseBinary(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    
    if ( netname == "All" || netname == "AeDifferentiableLutSimple" ) {
        MnistAeDifferentiableLutSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    if ( netname == "All" || netname == "AeDifferentiableLutCnn" ) {
        MnistAeDifferentiableLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    
//    if ( netname == "All" || netname == "LoopDenseCnn" ) {
//       MnistLoopDenseCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
//    }
    

    // カスタムモデルを自分で書く場合のサンプル
    if ( netname == "All" || strcmp(argv[1], "Custom") == 0 ) {
        MnistCustomModel(epoch_size, mini_batch_size, binary_mode, file_read);
    }

#ifdef BB_OBJECT_LOADER
    // 他で作ったネットの読み込み確認
    if ( strcmp(argv[1], "LoadNetFile") == 0 ) {
        MnistLoadNet(epoch_size, mini_batch_size, load_net_file);
    }
#endif

    return 0;
}

