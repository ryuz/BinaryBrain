﻿// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <omp.h>
#include <iostream>
#include <string.h>

#ifdef BB_WITH_CUDA
#include "bbcu/bbcu.h"
#endif


void Cifar10StochasticLutSimple(int epoch_size, int mini_batch_size,                            int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10StochasticLutCnn   (int epoch_size, int mini_batch_size,                            int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10SparseLutSimple    (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10SparseLutCnn       (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10MicroMlpLutSimple  (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10MicroMlpLutCnn     (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10DenseSimple        (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10DenseCnn           (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10AeSparseLutSimple  (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10AeSparseLutCnn     (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);

void Cifar10StochasticLutBnCnn(int epoch_size, int mini_batch_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10MobileNet(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10ShuffleNet(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10BinarizeTest(void);

// メイン関数
int main(int argc, char *argv[])
{
    // set default parameter
    std::string netname               = "All";
    int         device                = 0;
    int         epoch_size            = 8;
    int         mini_batch_size       = 32;
    int         train_modulation_size = 7;
    int         test_modulation_size  = 0;
    bool        file_read             = false;
    bool        binary_mode           = true;
    bool        print_device          = false;

    if ( argc < 2 ) {
        std::cout << "usage:" << std::endl;
        std::cout << argv[0] << " [options] <netname>" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "options" << std::endl;
        std::cout << "  -epoch <epoch size>                     set epoch size" << std::endl;
        std::cout << "  -mini_batch <mini_batch size>           set mini batch size" << std::endl;
        std::cout << "  -modulation_size <modulation_size>      set train modulation size" << std::endl;
        std::cout << "  -test_modulation_size <modulation_size> set test modulation size" << std::endl;
        std::cout << "  -binary <0|1>                           set binary mode" << std::endl;
        std::cout << "  -read <0|1>                             file read" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "netname" << std::endl;
        std::cout << "  StochasticLutSimple Stochastic-Lut LUT-Network Simple DNN" << std::endl;
        std::cout << "  StochasticLutCnn    Stochastic-Lut LUT-Network CNN" << std::endl;
        std::cout << "  SparseLutSimple     Sparse LUT-Network Simple DNN" << std::endl;
        std::cout << "  SparseLutCnn        Sparse LUT-Network CNN" << std::endl;
        std::cout << "  DenseMlp            Dense Simple DNN" << std::endl;
        std::cout << "  DenseCnn            Dense CNN" << std::endl;
        std::cout << "  All                 run all" << std::endl;
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
        else if (strcmp(argv[i], "-device") == 0 && i + 1 < argc) {
            ++i;
            device = (strtoul(argv[i], NULL, 0) != 0);
        }
        else if (strcmp(argv[i], "-print_device") == 0 ) {
            print_device = true;
        }
        else {
            netname = argv[i];
        }
    }
    
    if (test_modulation_size <= 0) {
        test_modulation_size = train_modulation_size;
    }


#ifdef BB_WITH_CUDA
    bbcu_SetDevice(device);
    std::cout << " devide : " << bbcu_GetDevice() << std::endl;

    if ( print_device ) {
        bbcu::PrintDeviceProperties(device);
    }
#endif
    

    if ( netname == "All" || netname == "StochasticLutSimple" ) {
        Cifar10StochasticLutSimple(epoch_size, mini_batch_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "StochasticLutCnn" ) {
        Cifar10StochasticLutCnn(epoch_size, mini_batch_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "StochasticLutBnCnn" ) {
        Cifar10StochasticLutBnCnn(epoch_size, mini_batch_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "SparseLutSimple" ) {
        Cifar10SparseLutSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "SparseLutCnn" ) {
        Cifar10SparseLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "MicroMlpLutSimple" ) {
        Cifar10MicroMlpLutSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "MicroMlpLuCnn" ) {
        Cifar10MicroMlpLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DenseSimple" ) {
        Cifar10DenseSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DenseCnn" ) {
        Cifar10DenseCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    
    if ( netname == "All" || netname == "AeSparseLutSimple" ) {
        Cifar10AeSparseLutSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "AeSparseLutCnn" ) {
        Cifar10AeSparseLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "MobileNet" ) {
        Cifar10MobileNet(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    
    if ( netname == "All" || netname == "ShuffleNet" ) {
        Cifar10ShuffleNet(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "BinarizeTest" ) {
        Cifar10BinarizeTest();
    }

    return 0;
}


// end of file
