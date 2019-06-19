// --------------------------------------------------------------------------
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


void Cifar10StochasticLutMlp(int epoch_size, int mini_batch_size,                            int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10StochasticLutCnn(int epoch_size, int mini_batch_size,                            int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10SparseLutMlp    (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10SparseLutCnn    (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10MicroMlpLutMlp  (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10MicroMlpLutCnn  (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10DenseMlp        (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10DenseCnn        (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void Cifar10NrSparseLutCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);


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
        std::cout << "  StochasticLutMlp Stochastic-Lut LUT-Network Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  StochasticLutCnn Stochastic-Lut LUT-Network CNN" << std::endl;
        std::cout << "  SparseLutMlp     Sparse LUT-Network Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  SparseLutCnn     Sparse LUT-Network CNN" << std::endl;
        std::cout << "  DenseMlp         Dense Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  DenseCnn         Dense Simple CNN" << std::endl;
        std::cout << "  All              run all" << std::endl;
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
        else {
            netname = argv[i];
        }
    }
    
    if (test_modulation_size <= 0) {
        test_modulation_size = train_modulation_size;
    }


#ifdef BB_WITH_CUDA
    if ( print_device ) {
        bbcu::PrintDeviceProperties();
    }
#endif

    if ( netname == "All" || netname == "StochasticLutMlp" ) {
        Cifar10StochasticLutMlp(epoch_size, mini_batch_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "StochasticLutCnn" ) {
        Cifar10StochasticLutCnn(epoch_size, mini_batch_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "SparseLutMlp" ) {
        Cifar10SparseLutMlp(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "SparseLutCnn" ) {
        Cifar10SparseLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "MicroMlpLutMlp" ) {
        Cifar10MicroMlpLutMlp(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "MicroMlpLuCnn" ) {
        Cifar10MicroMlpLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DenseMlp" ) {
        Cifar10DenseMlp(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "DenseCnn" ) {
        Cifar10DenseCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "NrSparseLutCnn" ) {
        Cifar10NrSparseLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    return 0;
}


// end of file
