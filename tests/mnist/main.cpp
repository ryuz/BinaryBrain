// --------------------------------------------------------------------------
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


void MnistStochasticLutSimple(int epoch_size, int mini_batch_size,                            int test_modulation_size, bool binary_mode, bool file_read);
void MnistStochasticLutCnn   (int epoch_size, int mini_batch_size,                            int test_modulation_size, bool binary_mode, bool file_read);
void MnistSparseLutSimple    (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistSparseLutCnn       (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistSparseLutCnnDa     (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistSparseLutCnnDa2    (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistMicroMlpLutSimple  (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistMicroMlpLutCnn     (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistDenseSimple        (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistDenseCnn           (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistCustomModel        (int epoch_size, int mini_batch_size,                                                      bool binary_mode                );
void MnistAeSparseLutSimple  (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistAeSparseLutCnn     (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);

void MnistDetectionSparseLutSimple(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistDetectionSparseLutCnn   (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);

void MnistMobileNet          (int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistSegmentationDenseCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistSegmentationSparseLutCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);
void MnistSegmentationMobileNet(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read);



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

    std::cout << "BinaryBrain version " << bb::GetVersionString() << std::endl;

    if ( argc < 2 ) {
        std::cout << "usage:" << std::endl;
        std::cout << argv[0] << " [options] <model name>" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "options" << std::endl;
        std::cout << "  -epoch <epoch size>                     set epoch size" << std::endl;
        std::cout << "  -mini_batch <mini_batch size>           set mini batch size" << std::endl;
        std::cout << "  -modulation_size <modulation_size>      set train modulation size" << std::endl;
        std::cout << "  -test_modulation_size <modulation_size> set test modulation size" << std::endl;
        std::cout << "  -binary <0|1>                           set binary mode" << std::endl;
        std::cout << "  -read <0|1>                             file read" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "<model name>" << std::endl;
        std::cout << "  StochasticLutSimple Stochastic-Lut LUT-Network Simple DNN" << std::endl;
        std::cout << "  StochasticLutCnn    Stochastic-Lut LUT-Network CNN" << std::endl;
        std::cout << "  SparseLutSimple     Sparse LUT-Network Simple DNN" << std::endl;
        std::cout << "  SparseLutCnn        Sparse LUT-Network CNN" << std::endl;
        std::cout << "  DenseSimple         Dense Simple DNN" << std::endl;
        std::cout << "  DenseCnn            Dense CNN" << std::endl;
        std::cout << "  AeSparseLutSimple   AutoEncoder Simple DNN" << std::endl;
        std::cout << "  AeSparseLutCnn      AutoEncoder CNN" << std::endl;
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
        MnistStochasticLutSimple(epoch_size, mini_batch_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "StochasticLutCnn" ) {
        MnistStochasticLutCnn(epoch_size, mini_batch_size,test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "SparseLutSimple" ) {
        MnistSparseLutSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "SparseLutCnn" ) {
        MnistSparseLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "SparseLutCnnDa" ) {
//        MnistSparseLutCnnDa(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    if ( netname == "SparseLutCnnDa2" ) {
//        MnistSparseLutCnnDa2(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
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

    if ( netname == "All" || netname == "DenseCnn" ) {
        MnistDenseCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "AeSparseLutSimple" ) {
        MnistAeSparseLutSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "AeSparseLutCnn" ) {
        MnistAeSparseLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "All" || netname == "MobileNet" ) {
        MnistMobileNet(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    // (おまけ)レイヤー内部を自分で書く人向けサンプル
    if ( strcmp(argv[1], "Custom") == 0 ) {
        MnistCustomModel(epoch_size, mini_batch_size, binary_mode);
    }


    if ( netname == "DetectionSparseLutSimple" ) {
        MnistDetectionSparseLutSimple(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "DetectionSparseLutCnn" ) {
        MnistDetectionSparseLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "SegmentationDenseCnn" ) {
        MnistSegmentationDenseCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "SegmentationSparseLutCnn" ) {
        MnistSegmentationSparseLutCnn(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    if ( netname == "SegmentationMobileNet" ) {
        MnistSegmentationMobileNet(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }

    return 0;
}

