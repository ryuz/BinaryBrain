// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <omp.h>
#include <string.h>

void MnistDenseMlp(int epoch_size, size_t mini_batch_size);
void MnistDenseCnn(int epoch_size, size_t mini_batch_size);
void MnistStochasticLut2Mlp(int epoch_size, size_t mini_batch_size, int lut_frame_mux_size, bool binary_mode);
void MnistStochasticLut4Mlp(int epoch_size, size_t mini_batch_size, int lut_frame_mux_size, bool binary_mode);
void MnistStochasticLut6Mlp(int epoch_size, size_t mini_batch_size, int lutframe_mux_size, bool binary_mode);
void MnistStochasticLut6Cnn(int epoch_size, size_t mini_batch_size, int lutframe_mux_size, bool binary_mode);
void MnistMicroMlpLutMlp(int epoch_size, size_t mini_batch_size, int frame_mux_size, int lut_frame_mux_size, bool binary_mode);
void MnistMicroMlpLutCnn(int epoch_size, size_t mini_batch_size, int frame_mux_size, int lut_frame_mux_size, bool binary_mode);
void MnistMicroMlpScratch(int epoch_size, size_t mini_batch_size, bool binary_mode);


// メイン関数
int main(int argc, char *argv[])
{
 	omp_set_num_threads(4);

    std::string netname = "All";
    int         epoch_size         = 8;
    int         mini_batch_size    = 32;
    int         frame_mux_size     = 1;
    int         lut_frame_mux_size = 15;
    bool        binary_mode = true;

	if ( argc < 2 ) {
        std::cout << "usage:" << std::endl;
        std::cout << argv[0] << " [options] <netname>" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "options" << std::endl;
        std::cout << "  -epoch <epoch size>                set epoch size" << std::endl;
        std::cout << "  -mini_batch <mini_batch size>      set mini batch size" << std::endl;
        std::cout << "  -frame_mux_size <frame_mux_size>     set training modulation size" << std::endl;
        std::cout << "  -lut_frame_mux_size <frame_mux_size> set binary-lut modulation size" << std::endl;
        std::cout << "  -binary <0|1>                      set binary mode" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "netname" << std::endl;
        std::cout << "  StochasticLutMlp Stochastic-Lut LUT-Network Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  StochasticLutCnn Stochastic-Lut  LUT-Network Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  LutMlp           micro-MLP LUT-Network Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  LutCnn           micro-MLP LUT-Network Simple CNN" << std::endl;
        std::cout << "  DenseMlp         FP32 Fully Connection Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  DenseCnn         FP32 Fully Connection Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  All              run all" << std::endl;
		return 1;
	}

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-epoch") == 0 && i + 1 < argc) {
            ++i;
            epoch_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-mini_batch") == 0 && i + 1 < argc) {
            ++i;
            mini_batch_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-frame_mux_size") == 0 && i + 1 < argc) {
            ++i;
            frame_mux_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-lut_frame_mux_size") == 0 && i + 1 < argc) {
            ++i;
            lut_frame_mux_size = (int)strtoul(argv[i], NULL, 0);
        }
        else if (strcmp(argv[i], "-binary_mode") == 0 && i + 1 < argc) {
            ++i;
            binary_mode = (strtoul(argv[i], NULL, 0) != 0);
        }
        else {
            netname = argv[i];
        }
    }

//  MnistStochasticLut2Mlp(epoch_size, mini_batch_size, lut_frame_mux_size, true);
//  MnistStochasticLut4Mlp(epoch_size, mini_batch_size, lut_frame_mux_size, true);

	if ( netname == "All" || netname == "StochasticLutMlp" ) {
		MnistStochasticLut6Mlp(epoch_size, mini_batch_size, lut_frame_mux_size, true);
	}

	if ( netname == "All" || netname == "StochasticLutCnn" ) {
    	MnistStochasticLut6Cnn(epoch_size, mini_batch_size, lut_frame_mux_size, true);
	}

	if ( netname == "All" || netname == "LutMlp" ) {
		MnistMicroMlpLutMlp(epoch_size, mini_batch_size, frame_mux_size, lut_frame_mux_size, true);
	}

	if ( netname == "All" || netname == "LutCnn" ) {
    	MnistMicroMlpLutCnn(epoch_size, mini_batch_size, frame_mux_size, lut_frame_mux_size, true);
	}

	if ( netname == "All" || netname == "DenseMlp" ) {
		MnistDenseMlp(epoch_size, mini_batch_size);
	}

	if ( netname == "All" || netname == "DenseCnn" ) {
		MnistDenseCnn(epoch_size, mini_batch_size);
	}

	if ( strcmp(argv[1], "Scratch") == 0 ) {
        // レイヤー内部を自分で書く人向けサンプル
		MnistMicroMlpScratch(epoch_size, mini_batch_size, true);
	}

	return 0;
}

