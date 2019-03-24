// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <omp.h>
#include <string.h>


void Cifar10SimpleLutMlp(int epoch_size, size_t mini_batch_size, bool binary_mode);
void Cifar10SimpleLutCnn(int epoch_size, size_t mini_batch_size, bool binary_mode);


// メイン関数
int main(int argc, char *argv[])
{
 	omp_set_num_threads(2);

    std::string netname = "All";
    int         epoch_size      = 8;
    int         mini_batch_size = 8;
    bool        binary_mode = true;

	if ( argc < 2 ) {
        std::cout << "usage:" << std::endl;
        std::cout << argv[0] << " [options] <netname>" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "options" << std::endl;
        std::cout << "  -epoch      <epoch size>        set epoch size" << std::endl;
        std::cout << "  -mini_batch <mini_batch size>   set mini batch size" << std::endl;
        std::cout << "  -binary     <0|1>               set binary mode" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "netname" << std::endl;
        std::cout << "  LutMlp       LUT-Network Simple Multi Layer Perceptron" << std::endl;
        std::cout << "  LutCnn       LUT-Network Simple CNN" << std::endl;
        std::cout << "  All          run all" << std::endl;
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
        else if (strcmp(argv[i], "-binary_mode") == 0 && i + 1 < argc) {
            ++i;
            binary_mode = (strtoul(argv[i], NULL, 0) != 0);
        }
        else {
            netname = argv[i];
        }
    }


	if ( netname == "All" || netname == "LutMlp" ) {
		Cifar10SimpleLutMlp(epoch_size, mini_batch_size, true);
	}

	if ( netname == "All" || netname == "LutCnn" ) {
    	Cifar10SimpleLutCnn(epoch_size, mini_batch_size, true);
	}

	return 0;
}

