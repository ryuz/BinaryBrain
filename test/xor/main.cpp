// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/json.hpp>

#include "bb/NeuralNet.h"
#include "bb/NeuralNetUtility.h"

#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetReLU.h"
#include "bb/NeuralNetSoftmax.h"
#include "bb/NeuralNetBinarize.h"

#include "bb/NeuralNetLut.h"

#include "bb/NeuralNetBinaryMultiplex.h"

#include "bb/NeuralNetBatchNormalization.h"

#include "bb/NeuralNetAffine.h"
#include "bb/NeuralNetSparseAffine.h"
#include "bb/NeuralNetSparseBinaryAffine.h"

#include "bb/NeuralNetRealToBinary.h"
#include "bb/NeuralNetBinaryToReal.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryLut6VerilogXilinx.h"

#include "bb/NeuralNetSparseAffineSigmoid.h"

#include "bb/NeuralNetOptimizerSgd.h"
#include "bb/NeuralNetOptimizerAdam.h"

#include "bb/NeuralNetConvolution.h"
#include "bb/NeuralNetMaxPooling.h"

#include "bb/NeuralNetLossCrossEntropyWithSoftmax.h"
#include "bb/NeuralNetLossMeanSquaredError.h"

#include "bb/NeuralNetAccuracyCategoricalClassification.h"
#include "bb/NeuralNetAccuracyBool.h"

#include "bb/NeuralNetConvolutionPack.h"

#include "bb/ShuffleSet.h"

#include "bb/LoadXor.h"




// DenseAffine Real network
template <int N, int M>
bool TestXor6(int max_epoc_size, size_t max_batch_size, std::uint64_t seed, int& epocs)
{
	// run name
	std::string run_name = "TestXor6";
	int			num_class = 1;

	// load MNIST data
	int mul = 1024;
	auto data = bb::LoadXor<float>::Load(N, mul);
	
	// build layer
	bb::NeuralNetLut<N, M>		layer0_lut(N, 1, seed);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_lut);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	for (int i = 0; i < N; i++) {
		layer0_lut.SetNodeInput(0, i, i);
	}

	// binaraize
	net.SetBinaryMode(true);
//	net.SetBinaryMode(false);

	// run fitting
//	bb::NeuralNetLossCrossEntropyWithSoftmax<>				loss_func;
	bb::NeuralNetLossMeanSquaredError<>				loss_func;
	bb::NeuralNetAccuracyBool<>								acc_func;

	epocs = 0;
	for ( int epoc = 0; epoc < max_epoc_size; ++epoc ) {
		epocs += mul;
		net.RunCalculation(data.x_test, data.y_test, max_batch_size, max_batch_size, &acc_func, &loss_func, true, false);
		auto accuracy = net.RunCalculation(data.x_test, data.y_test, max_batch_size, 0, &acc_func);
		if (accuracy == 1.0) { return true; }
	}
	return false;
}


template <int N, int M>
void TestXor6Loop(int max_epoc_size, size_t max_batch_size, std::uint64_t num)
{
	int ok = 0;
	int epocs_total = 0;
	for (std::uint64_t seed = 1; seed <= num; ++seed) {
		int epocs;
		if (TestXor6<N, M>(max_epoc_size, max_batch_size, seed, epocs)) {
			epocs_total += epocs;
			ok++;
		}
	}

	double ave_epocs = 0;
	if (ok > 0) {
		ave_epocs = (double)epocs_total / ok;
	}

	std::cout << "M = " << M << "  (" << ok << " / " << num << ")  average_epocs = " << ave_epocs << std::endl;
}


// メイン関数
int main()
{
	omp_set_num_threads(1);

	/*
	TestXor6Loop<6, 64>(4096, 64, 100);
	TestXor6Loop<6, 48>(4096, 64, 100);
	TestXor6Loop<6, 32>(4096, 64, 100);
	TestXor6Loop<6, 16>(4096, 64, 100);
	TestXor6Loop<6, 15>(4096, 64, 100);
	TestXor6Loop<6, 14>(4096, 64, 100);
	TestXor6Loop<6, 13>(4096, 64, 100);
	TestXor6Loop<6, 12>(4096, 64, 100);
	TestXor6Loop<6, 11>(4096, 64, 100);
	TestXor6Loop<6, 10>(4096, 64, 100);
	TestXor6Loop<6, 9> (4096, 64, 100);
	TestXor6Loop<6, 8> (4096, 64, 100);
	TestXor6Loop<6, 7> (4096, 64, 100);
	TestXor6Loop<6, 6> (4096, 64, 100);
	TestXor6Loop<6, 5> (4096, 64, 100);
	*/

	int max_epoc = 4096;
	int num = 100;
	TestXor6Loop<6, 64>(max_epoc, 64, num);

#if 1
	TestXor6Loop<6, 1>(max_epoc, 64, num);
	TestXor6Loop<6, 2>(max_epoc, 64, num);
	TestXor6Loop<6, 3>(max_epoc, 64, num);
	TestXor6Loop<6, 4>(max_epoc, 64, num);
	TestXor6Loop<6, 5>(max_epoc, 64, num);
	TestXor6Loop<6, 6>(max_epoc, 64, num);
	TestXor6Loop<6, 7>(max_epoc, 64, num);
	TestXor6Loop<6, 8>(max_epoc, 64, num);
	TestXor6Loop<6, 9>(max_epoc, 64, num);
	TestXor6Loop<6, 10>(max_epoc, 64, num);
	TestXor6Loop<6, 11>(max_epoc, 64, num);
	TestXor6Loop<6, 12>(max_epoc, 64, num);
	TestXor6Loop<6, 14>(max_epoc, 64, num);
	TestXor6Loop<6, 15>(max_epoc, 64, num);
	TestXor6Loop<6, 16>(max_epoc, 64, num);
	TestXor6Loop<6, 17>(max_epoc, 64, num);
	TestXor6Loop<6, 18>(max_epoc, 64, num);
	TestXor6Loop<6, 19>(max_epoc, 64, num);
	TestXor6Loop<6, 20>(max_epoc, 64, num);
	TestXor6Loop<6, 24>(max_epoc, 64, num);
	TestXor6Loop<6, 32>(max_epoc, 64, num);
	TestXor6Loop<6, 48>(max_epoc, 64, num);
	TestXor6Loop<6, 64>(max_epoc, 64, num);
	TestXor6Loop<6, 128>(max_epoc, 64, num);
#endif

	getchar();

	return 0;
}

