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
#include "bb/NeuralNetDropout.h"

#include "bb/NeuralNetSparseMicroMlp.h"
#include "bb/NeuralNetSparseMicroMlpDiscrete.h"

#include "bb/NeuralNetBinaryMultiplex.h"

#include "bb/NeuralNetBatchNormalization.h"

#include "bb/NeuralNetDenseAffine.h"
#include "bb/NeuralNetSparseAffine.h"
#include "bb/NeuralNetSparseBinaryAffine.h"

#include "bb/NeuralNetRealToBinary.h"
#include "bb/NeuralNetBinaryToReal.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryLut6VerilogXilinx.h"
#include "bb/NeuralNetBinaryLutVerilog.h"

#include "bb/NeuralNetSparseAffineSigmoid.h"

#include "bb/NeuralNetOptimizerSgd.h"
#include "bb/NeuralNetOptimizerAdam.h"

#include "bb/NeuralNetDenseConvolution.h"
#include "bb/NeuralNetMaxPooling.h"

#include "bb/NeuralNetLossCrossEntropyWithSoftmax.h"
#include "bb/NeuralNetAccuracyCategoricalClassification.h"

#include "bb/NeuralNetLoweringConvolution.h"

#include "bb/ShuffleSet.h"

#include "bb/LoadMnist.h"
#include "bb/DataAugmentationMnist.h"


void WriteTestImage(void);


void MnistMlpLut(int epoc_size, size_t max_batch_size, bool binary_mode = true);
void MnistCnnBin(int epoc_size, size_t max_batch_size, bool binary_mode = true);

void MnistFullyCnn(int epoc_size, size_t max_batch_size, bool binary_mode = true);
void MnistFullyCnn2(int epoc_size, size_t max_batch_size, bool binary_mode = true);

void MnistDenseAffineReal(int epoc_size, size_t max_batch_size);
void MnistDenseAffineBinary(int epoc_size, size_t max_batch_size, bool binary_mode = true);

void MnistSparseAffineReal(int epoc_size, size_t max_batch_size);
void MnistSparseAffineBinary(int epoc_size, size_t max_batch_size, bool binary_mode = true);
void MnistSparseAffineLut6(int epoc_size, size_t max_batch_size);
void MnistSparseAffineBinToLut(int bin_epoc_size, size_t bin_max_batch_size, int lut_epoc_size, size_t lut_max_batch_size);

void MnistDenseSimpleConvolution(int epoc_size, size_t max_batch_size);

void MnistLutBinary(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistSparseAffineBinary2(int epoc_size, size_t max_batch_size, bool binary_mode = true);

void MnistLut2(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistLut3(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistLut5(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistLutA(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistLutC(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistLutD(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistLutE(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistLutF(int epoc_size, size_t max_batch_size, bool binary_mode);


void MnistLutSimpleConvolutionBinary(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistLutSimpleConvolutionBinary2(int epoc_size, size_t max_batch_size, bool binary_mode);


void MnistMlpLutMini(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistMlpLut2(int epoc_size, size_t max_batch_size, bool binary_mode);


void MnistDenseFullyCnn(int epoc_size, size_t max_batch_size, bool binary_mode);
void MnistDenseSimpleConvolution(int epoc_size, size_t max_batch_size, bool binary_mode, bool binarize_input);
void MnistDenseSimpleConvolution5x5(int epoc_size, size_t max_batch_size, bool binary_mode, bool binarize_input);
void MnistDenseSimpleConvolution3x3(int epoc_size, size_t max_batch_size, bool binary_mode, bool binarize_input, std::string name);


// メイン関数
int main()
{
//	WriteTestImage();
//	return 0;

	omp_set_num_threads(4);

//	MnistDenseSimpleConvolution(16, 256, false, true);
//	MnistDenseSimpleConvolution5x5(16, 256, false, false);

//	MnistDenseSimpleConvolution3x3(32, 64, true,  true,  "MnistDenseSimpleConvolution3x3_binact_binin");
//	MnistDenseSimpleConvolution3x3(32, 64, false, true,  "MnistDenseSimpleConvolution3x3_realact_binin");
//	MnistDenseSimpleConvolution3x3(32, 64, true,  false, "MnistDenseSimpleConvolution3x3_binact_realin");
//	MnistDenseSimpleConvolution3x3(32, 64, false, false, "MnistDenseSimpleConvolution3x3_realact_realin");

//	MnistDenseSimpleConvolution3x3(32, 64, true,  true,  "MnistDenseSimpleConvolution3x3_2_binact_binin");
	MnistDenseSimpleConvolution3x3(32, 64, false, true,  "MnistDenseSimpleConvolution3x3_2_realact_binin");


//	MnistCnnBin(512, 256, true);
	getchar();
	return 0;

//	MnistDenseFullyCnn(1, 256, true);
//	return 0;

//	MnistMlpLut(16, 256, true);
//	MnistMlpLut2(16, 256, true);
//	MnistMlpLutMini(16, 256, true);
//	getchar();
//	return 0;


	MnistFullyCnn(0, 64);
//	MnistFullyCnn2(256, 64);

//	MnistMlpLut(0, 256);
	return 0;


//	MnistCnnLut(2, 64);

//	MnistLutSimpleConvolutionBinary(1, 64, true);
//	MnistLutSimpleConvolutionBinary(4, 64, false);
//	MnistLutSimpleConvolutionBinary(128, 64, true);

//	MnistLutSimpleConvolutionBinary2(128, 64, true);

//	MnistLut2(128, 128, true);
//	MnistLut3(128, 256, true);
//	MnistLut5(128, 128, true);

//	MnistLutA(128, 128, true);
//	MnistLutD(128, 128, true);
//	MnistLutE(128, 256, true);
//	MnistLutF(128, 256, true);


	// Dense Affine
//	MnistDenseAffineReal(16, 128);

#if 1
//	MnistDenseAffineBinary(16, 128, false);
	MnistDenseAffineBinary(16, 128, true);
#endif


	// Sparse Affine
#if 0
	MnistSparseAffineReal(16, 128);
#endif

#if 0
//	MnistSparseAffineBinary(16, 128, false);
	MnistSparseAffineBinary(64, 128, true);
#endif


//	MnistLutBinary(64, 256, true);
#if 0
	MnistSparseAffineLut6(8, 8192);
#endif

#if 0
	MnistSparseAffineBinToLut(2, 128, 2, 8192);
#endif


	// Simple Convolution
#if 0
	MnistDenseSimpleConvolution(16, 128);
#endif

	getchar();

	return 0;
}


void MnistDenseSimpleConvolution(int epoc_size, size_t max_batch_size, bool binary_mode, bool binarize_input)
{
	// run name
	std::string run_name = "MnistDenseSimpleConvolution";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	if (binarize_input) {
		std::cout << "binarize input data" << std::endl;
		for (auto& vec_x : td.x_train) {
			for (auto& x : vec_x) {
				x = x > 0.5f ? 1.0f : 0.0f;
			}
		}
		for (auto& vec_x : td.x_test) {
			for (auto& x : vec_x) {
				x = x > 0.5f ? 1.0f : 0.0f;
			}
		}
	}

	bb::NeuralNetDenseConvolution<>			layer0_conv(1, 28, 28, 32, 3, 3);
	bb::NeuralNetBatchNormalization<>		layer0_norm(32 * 26 * 26);
	bb::NeuralNetReLU<>						layer0_act(32 * 26 * 26);
	bb::NeuralNetDropout<>					layer0_dropout(32 * 26 * 26, 0.5, 1);
	bb::NeuralNetDenseConvolution<>			layer1_conv(32, 26, 26, 32, 3, 3);
	bb::NeuralNetBatchNormalization<>		layer1_norm(32 * 24 * 24);
	bb::NeuralNetReLU<>						layer1_act(32 * 24 * 24);
	bb::NeuralNetDropout<>					layer1_dropout(32 * 24 * 24, 0.5, 2);
	bb::NeuralNetMaxPooling<>				layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetDenseConvolution<>			layer3_conv(32, 12, 12, 32, 3, 3);
	bb::NeuralNetBatchNormalization<>		layer3_norm(32 * 10 * 10);
	bb::NeuralNetReLU<>						layer3_act(32 * 10 * 10);
	bb::NeuralNetDropout<>					layer3_dropout(32 * 10 * 10, 0.5, 3);
	bb::NeuralNetDenseConvolution<>			layer4_conv(32, 10, 10, 32, 3, 3);
	bb::NeuralNetBatchNormalization<>		layer4_norm(32 * 8 * 8);
	bb::NeuralNetReLU<>						layer4_act(32 * 8 * 8);
	bb::NeuralNetDropout<>					layer4_dropout(32 * 8 * 8, 0.5, 4);
	bb::NeuralNetMaxPooling<>				layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetDenseAffine<>				layer6_affine(32 * 4 * 4, 256);
	bb::NeuralNetBatchNormalization<>		layer6_norm(256);
	bb::NeuralNetReLU<>						layer6_act(256);
	bb::NeuralNetDropout<>					layer6_dropout(256, 0.5, 5);
	bb::NeuralNetDenseAffine<>				layer7_affine(256, 10);
	bb::NeuralNetBatchNormalization<>		layer7_norm(10);
	bb::NeuralNetReLU<>						layer7_act(10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer0_norm);
	net.AddLayer(&layer0_act);
	net.AddLayer(&layer0_dropout);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer1_norm);
	net.AddLayer(&layer1_act);
	net.AddLayer(&layer1_dropout);
	net.AddLayer(&layer2_maxpol);
	net.AddLayer(&layer3_conv);
	net.AddLayer(&layer3_norm);
	net.AddLayer(&layer3_act);
	net.AddLayer(&layer3_dropout);
	net.AddLayer(&layer4_conv);
	net.AddLayer(&layer4_norm);
	net.AddLayer(&layer4_act);
	net.AddLayer(&layer4_dropout);
	net.AddLayer(&layer5_maxpol);
	net.AddLayer(&layer6_affine);
	net.AddLayer(&layer6_norm);
	net.AddLayer(&layer6_act);
	net.AddLayer(&layer6_dropout);
	net.AddLayer(&layer7_affine);
	net.AddLayer(&layer7_norm);
	net.AddLayer(&layer7_act);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true, true);
}


void MnistDenseSimpleConvolution5x5(int epoc_size, size_t max_batch_size, bool binary_mode, bool binarize_input)
{
	// run name
	std::string run_name = "MnistDenseSimpleConvolution5x5";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	if (binarize_input) {
		std::cout << "binarize input data" << std::endl;
		for (auto& vec_x : td.x_train) {
			for (auto& x : vec_x) {
				x = x > 0.5f ? 1.0f : 0.0f;
			}
		}
		for (auto& vec_x : td.x_test) {
			for (auto& x : vec_x) {
				x = x > 0.5f ? 1.0f : 0.0f;
			}
		}
	}

	bb::NeuralNetDenseConvolution<>			layer0_conv(1, 28, 28, 32, 5, 5);
	bb::NeuralNetBatchNormalization<>		layer0_norm(32 * 24 * 24);
	bb::NeuralNetReLU<>						layer0_act(32 * 24 * 24);
	bb::NeuralNetDropout<>					layer0_dropout(32 * 24 * 24, 0.5, 1);
	bb::NeuralNetMaxPooling<>				layer1_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetDenseConvolution<>			layer2_conv(32, 12, 12, 32, 5, 5);
	bb::NeuralNetBatchNormalization<>		layer2_norm(32 * 8 * 8);
	bb::NeuralNetReLU<>						layer2_act(32 * 8 * 8);
	bb::NeuralNetDropout<>					layer2_dropout(32 * 8 * 8, 0.5, 3);
	bb::NeuralNetMaxPooling<>				layer3_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetDenseAffine<>				layer4_affine(32 * 4 * 4, 256);
	bb::NeuralNetBatchNormalization<>		layer4_norm(256);
	bb::NeuralNetReLU<>						layer4_act(256);
	bb::NeuralNetDropout<>					layer4_dropout(256, 0.5, 5);
	bb::NeuralNetDenseAffine<>				layer5_affine(256, 10);
	bb::NeuralNetBatchNormalization<>		layer5_norm(10);
	bb::NeuralNetReLU<>						layer5_act(10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer0_norm);
	net.AddLayer(&layer0_act);
	net.AddLayer(&layer0_dropout);
	net.AddLayer(&layer1_maxpol);
	net.AddLayer(&layer2_conv);
	net.AddLayer(&layer2_norm);
	net.AddLayer(&layer2_act);
	net.AddLayer(&layer2_dropout);
	net.AddLayer(&layer3_maxpol);
	net.AddLayer(&layer4_affine);
	net.AddLayer(&layer4_norm);
	net.AddLayer(&layer4_act);
	net.AddLayer(&layer4_dropout);
	net.AddLayer(&layer5_affine);
	net.AddLayer(&layer5_norm);
	net.AddLayer(&layer5_act);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true, true);
}


// Kerasのサンプルと揃えてみる
void MnistDenseSimpleConvolution3x3(int epoc_size, size_t max_batch_size, bool binary_mode, bool binarize_input, std::string name)
{
	// run name
	std::string run_name = name; //  "MnistDenseSimpleConvolution3x3";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	if (binarize_input) {
		std::cout << "binarize input data" << std::endl;
		for (auto& vec_x : td.x_train) {
			for (auto& x : vec_x) {
				x = x > 0.5f ? 1.0f : 0.0f;
			}
		}
		for (auto& vec_x : td.x_test) {
			for (auto& x : vec_x) {
				x = x > 0.5f ? 1.0f : 0.0f;
			}
		}
	}

	bb::NeuralNetDenseConvolution<>			layer0_conv(1, 28, 28, 32, 3, 3);
	bb::NeuralNetBatchNormalization<>		layer0_norm(32 * 26 * 26);
	bb::NeuralNetReLU<>						layer0_act(32 * 26 * 26);
//	bb::NeuralNetDropout<>					layer0_dropout(32 * 26 * 26, 0.5, 1);
	bb::NeuralNetDenseConvolution<>			layer1_conv(32, 26, 26, 64, 3, 3);
	bb::NeuralNetBatchNormalization<>		layer1_norm(64 * 24 * 24);
	bb::NeuralNetReLU<>						layer1_act(64 * 24 * 24);
	bb::NeuralNetMaxPooling<>				layer2_maxpol(64, 24, 24, 2, 2);
	bb::NeuralNetDropout<>					layer2_dropout(64 * 12 * 12, 0.25, 1);
	bb::NeuralNetDenseAffine<>				layer3_affine(64 * 12 * 12, 256);
	bb::NeuralNetBatchNormalization<>		layer3_norm(256);
	bb::NeuralNetReLU<>						layer3_act(256);
	bb::NeuralNetDropout<>					layer3_dropout(256, 0.5, 5);
	bb::NeuralNetDenseAffine<>				layer4_affine(256, 10);
	bb::NeuralNetBatchNormalization<>		layer4_norm(10);
	bb::NeuralNetReLU<>						layer4_act(10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer0_norm);
	net.AddLayer(&layer0_act);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer1_norm);
	net.AddLayer(&layer1_act);
	net.AddLayer(&layer2_maxpol);
//	net.AddLayer(&layer2_dropout);
	net.AddLayer(&layer3_affine);
	net.AddLayer(&layer3_norm);
	net.AddLayer(&layer3_act);
//	net.AddLayer(&layer3_dropout);
	net.AddLayer(&layer4_affine);
	if (binary_mode) {
		net.AddLayer(&layer4_norm);
		net.AddLayer(&layer4_act);
	}

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true, true);
}



// MNIST CNN with LUT networks
void MnistDenseFullyCnn(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistDenseFullyCnn";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	// sub-networks for convolution(3x3)
	bb::NeuralNetDenseAffine<>	sub0_affine(1 * 3 * 3, 32);
//	bb::NeuralNetSigmoid<>		sub0_act(32);
	bb::NeuralNetGroup<>		sub0_net;
	sub0_net.AddLayer(&sub0_affine);
//	sub0_net.AddLayer(&sub0_act);

	// sub-networks for convolution(3x3)
	bb::NeuralNetDenseAffine<>	sub1_affine(32 * 3 * 3, 32);
//	bb::NeuralNetSigmoid<>		sub1_act(32);
	bb::NeuralNetGroup<>		sub1_net;
	sub1_net.AddLayer(&sub1_affine);
//	sub1_net.AddLayer(&sub1_act);

	// sub-networks for convolution(3x3)
	bb::NeuralNetDenseAffine<>	sub3_affine(32 * 3 * 3, 32);
//	bb::NeuralNetSigmoid<>		sub3_act(32);
	bb::NeuralNetGroup<>		sub3_net;
	sub3_net.AddLayer(&sub3_affine);
//	sub3_net.AddLayer(&sub3_act);

	// sub-networks for convolution(3x3)
	bb::NeuralNetDenseAffine<>	sub4_affine(32 * 3 * 3, 32);
//	bb::NeuralNetSigmoid<>		sub4_act(32);
	bb::NeuralNetGroup<>		sub4_net;
	sub4_net.AddLayer(&sub4_affine);
//	sub4_net.AddLayer(&sub4_act);

	// sub-networks for convolution(3x3)
	bb::NeuralNetDenseAffine<>	sub6_affine(32 * 3 * 3, 32);
//	bb::NeuralNetSigmoid<>		sub6_act(32);
	bb::NeuralNetGroup<>		sub6_net;
	sub6_net.AddLayer(&sub6_affine);
//	sub6_net.AddLayer(&sub6_act);

	// sub-networks for convolution(3x3)
	bb::NeuralNetDenseAffine<>	sub7_affine(32 * 2 * 2, 10);
//	bb::NeuralNetSigmoid<>		sub7_act(10);
	bb::NeuralNetGroup<>		sub7_net;
	sub7_net.AddLayer(&sub7_affine);
//	sub7_net.AddLayer(&sub7_act);


	bb::NeuralNetLoweringConvolution<>		layer0_conv(&sub0_net, 1, 28, 28, 32, 3, 3);
//	bb::NeuralNetDenseConvolution<>			layer0_conv(1, 28, 28, 32, 3, 3);
	bb::NeuralNetReLU<>						layer0_act(32 * 26 * 26);
	bb::NeuralNetLoweringConvolution<>		layer1_conv(&sub1_net, 32, 26, 26, 32, 3, 3);
//	bb::NeuralNetDenseConvolution<>			layer1_conv(32, 26, 26, 32, 3, 3);
	bb::NeuralNetReLU<>						layer1_act(32 * 24 * 24);
	bb::NeuralNetMaxPooling<>				layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<>		layer3_conv(&sub3_net, 32, 12, 12, 32, 3, 3);
//	bb::NeuralNetDenseConvolution<>			layer3_conv(32, 12, 12, 32, 3, 3);
	bb::NeuralNetReLU<>						layer3_act(32 * 10 * 10);
	bb::NeuralNetLoweringConvolution<>		layer4_conv(&sub4_net, 32, 10, 10, 32, 3, 3);
//	bb::NeuralNetDenseConvolution<>			layer4_conv(32, 10, 10, 32, 3, 3);
	bb::NeuralNetReLU<>						layer4_act(32 * 8 * 8);
	bb::NeuralNetMaxPooling<>				layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetLoweringConvolution<>		layer6_conv(&sub6_net, 32, 4, 4, 32, 3, 3);
//	bb::NeuralNetDenseConvolution<>			layer6_conv(32, 4, 4, 32, 3, 3);
	bb::NeuralNetReLU<>						layer6_act(32 * 2 * 2);
	bb::NeuralNetLoweringConvolution<>		layer7_conv(&sub7_net, 32, 2, 2, 10, 2, 2);
//	bb::NeuralNetDenseConvolution<>			layer7_conv(32, 2, 2, 10, 2, 2);
//	bb::NeuralNetReLU<>						layer7_act(10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer0_act);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer1_act);
	net.AddLayer(&layer2_maxpol);
	net.AddLayer(&layer3_conv);
	net.AddLayer(&layer3_act);
	net.AddLayer(&layer4_conv);
	net.AddLayer(&layer4_act);
	net.AddLayer(&layer5_maxpol);
	net.AddLayer(&layer6_conv);
	net.AddLayer(&layer6_act);
	net.AddLayer(&layer7_conv);
//	net.AddLayer(&layer7_act);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true, true);
}



void WriteTestImage(void)
{
	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	const int w = 640 / 4;
	const int h = 480 / 4;

	unsigned char img[h][w];
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			int idx = (y / 28) * (w / 28) + (x / 28);
			int xx = x % 28;
			int yy = y % 28;
			img[y][x] = (unsigned char)(td.x_test[idx][yy * 28 + xx] * 255.0f);
		}
	}

	{
		std::ofstream ofs("mnist_test.pgm");
		ofs << "P2" << std::endl;
		ofs << w << " " << h << std::endl;
		ofs << "255" << std::endl;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				ofs << (int)img[y][x] << std::endl;
			}
		}
	}

	{
		std::ofstream ofs("mnist_test.ppm");
		ofs << "P3" << std::endl;
		ofs << w << " " << h << std::endl;
		ofs << "255" << std::endl;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				ofs << (int)img[y][x] << " " << (int)img[y][x] << " " << (int)img[y][x] << std::endl;
			}
		}
	}
}


std::vector<int> make_random_order(int size, std::uint64_t seed = 1)
{
	std::vector<int>	order(size);
	for (int i = 0; i < size; ++i) {
		order[i] = i;
	}

	std::mt19937_64 mt(seed);
	std::uniform_int_distribution<int> distribution(0, size - 1);
	for (int i = 0; i < size; ++i) {
		std::swap(order[i], order[distribution(mt)]);
	}

	return order;
}



void bin_net_cmp(bb::NeuralNetLayer<>& org_net, bb::NeuralNetLayer<>& cpy_net)
{
	// node単位計算比較
	for (int n = 0; n < (int)org_net.GetOutputNodeSize(); ++n) {
		for (int i = 0; i < 64; i++) {
			std::vector<float> vec_in(6);
			for (int j = 0; j < 6; ++j) {
				vec_in[j] = ((i >> j) & 1) ? 1.0f : 0.0f;
			}
			auto org_val = org_net.CalcNode(n, vec_in);
			auto cpy_val = cpy_net.CalcNode(n, vec_in);

			if (org_val[0] != cpy_val[0]) {
				std::cout << "node NG : " << n << ":" << org_val[0] << " : " << cpy_val[0] << std::endl;
			}
		}
	}

	org_net.SetBatchSize(2);
	cpy_net.SetBatchSize(2);

	org_net.SetInputSignalBuffer(org_net.CreateInputSignalBuffer());
	org_net.SetInputErrorBuffer(org_net.CreateInputErrorBuffer());
	org_net.SetOutputSignalBuffer(org_net.CreateOutputSignalBuffer());
	org_net.SetOutputErrorBuffer(org_net.CreateOutputErrorBuffer());

	cpy_net.SetInputSignalBuffer(cpy_net.CreateInputSignalBuffer());
	cpy_net.SetInputErrorBuffer(cpy_net.CreateInputErrorBuffer());
	cpy_net.SetOutputSignalBuffer(cpy_net.CreateOutputSignalBuffer());
	cpy_net.SetOutputErrorBuffer(cpy_net.CreateOutputErrorBuffer());

	auto org_in_buf = org_net.GetInputSignalBuffer();
	auto cpy_in_buf = cpy_net.GetInputSignalBuffer();
	std::mt19937_64 mt(1);
	size_t input_node_size = org_in_buf.GetNodeSize();
	for (size_t i = 0; i < input_node_size; ++i) {
		bool v = ((mt() % 2) == 0);
		org_in_buf.SetBinary(0, i, v);
		cpy_in_buf.SetBinary(0, i, v);
	}

	org_net.Forward(false);
	cpy_net.Forward(false);

	auto org_out_buf = org_net.GetOutputSignalBuffer();
	auto cpy_out_buf = cpy_net.GetOutputSignalBuffer();
	size_t output_node_size = org_out_buf.GetNodeSize();
	for (size_t i = 0; i < output_node_size; ++i) {
		bool v0 = org_out_buf.GetBinary(0, i);
		bool v1 = cpy_out_buf.GetBinary(0, i);
		if (v0 != v1) {
			std::cout << "forward NG : " << i << " : " << v0 << " : " << v1 << std::endl;
		}
	}

}

void WriteVerilogData(std::string fname, bb::NeuralNetBuffer<>& buf)
{
	std::ofstream ofs(fname);
	int frame_size = (int)buf.GetFrameSize();
	int node_size = (int)buf.GetNodeSize();

	for (int frame = 0; frame < frame_size; ++frame) {
		for (int node = node_size-1; node >= 0; --node) {
			if (buf.GetBinary(frame, node)) {
				ofs << "1";
			}
			else {
				ofs << "0";
			}
		}
		ofs << std::endl;
	}
}



// MNIST Multilayer perceptron with LUT networks
void MnistMlpLutMini(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistMlpLutMini";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseMicroMlpDiscrete<6, 32>	layer0_lut(28 * 28, 1080);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 32>	layer1_lut(1080, 180);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 32>	layer2_lut(180, 30);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_lut);
	net.AddLayer(&layer1_lut);
	net.AddLayer(&layer2_lut);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
#if 1
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true);
#else
	{
		std::ifstream ifs("MnistMlpLut_net.json");
		if (ifs.is_open()) {
			cereal::JSONInputArchive ar(ifs);
			int epoc;
			ar(cereal::make_nvp("epoc", epoc));
			net.Load(ar);
		}
	}
#endif

	// convert FPGA model
	{
		// build binary network
		bb::NeuralNetRealToBinary<>	bin_input_real2bin(28 * 28, 28 * 28);
		bb::NeuralNetBinaryLut6<>	bin_layer0_lut(28 * 28, 1080);
		bb::NeuralNetBinaryLut6<>	bin_layer1_lut(1080, 180);
		bb::NeuralNetBinaryLut6<>	bin_layer2_lut(180, 30);
		bb::NeuralNetBinaryToReal<>	bin_output_bin2real(30, 10);

		bb::NeuralNet<> bin_net;
		bin_net.AddLayer(&bin_input_real2bin);
		bin_net.AddLayer(&bin_layer0_lut);
		bin_net.AddLayer(&bin_layer1_lut);
		bin_net.AddLayer(&bin_layer2_lut);
		bin_net.AddLayer(&bin_output_bin2real);

		// Accuracy Function
		bb::NeuralNetAccuracyCategoricalClassification<>	bin_acc_func(num_class);

		// parameter copy
		std::cout << "parameter copy" << std::endl;
		bin_layer0_lut.ImportLayer(layer0_lut);
		bin_layer1_lut.ImportLayer(layer1_lut);
		bin_layer2_lut.ImportLayer(layer2_lut);

		bin_net_cmp(layer0_lut, bin_layer0_lut);
		bin_net_cmp(layer1_lut, bin_layer1_lut);
		bin_net_cmp(layer2_lut, bin_layer2_lut);

		auto test_accuracy = net.RunCalculation(td.x_test, td.y_test, max_batch_size, 0, &bin_acc_func);
		std::cout << "test_accuracy : " << test_accuracy << std::endl;
		auto train_accuracy = net.RunCalculation(td.x_train, td.y_train, max_batch_size, 0, &bin_acc_func);
		std::cout << "train_accuracy : " << train_accuracy << std::endl;

		// evaluation
		//		bin_mux.SetMuxSize(1);
		auto bin_test_accuracy = bin_net.RunCalculation(td.x_test, td.y_test, max_batch_size, 0, &bin_acc_func);
		std::cout << "bin_test_accuracy : " << bin_test_accuracy << std::endl;
		auto bin_train_accuracy = bin_net.RunCalculation(td.x_train, td.y_train, max_batch_size, 0, &bin_acc_func);
		std::cout << "bin_train_accuracy : " << bin_train_accuracy << std::endl;

		// Write RTL
		std::ofstream ofs(run_name + ".v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer0_lut, "lutnet_layer0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer1_lut, "lutnet_layer1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer2_lut, "lutnet_layer2");


		// write test bench
		td = bb::LoadMnist<>::Load();
		bin_net.RunCalculation(td.x_test, td.y_test, max_batch_size, 0, &bin_acc_func);

		// バッチサイズ設定
		bin_net.SetBatchSize(256);

		auto in_sig_buf = bin_net.GetInputSignalBuffer();
		auto out_sig_buf = bin_net.GetOutputSignalBuffer();

		// データ格納
		for ( size_t frame = 0; frame < 256; ++frame) {
			for (size_t node = 0; node < 28*28; ++node) {
				in_sig_buf.SetReal(frame, node, td.x_test[frame][node]);
			}
		}

		// 予測
		bin_net.Forward(false);
		

		// RTLデバッグ用中間データ出力
		auto input_buf = bin_layer0_lut.GetInputSignalBuffer();
		auto lut0_buf = bin_layer0_lut.GetOutputSignalBuffer();
		auto lut1_buf = bin_layer1_lut.GetOutputSignalBuffer();
		auto lut2_buf = bin_layer2_lut.GetOutputSignalBuffer();
		WriteVerilogData("mnist_mpl_mini_input.dat", input_buf);
		WriteVerilogData("mnist_mpl_mini_lut0.dat", lut0_buf);
		WriteVerilogData("mnist_mpl_mini_lut1.dat", lut1_buf);
		WriteVerilogData("mnist_mpl_mini_lut2.dat", lut2_buf);
	}
}



// MNIST Multilayer perceptron with LUT networks
void MnistMlpLut(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistMlpLut";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetRealToBinary<float>			input_real2bin(28 * 28, 28 * 28);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer0_lut(28 * 28, 8192);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer1_lut(8192, 4096);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer2_lut(4096, 1080);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer3_lut(1080, 180);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer4_lut(180, 30);
	bb::NeuralNetBinaryToReal<float>			output_bin2real(30, 10);

	int index0[4] = { 0, 1, 28 + 0, 28 + 1};
	auto order0 = make_random_order(28 * 28, 1);
	for (int i = 0; i < 8192; ++i) {
		for (int j = 0; j < 6; ++j) {
			if (i < 4096 && j < 4) {
				layer0_lut.SetNodeInput(i, j, (i + index0[j]) % (28 * 28));
			}
			else {
//				layer0_lut.SetNodeInput(i, j, order0[(i / 10 + j) % (28 * 28)]);
				layer0_lut.SetNodeInput(i, j, (i/10 + j) % (28 * 28));
			}
		}
	}
//	for (int i = 0; i < 8192; ++i) {
//		for (int j = 0; j < 6; ++j) {
//			layer0_lut.SetNodeInput(i, j, order0[(i / 10 + j) % (28 * 28)]);
//		}
//	}

	auto order1 = make_random_order(8192, 2);
	for (int i = 0; i < 4096; ++i) {
		for (int j = 0; j < 6; ++j) {
//			layer1_lut.SetNodeInput(i, j, order1[(i * 2 + j) % 8192]);
			layer1_lut.SetNodeInput(i, j, (i * 2 + j) % 8192);
		}
	}
	auto order2 = make_random_order(4096, 3);
	for (int i = 0; i < 1080; ++i) {
		for (int j = 0; j < 6; ++j) {
			layer2_lut.SetNodeInput(i, j, order2[(i * 4 + j) % 4096]);
		}
	}

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&input_real2bin);
	net.AddLayer(&layer0_lut);
	net.AddLayer(&layer1_lut);
	net.AddLayer(&layer2_lut);
	net.AddLayer(&layer3_lut);
	net.AddLayer(&layer4_lut);
	net.AddLayer(&output_bin2real);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
#if 1
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true);
#else
	{
		std::ifstream ifs("MnistMlpLut_net.json");
		if (ifs.is_open()) {
			cereal::JSONInputArchive ar(ifs);
			int epoc;
			ar(cereal::make_nvp("epoc", epoc));
			net.Load(ar);
		}
	}
#endif

	// convert FPGA model
	{
		// build binary network
		bb::NeuralNetRealToBinary<bool>	bin_input_real2bin(28 * 28, 28 * 28);
		bb::NeuralNetBinaryLut6<>	bin_layer0_lut(28 * 28, 8192);
		bb::NeuralNetBinaryLut6<>	bin_layer1_lut(8192, 4096);
		bb::NeuralNetBinaryLut6<>	bin_layer2_lut(4096, 1080);
		bb::NeuralNetBinaryLut6<>	bin_layer3_lut(1080, 180);
		bb::NeuralNetBinaryLut6<>	bin_layer4_lut(180, 30);
		bb::NeuralNetBinaryToReal<bool>	bin_output_bin2real(30, 10);

		/*
		bb::NeuralNetGroup<>		bin_mux_group;
		bin_mux_group.AddLayer(&bin_layer0_lut);
		bin_mux_group.AddLayer(&bin_layer1_lut);
		bin_mux_group.AddLayer(&bin_layer2_lut);
		bin_mux_group.AddLayer(&bin_layer3_lut);
		bin_mux_group.AddLayer(&bin_layer4_lut);
		bb::NeuralNetBinaryMultiplex<>	bin_mux(&bin_mux_group, 28 * 28, 10, 1, 3);
	
		bb::NeuralNet<> bin_net;
		bin_net.AddLayer(&bin_mux);
		*/

		bb::NeuralNet<> bin_net;
		bin_net.AddLayer(&bin_input_real2bin);
		bin_net.AddLayer(&bin_layer0_lut);
		bin_net.AddLayer(&bin_layer1_lut);
		bin_net.AddLayer(&bin_layer2_lut);
		bin_net.AddLayer(&bin_layer3_lut);
		bin_net.AddLayer(&bin_layer4_lut);
		bin_net.AddLayer(&bin_output_bin2real);

		// Accuracy Function
		bb::NeuralNetAccuracyCategoricalClassification<>	bin_acc_func(num_class);

		// parameter copy
		std::cout << "parameter copy" << std::endl;
		bin_layer0_lut.ImportLayer(layer0_lut);
		bin_layer1_lut.ImportLayer(layer1_lut);
		bin_layer2_lut.ImportLayer(layer2_lut);
		bin_layer3_lut.ImportLayer(layer3_lut);
		bin_layer4_lut.ImportLayer(layer4_lut);

		bin_net_cmp(layer0_lut, bin_layer0_lut);
		bin_net_cmp(layer1_lut, bin_layer1_lut);
		bin_net_cmp(layer2_lut, bin_layer2_lut);
		bin_net_cmp(layer3_lut, bin_layer3_lut);
		bin_net_cmp(layer4_lut, bin_layer4_lut);

		auto test_accuracy = net.RunCalculation(td.x_test, td.y_test, max_batch_size, 0, &bin_acc_func);
		std::cout << "test_accuracy : " << test_accuracy << std::endl;
		auto train_accuracy = net.RunCalculation(td.x_train, td.y_train, max_batch_size, 0, &bin_acc_func);
		std::cout << "train_accuracy : " << train_accuracy << std::endl;

		// evaluation
//		bin_mux.SetMuxSize(1);
		auto bin_test_accuracy = bin_net.RunCalculation(td.x_test, td.y_test, max_batch_size, 0, &bin_acc_func);
		std::cout << "bin_test_accuracy : " << bin_test_accuracy << std::endl;
		auto bin_train_accuracy = bin_net.RunCalculation(td.x_train, td.y_train, max_batch_size, 0, &bin_acc_func);
		std::cout << "bin_train_accuracy : " << bin_train_accuracy << std::endl;
		
		// Write RTL
		std::string rtl_fname = "lut_net_mlp.v";
		std::ofstream ofs(rtl_fname);
		bb::NeuralNetBinaryLutVerilog(ofs, bin_layer0_lut, "lutnet_layer0");
		bb::NeuralNetBinaryLutVerilog(ofs, bin_layer1_lut, "lutnet_layer1");
		bb::NeuralNetBinaryLutVerilog(ofs, bin_layer2_lut, "lutnet_layer2");
		bb::NeuralNetBinaryLutVerilog(ofs, bin_layer3_lut, "lutnet_layer3");
		bb::NeuralNetBinaryLutVerilog(ofs, bin_layer4_lut, "lutnet_layer4");
		std::cout << "write RTL : " << rtl_fname << std::endl;
	}
	std::cout << "end\n" << std::endl;
}



void ConnectShift(bb::NeuralNetSparseLayer<>& layer, size_t step, size_t stride)
{
	size_t input_size = layer.GetInputNodeSize();
	size_t output_size = layer.GetOutputNodeSize();
	for (size_t i = 0; i < output_size; ++i) {
		int n = layer.GetNodeInputSize(i);
		for (int j = 0; j < n; ++j) {
			layer.SetNodeInput(i, j, (i*step + j*stride) % input_size);
		}
	}
}

void ConnectShift6(bb::NeuralNetSparseLayer<>& layer)
{
	int table[6] = { 0, 1, 3, 6, 10, 15 };
	size_t input_size = layer.GetInputNodeSize();
	size_t output_size = layer.GetOutputNodeSize();
	for (size_t i = 0; i < output_size; ++i) {
		int n = layer.GetNodeInputSize(i);
		for (int j = 0; j < n; ++j) {
			layer.SetNodeInput(i, j, (i + table[j]) % input_size);
		}
	}
}



// MNIST Multilayer perceptron with LUT networks
void MnistMlpLut2(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistMlpLut2";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();
	bb::DataAugmentationMnist<float>(td.x_train, td.y_train, (int)td.x_train.size()*3);
	bb::DataAugmentationMnist<float>(td.x_test, td.y_test);


	// build layer
	bb::NeuralNetRealToBinary<float>			input_real2bin(28 * 28, 28 * 28);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer0_lut(28 * 28, 256);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer1_lut(256, 256);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer2_lut(256, 128);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer3_lut(128, 128);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer4_lut(128, 128);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer5_lut(128, 128);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer6_lut(128, 128);
	bb::NeuralNetSparseMicroMlpDiscrete<6, 16>	layer7_lut(128, 30);
	bb::NeuralNetBinaryToReal<float>			output_bin2real(30, 10);

//	ConnectShift6(layer0_lut);
//	ConnectShift6(layer1_lut);
//	ConnectShift6(layer2_lut);
//	ConnectShift6(layer3_lut);
//	ConnectShift6(layer4_lut);
//	ConnectShift6(layer5_lut);

//	ConnectShift(layer0_lut, 1, 1);
//	ConnectShift(layer1_lut, 3, 1);
//	ConnectShift(layer2_lut, 3, 1);
//	ConnectShift(layer3_lut, 3, 1);
//	ConnectShift(layer4_lut, 2, 1);
//	ConnectShift(layer5_lut, 1, 1);
//	ConnectShift(layer6_lut, 1, 2);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&input_real2bin);
	net.AddLayer(&layer0_lut);
	net.AddLayer(&layer1_lut);
	net.AddLayer(&layer2_lut);
	net.AddLayer(&layer3_lut);
	net.AddLayer(&layer4_lut);
	net.AddLayer(&layer5_lut);
	net.AddLayer(&layer6_lut);
	net.AddLayer(&layer7_lut);
	net.AddLayer(&output_bin2real);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true);


	// convert FPGA model
	{
		// build binary network
		bb::NeuralNetRealToBinary<bool>	bin_input_real2bin(28 * 28, 28 * 28);
		bb::NeuralNetBinaryLut6<>		bin_layer0_lut(28 * 28, 256);
		bb::NeuralNetBinaryLut6<>		bin_layer1_lut(256, 256);
		bb::NeuralNetBinaryLut6<>		bin_layer2_lut(256, 128);
		bb::NeuralNetBinaryLut6<>		bin_layer3_lut(128, 128);
		bb::NeuralNetBinaryLut6<>		bin_layer4_lut(128, 128);
		bb::NeuralNetBinaryLut6<>		bin_layer5_lut(128, 128);
		bb::NeuralNetBinaryLut6<>		bin_layer6_lut(128, 128);
		bb::NeuralNetBinaryLut6<>		bin_layer7_lut(128, 30);
		bb::NeuralNetBinaryToReal<bool>	bin_output_bin2real(30, 10);

		/*
		bb::NeuralNetGroup<>		bin_mux_group;
		bin_mux_group.AddLayer(&bin_layer0_lut);
		bin_mux_group.AddLayer(&bin_layer1_lut);
		bin_mux_group.AddLayer(&bin_layer2_lut);
		bin_mux_group.AddLayer(&bin_layer3_lut);
		bin_mux_group.AddLayer(&bin_layer4_lut);
		bb::NeuralNetBinaryMultiplex<>	bin_mux(&bin_mux_group, 28 * 28, 10, 1, 3);

		bb::NeuralNet<> bin_net;
		bin_net.AddLayer(&bin_mux);
		*/

		bb::NeuralNet<> bin_net;
		bin_net.AddLayer(&bin_input_real2bin);
		bin_net.AddLayer(&bin_layer0_lut);
		bin_net.AddLayer(&bin_layer1_lut);
		bin_net.AddLayer(&bin_layer2_lut);
		bin_net.AddLayer(&bin_layer3_lut);
		bin_net.AddLayer(&bin_layer4_lut);
		bin_net.AddLayer(&bin_layer5_lut);
		bin_net.AddLayer(&bin_layer6_lut);
		bin_net.AddLayer(&bin_layer7_lut);
		bin_net.AddLayer(&bin_output_bin2real);

		// Accuracy Function
		bb::NeuralNetAccuracyCategoricalClassification<>	bin_acc_func(num_class);

		// parameter copy
		std::cout << "parameter copy" << std::endl;
		bin_layer0_lut.ImportLayer(layer0_lut);
		bin_layer1_lut.ImportLayer(layer1_lut);
		bin_layer2_lut.ImportLayer(layer2_lut);
		bin_layer3_lut.ImportLayer(layer3_lut);
		bin_layer4_lut.ImportLayer(layer4_lut);
		bin_layer5_lut.ImportLayer(layer5_lut);
		bin_layer6_lut.ImportLayer(layer6_lut);
		bin_layer7_lut.ImportLayer(layer7_lut);

		bin_net_cmp(layer0_lut, bin_layer0_lut);
		bin_net_cmp(layer1_lut, bin_layer1_lut);
		bin_net_cmp(layer2_lut, bin_layer2_lut);
		bin_net_cmp(layer3_lut, bin_layer3_lut);
		bin_net_cmp(layer4_lut, bin_layer4_lut);
		bin_net_cmp(layer5_lut, bin_layer5_lut);
		bin_net_cmp(layer6_lut, bin_layer6_lut);
		bin_net_cmp(layer7_lut, bin_layer7_lut);

		auto test_accuracy = net.RunCalculation(td.x_test, td.y_test, max_batch_size, 0, &bin_acc_func);
		std::cout << "test_accuracy : " << test_accuracy << std::endl;
		auto train_accuracy = net.RunCalculation(td.x_train, td.y_train, max_batch_size, 0, &bin_acc_func);
		std::cout << "train_accuracy : " << train_accuracy << std::endl;

		// evaluation
		//		bin_mux.SetMuxSize(1);
		auto bin_test_accuracy = bin_net.RunCalculation(td.x_test, td.y_test, max_batch_size, 0, &bin_acc_func);
		std::cout << "bin_test_accuracy : " << bin_test_accuracy << std::endl;
		auto bin_train_accuracy = bin_net.RunCalculation(td.x_train, td.y_train, max_batch_size, 0, &bin_acc_func);
		std::cout << "bin_train_accuracy : " << bin_train_accuracy << std::endl;

		// Write RTL
		std::vector< bb::NeuralNetBinaryLut<>* > lut_layers;
		lut_layers.push_back(&bin_layer0_lut);
		lut_layers.push_back(&bin_layer1_lut);
		lut_layers.push_back(&bin_layer2_lut);
		lut_layers.push_back(&bin_layer3_lut);
		lut_layers.push_back(&bin_layer4_lut);
		lut_layers.push_back(&bin_layer5_lut);
		lut_layers.push_back(&bin_layer6_lut);
		lut_layers.push_back(&bin_layer7_lut);
		std::string rtl_fname = run_name + ".v";
		std::ofstream ofs(rtl_fname);
		bb::NeuralNetMultilayerBinaryLutVerilog(ofs, lut_layers, "lutnet_layers");
		std::cout << "write RTL : " << rtl_fname << std::endl;

		/*
		std::string rtl_fname = "lut_net_mlp2.v";
		std::ofstream ofs(rtl_fname);
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer0_lut, "lutnet_layer0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer1_lut, "lutnet_layer1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer2_lut, "lutnet_layer2");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer3_lut, "lutnet_layer3");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer4_lut, "lutnet_layer4");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer5_lut, "lutnet_layer5");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer6_lut, "lutnet_layer6");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer7_lut, "lutnet_layer7");
		std::cout << "write RTL : " << rtl_fname << std::endl;
		*/
	}
	std::cout << "end\n" << std::endl;
}



// MNIST CNN with LUT networks
void MnistCnnBin(int epoc_size, size_t mini_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistCnnBin";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();


	// ------------------------------------
	//  Binarized Sparse Mini-Mlp Networks
	// ------------------------------------

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm0(1 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm1(192, 32);
	bb::NeuralNetGroup<>				sub0_net;
	sub0_net.AddLayer(&sub0_smm0);
	sub0_net.AddLayer(&sub0_smm1);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm0(32 * 3 * 3, 512);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm1(512, 128);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm2(128, 32);
	bb::NeuralNetGroup<>		sub1_net;
	sub1_net.AddLayer(&sub1_smm0);
	sub1_net.AddLayer(&sub1_smm1);
	sub1_net.AddLayer(&sub1_smm2);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm0(32 * 3 * 3, 512);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm1(512, 128);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm2(128, 32);
	bb::NeuralNetGroup<>				sub3_net;
	sub3_net.AddLayer(&sub3_smm0);
	sub3_net.AddLayer(&sub3_smm1);
	sub3_net.AddLayer(&sub3_smm2);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm0(32 * 3 * 3, 512);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm1(512, 128);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm2(128, 32);
	bb::NeuralNetGroup<>				sub4_net;
	sub4_net.AddLayer(&sub4_smm0);
	sub4_net.AddLayer(&sub4_smm1);
	sub4_net.AddLayer(&sub4_smm2);

	bb::NeuralNetRealToBinary<float>	input_real2bin(28 * 28, 28 * 28);
	bb::NeuralNetLoweringConvolution<>	layer0_conv(&sub0_net, 1, 28, 28, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>	layer1_conv(&sub1_net, 32, 26, 26, 32, 3, 3);
	bb::NeuralNetMaxPooling<>			layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<>	layer3_conv(&sub3_net, 32, 12, 12, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>	layer4_conv(&sub4_net, 32, 10, 10, 32, 3, 3);
	bb::NeuralNetMaxPooling<>			layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer6_smm(32 * 4 * 4, 512);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer7_smm(512, 128);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer8_smm(128, 30);
	bb::NeuralNetBinaryToReal<float>	output_bin2real(30, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&input_real2bin);
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer2_maxpol);
	net.AddLayer(&layer3_conv);
	net.AddLayer(&layer4_conv);
	net.AddLayer(&layer5_maxpol);
	net.AddLayer(&layer6_smm);
	net.AddLayer(&layer7_smm);
	net.AddLayer(&layer8_smm);
	net.AddLayer(&output_bin2real);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, mini_batch_size, &acc_func, &loss_func, true, true);



	// ------------------------------------
	//  Look-up Table Networks
	// ------------------------------------

	// sub-networks for convolution(3x3)
	bb::NeuralNetBinaryLut6<>	lut_sub0_lut0(1 * 3 * 3, 192);
	bb::NeuralNetBinaryLut6<>	lut_sub0_lut1(192, 32);
	bb::NeuralNetGroup<>		lut_sub0_net;
	lut_sub0_net.AddLayer(&lut_sub0_lut0);
	lut_sub0_net.AddLayer(&lut_sub0_lut1);

	// sub-networks for convolution(3x3)
	bb::NeuralNetBinaryLut6<>	lut_sub1_lut0(32 * 3 * 3, 512);
	bb::NeuralNetBinaryLut6<>	lut_sub1_lut1(512, 128);
	bb::NeuralNetBinaryLut6<>	lut_sub1_lut2(128, 32);
	bb::NeuralNetGroup<>		lut_sub1_net;
	lut_sub1_net.AddLayer(&lut_sub1_lut0);
	lut_sub1_net.AddLayer(&lut_sub1_lut1);
	lut_sub1_net.AddLayer(&lut_sub1_lut2);

	// sub-networks for convolution(3x3)
	bb::NeuralNetBinaryLut6<>	lut_sub3_lut0(32 * 3 * 3, 512);
	bb::NeuralNetBinaryLut6<>	lut_sub3_lut1(512, 128);
	bb::NeuralNetBinaryLut6<>	lut_sub3_lut2(128, 32);
	bb::NeuralNetGroup<>		lut_sub3_net;
	lut_sub3_net.AddLayer(&lut_sub3_lut0);
	lut_sub3_net.AddLayer(&lut_sub3_lut1);
	lut_sub3_net.AddLayer(&lut_sub3_lut2);

	// sub-networks for convolution(3x3)
	bb::NeuralNetBinaryLut6<>	lut_sub4_lut0(32 * 3 * 3, 512);
	bb::NeuralNetBinaryLut6<>	lut_sub4_lut1(512, 128);
	bb::NeuralNetBinaryLut6<>	lut_sub4_lut2(128, 32);
	bb::NeuralNetGroup<>		lut_sub4_net;
	lut_sub4_net.AddLayer(&lut_sub4_lut0);
	lut_sub4_net.AddLayer(&lut_sub4_lut1);
	lut_sub4_net.AddLayer(&lut_sub4_lut2);

	bb::NeuralNetLoweringConvolution<bool>	lut_layer0_conv(&lut_sub0_net, 1, 28, 28, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<bool>	lut_layer1_conv(&lut_sub1_net, 32, 26, 26, 32, 3, 3);
	bb::NeuralNetMaxPooling<bool>			lut_layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<bool>	lut_layer3_conv(&lut_sub3_net, 32, 12, 12, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<bool>	lut_layer4_conv(&lut_sub4_net, 32, 10, 10, 32, 3, 3);
	bb::NeuralNetMaxPooling<bool>			lut_layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetBinaryLut6<>				lut_layer6_lut(32 * 4 * 4, 512);
	bb::NeuralNetBinaryLut6<>				lut_layer7_lut(512, 128);
	bb::NeuralNetBinaryLut6<>				lut_layer8_lut(128, 30);

	bb::NeuralNetGroup<>			lut_mux_group;
	lut_mux_group.AddLayer(&lut_layer0_conv);
	lut_mux_group.AddLayer(&lut_layer1_conv);
	lut_mux_group.AddLayer(&lut_layer2_maxpol);
	lut_mux_group.AddLayer(&lut_layer3_conv);
	lut_mux_group.AddLayer(&lut_layer4_conv);
	lut_mux_group.AddLayer(&lut_layer5_maxpol);
	lut_mux_group.AddLayer(&lut_layer6_lut);
	lut_mux_group.AddLayer(&lut_layer7_lut);
	lut_mux_group.AddLayer(&lut_layer8_lut);
	bb::NeuralNetBinaryMultiplex<>	lut_bin_mux(&lut_mux_group, 28 * 28, 10, 1, 3);

	// build network
	bb::NeuralNet<> lut_net;
	lut_net.AddLayer(&lut_bin_mux);

	// copy
	std::cout << "[copy] BSMM-Network -> LUT-Network" << std::endl;
	lut_sub0_lut0.ImportLayer(sub0_smm0);
	lut_sub0_lut1.ImportLayer(sub0_smm1);
	lut_sub1_lut0.ImportLayer(sub1_smm0);
	lut_sub1_lut1.ImportLayer(sub1_smm1);
	lut_sub1_lut2.ImportLayer(sub1_smm2);
	lut_sub3_lut0.ImportLayer(sub3_smm0);
	lut_sub3_lut1.ImportLayer(sub3_smm1);
	lut_sub3_lut2.ImportLayer(sub3_smm2);
	lut_sub4_lut0.ImportLayer(sub4_smm0);
	lut_sub4_lut1.ImportLayer(sub4_smm1);
	lut_sub4_lut2.ImportLayer(sub4_smm2);
	lut_layer6_lut.ImportLayer(layer6_smm);
	lut_layer7_lut.ImportLayer(layer7_smm);
	lut_layer8_lut.ImportLayer(layer8_smm);

	// Accuracy Function
	bb::NeuralNetAccuracyCategoricalClassification<>	lut_acc_func(num_class);

	// evaluation
	lut_bin_mux.SetMuxSize(1);
	auto test_accuracy = lut_net.RunCalculation(td.x_test, td.y_test, mini_batch_size, 0, &lut_acc_func);
	std::cout << "[copied LUT-Network] test_accuracy : " << test_accuracy << std::endl;
	auto train_accuracy = lut_net.RunCalculation(td.x_train, td.y_train, mini_batch_size, 0, &lut_acc_func);
	std::cout << "[copied LUT-Network] train_accuracy : " << train_accuracy << std::endl;

	// Write RTL
	std::string rtl_fname = "lut_net_cnn.v";
	std::ofstream ofs(rtl_fname);
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub0_lut0, "lutnet_layer0_sub0");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub0_lut1, "lutnet_layer0_sub1");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut0, "lutnet_layer1_sub0");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut1, "lutnet_layer1_sub1");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut2, "lutnet_layer1_sub2");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub3_lut0, "lutnet_layer3_sub0");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub3_lut1, "lutnet_layer3_sub1");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub3_lut2, "lutnet_layer3_sub2");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub4_lut0, "lutnet_layer4_sub0");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub4_lut1, "lutnet_layer4_sub1");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub4_lut2, "lutnet_layer4_sub2");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer6_lut, "lutnet_layer6");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer7_lut, "lutnet_layer7");
	bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer8_lut, "lutnet_layer8");
	std::cout << "write : " << rtl_fname << std::endl;

	std::cout << "end\n" << std::endl;
}



// MNIST CNN with LUT networks
void MnistFullyCnn(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistFullyCnn";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm0(1 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm1(192, 32);
	bb::NeuralNetGroup<>				sub0_net;
	sub0_net.AddLayer(&sub0_smm0);
	sub0_net.AddLayer(&sub0_smm1);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm0(32 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm1(192, 32);
	bb::NeuralNetGroup<>				sub1_net;
	sub1_net.AddLayer(&sub1_smm0);
	sub1_net.AddLayer(&sub1_smm1);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm0(32 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm1(192, 32);
	bb::NeuralNetGroup<>				sub3_net;
	sub3_net.AddLayer(&sub3_smm0);
	sub3_net.AddLayer(&sub3_smm1);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm0(32 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm1(192, 32);
	bb::NeuralNetGroup<>				sub4_net;
	sub4_net.AddLayer(&sub4_smm0);
	sub4_net.AddLayer(&sub4_smm1);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub6_smm0(32 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub6_smm1(192, 32);
	bb::NeuralNetGroup<>				sub6_net;
	sub6_net.AddLayer(&sub6_smm0);
	sub6_net.AddLayer(&sub6_smm1);

	// sub-networks for convolution(2x2)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub7_smm0(32 * 2 * 2, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub7_smm1(192, 30);
	bb::NeuralNetGroup<>				sub7_net;
	sub7_net.AddLayer(&sub7_smm0);
	sub7_net.AddLayer(&sub7_smm1);


	bb::NeuralNetBinaryToReal<float>		input_real2bin(28 * 28, 28 * 28);
	bb::NeuralNetLoweringConvolution<>		layer0_conv(&sub0_net, 1, 28, 28, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer1_conv(&sub1_net, 32, 26, 26, 32, 3, 3);
	bb::NeuralNetMaxPooling<>				layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<>		layer3_conv(&sub3_net, 32, 12, 12, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer4_conv(&sub4_net, 32, 10, 10, 32, 3, 3);
	bb::NeuralNetMaxPooling<>				layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetLoweringConvolution<>		layer6_conv(&sub6_net, 32, 4, 4, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer7_conv(&sub7_net, 32, 2, 2, 30, 2, 2);
	bb::NeuralNetBinaryToReal<float>		output_bin2real(30, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&input_real2bin);
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer2_maxpol);
	net.AddLayer(&layer3_conv);
	net.AddLayer(&layer4_conv);
	net.AddLayer(&layer5_maxpol);
	net.AddLayer(&layer6_conv);
	net.AddLayer(&layer7_conv);
	net.AddLayer(&output_bin2real);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true);


	// copy to LUT Network
	{
		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	lut_sub0_lut0(1 * 3 * 3, 192);
		bb::NeuralNetBinaryLut6<>	lut_sub0_lut1(192, 32);
		bb::NeuralNetGroup<>		lut_sub0_net;
		lut_sub0_net.AddLayer(&lut_sub0_lut0);
		lut_sub0_net.AddLayer(&lut_sub0_lut1);

		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	lut_sub1_lut0(32 * 3 * 3, 192);
		bb::NeuralNetBinaryLut6<>	lut_sub1_lut1(192, 32);
		bb::NeuralNetGroup<>		lut_sub1_net;
		lut_sub1_net.AddLayer(&lut_sub1_lut0);
		lut_sub1_net.AddLayer(&lut_sub1_lut1);

		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	lut_sub3_lut0(32 * 3 * 3, 192);
		bb::NeuralNetBinaryLut6<>	lut_sub3_lut1(192, 32);
		bb::NeuralNetGroup<>		lut_sub3_net;
		lut_sub3_net.AddLayer(&lut_sub3_lut0);
		lut_sub3_net.AddLayer(&lut_sub3_lut1);

		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	lut_sub4_lut0(32 * 3 * 3, 192);
		bb::NeuralNetBinaryLut6<>	lut_sub4_lut1(192, 32);
		bb::NeuralNetGroup<>		lut_sub4_net;
		lut_sub4_net.AddLayer(&lut_sub4_lut0);
		lut_sub4_net.AddLayer(&lut_sub4_lut1);

		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	lut_sub6_lut0(32 * 3 * 3, 192);
		bb::NeuralNetBinaryLut6<>	lut_sub6_lut1(192, 32);
		bb::NeuralNetGroup<>		lut_sub6_net;
		lut_sub6_net.AddLayer(&lut_sub6_lut0);
		lut_sub6_net.AddLayer(&lut_sub6_lut1);

		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	lut_sub7_lut0(32 * 2 * 2, 180);
		bb::NeuralNetBinaryLut6<>	lut_sub7_lut1(180, 30);
		bb::NeuralNetGroup<>		lut_sub7_net;
		lut_sub7_net.AddLayer(&lut_sub7_lut0);
		lut_sub7_net.AddLayer(&lut_sub7_lut1);

		bb::NeuralNetLoweringConvolution<bool>	lut_layer0_conv(&lut_sub0_net, 1, 28, 28, 32, 3, 3);
		bb::NeuralNetLoweringConvolution<bool>	lut_layer1_conv(&lut_sub1_net, 32, 26, 26, 32, 3, 3);
		bb::NeuralNetMaxPooling<bool>			lut_layer2_maxpol(32, 24, 24, 2, 2);
		bb::NeuralNetLoweringConvolution<bool>	lut_layer3_conv(&lut_sub3_net, 32, 12, 12, 32, 3, 3);
		bb::NeuralNetLoweringConvolution<bool>	lut_layer4_conv(&lut_sub4_net, 32, 10, 10, 32, 3, 3);
		bb::NeuralNetMaxPooling<bool>			lut_layer5_maxpol(32, 8, 8, 2, 2);
		bb::NeuralNetLoweringConvolution<bool>	lut_layer6_conv(&lut_sub6_net, 32, 4, 4, 32, 3, 3);
		bb::NeuralNetLoweringConvolution<bool>	lut_layer7_conv(&lut_sub7_net, 32, 2, 2, 30, 2, 2);

		bb::NeuralNetGroup<>		lut_mux_group;
		lut_mux_group.AddLayer(&lut_layer0_conv);
		lut_mux_group.AddLayer(&lut_layer1_conv);
		lut_mux_group.AddLayer(&lut_layer2_maxpol);
		lut_mux_group.AddLayer(&lut_layer3_conv);
		lut_mux_group.AddLayer(&lut_layer4_conv);
		lut_mux_group.AddLayer(&lut_layer5_maxpol);
		lut_mux_group.AddLayer(&lut_layer6_conv);
		lut_mux_group.AddLayer(&lut_layer7_conv);
		bb::NeuralNetBinaryMultiplex<>	lut_bin_mux(&lut_mux_group, 28 * 28, 10, 1, 3);

		// build network
		bb::NeuralNet<> lut_net;
		lut_net.AddLayer(&lut_bin_mux);

		// copy
		std::cout << "[parameter copy]" << std::endl;
		lut_sub0_lut0.ImportLayer(sub0_smm0);
		lut_sub0_lut1.ImportLayer(sub0_smm1);
		lut_sub1_lut0.ImportLayer(sub1_smm0);
		lut_sub1_lut1.ImportLayer(sub1_smm1);
		lut_sub3_lut0.ImportLayer(sub3_smm0);
		lut_sub3_lut1.ImportLayer(sub3_smm1);
		lut_sub4_lut0.ImportLayer(sub4_smm0);
		lut_sub4_lut1.ImportLayer(sub4_smm1);
		lut_sub6_lut0.ImportLayer(sub6_smm0);
		lut_sub6_lut1.ImportLayer(sub6_smm1);
		lut_sub7_lut0.ImportLayer(sub7_smm0);
		lut_sub7_lut1.ImportLayer(sub7_smm1);

		// Accuracy Function
		bb::NeuralNetAccuracyCategoricalClassification<>	lut_acc_func(num_class);

		// evaluation
		lut_bin_mux.SetMuxSize(1);
		auto test_accuracy = lut_net.RunCalculation(td.x_test, td.y_test, max_batch_size, 0, &lut_acc_func);
		std::cout << "copy test_accuracy : " << test_accuracy << std::endl;
		auto train_accuracy = lut_net.RunCalculation(td.x_train, td.y_train, max_batch_size, 0, &lut_acc_func);
		std::cout << "copy train_accuracy : " << train_accuracy << std::endl;

		// Write RTL
		std::ofstream ofs("lut_net_fully_cnn.v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub0_lut0, "lutnet_layer0_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub0_lut1, "lutnet_layer0_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut0, "lutnet_layer1_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut1, "lutnet_layer1_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub3_lut0, "lutnet_layer3_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub3_lut1, "lutnet_layer3_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub4_lut0, "lutnet_layer4_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub4_lut1, "lutnet_layer4_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub6_lut0, "lutnet_layer6_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub6_lut1, "lutnet_layer6_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub6_lut0, "lutnet_layer7_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub6_lut1, "lutnet_layer7_sub0");
	}
}


void MnistFullyCnn2(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistFullyCnn2";
	int			num_class = 10;

	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm0(1 * 3 * 3, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm1(1024, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm2(1024, 384);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm3(384, 64);
	bb::NeuralNetGroup<>				sub0_net;
	sub0_net.AddLayer(&sub0_smm0);
	sub0_net.AddLayer(&sub0_smm1);
	sub0_net.AddLayer(&sub0_smm2);
	sub0_net.AddLayer(&sub0_smm3);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm0(64 * 3 * 3, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm1(1024, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm2(1024, 384);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm3(384, 64);
	bb::NeuralNetGroup<>				sub1_net;
	sub1_net.AddLayer(&sub1_smm0);
	sub1_net.AddLayer(&sub1_smm1);
	sub1_net.AddLayer(&sub1_smm2);
	sub1_net.AddLayer(&sub1_smm3);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm0(64 * 3 * 3, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm1(1024, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm2(1024, 384);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm3(384, 64);
	bb::NeuralNetGroup<>				sub3_net;
	sub3_net.AddLayer(&sub3_smm0);
	sub3_net.AddLayer(&sub3_smm1);
	sub3_net.AddLayer(&sub3_smm2);
	sub3_net.AddLayer(&sub3_smm3);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm0(64 * 3 * 3, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm1(1024, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm2(1024, 384);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm3(384, 64);
	bb::NeuralNetGroup<>				sub4_net;
	sub4_net.AddLayer(&sub4_smm0);
	sub4_net.AddLayer(&sub4_smm1);
	sub4_net.AddLayer(&sub4_smm2);
	sub4_net.AddLayer(&sub4_smm3);

	// sub-networks for convolution(3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub6_smm0(64 * 3 * 3, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub6_smm1(1024, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub6_smm2(1024, 384);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub6_smm3(384, 60);
	bb::NeuralNetGroup<>				sub6_net;
	sub6_net.AddLayer(&sub6_smm0);
	sub6_net.AddLayer(&sub6_smm1);
	sub6_net.AddLayer(&sub6_smm2);
	sub6_net.AddLayer(&sub6_smm3);

	bb::NeuralNetBinaryToReal<float>		input_real2bin(28 * 28, 28 * 28);
	bb::NeuralNetLoweringConvolution<>		layer0_conv(&sub0_net, 1, 28, 28, 64, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer1_conv(&sub1_net, 64, 26, 26, 64, 3, 3);
	bb::NeuralNetMaxPooling<>				layer2_maxpol(64, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<>		layer3_conv(&sub3_net, 64, 12, 12, 64, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer4_conv(&sub4_net, 64, 10, 10, 64, 3, 3);
	bb::NeuralNetMaxPooling<>				layer5_maxpol(64, 8, 8, 2, 2);
	bb::NeuralNetLoweringConvolution<>		layer6_conv(&sub6_net, 64, 4, 4, 60, 3, 3);
	bb::NeuralNetMaxPooling<>				layer7_maxpol(60, 2, 2, 2, 2);
	bb::NeuralNetBinaryToReal<float>		output_bin2real(60, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&input_real2bin);
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer2_maxpol);
	net.AddLayer(&layer3_conv);
	net.AddLayer(&layer4_conv);
	net.AddLayer(&layer5_maxpol);
	net.AddLayer(&layer6_conv);
	net.AddLayer(&layer7_maxpol);
	net.AddLayer(&output_bin2real);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
#if 1
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true);
#else
	{
		std::ifstream ifs("MnistCnnLut_net.json");
		if (ifs.is_open()) {
			cereal::JSONInputArchive ar(ifs);
			int epoc;
			ar(cereal::make_nvp("epoc", epoc));
			net.Load(ar);
		}
	}
#endif

#if 0
	// clone to LUT Network
	{
		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	bin_sub0_lut0(1 * 3 * 3, 192);
		bb::NeuralNetBinaryLut6<>	bin_sub0_lut1(192, 32);
		bb::NeuralNetGroup<>		bin_sub0_net;
		bin_sub0_net.AddLayer(&bin_sub0_lut0);
		bin_sub0_net.AddLayer(&bin_sub0_lut1);

		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	bin_sub1_lut0(32 * 3 * 3, 192);
		bb::NeuralNetBinaryLut6<>	bin_sub1_lut1(192, 32);
		bb::NeuralNetGroup<>		bin_sub1_net;
		bin_sub1_net.AddLayer(&bin_sub1_lut0);
		bin_sub1_net.AddLayer(&bin_sub1_lut1);

		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	bin_sub3_lut0(32 * 3 * 3, 192);
		bb::NeuralNetBinaryLut6<>	bin_sub3_lut1(192, 32);
		bb::NeuralNetGroup<>		bin_sub3_net;
		bin_sub3_net.AddLayer(&bin_sub3_lut0);
		bin_sub3_net.AddLayer(&bin_sub3_lut1);

		// sub-networks for convolution(3x3)
		bb::NeuralNetBinaryLut6<>	bin_sub4_lut0(32 * 3 * 3, 192);
		bb::NeuralNetBinaryLut6<>	bin_sub4_lut1(192, 32);
		bb::NeuralNetGroup<>		bin_sub4_net;
		bin_sub4_net.AddLayer(&bin_sub4_lut0);
		bin_sub4_net.AddLayer(&bin_sub4_lut1);

		bb::NeuralNetLoweringConvolution<bool>	bin_layer0_conv(&bin_sub0_net, 1, 28, 28, 32, 3, 3);
		bb::NeuralNetLoweringConvolution<bool>	bin_layer1_conv(&bin_sub1_net, 32, 26, 26, 32, 3, 3);
		bb::NeuralNetMaxPooling<bool>		bin_layer2_maxpol(32, 24, 24, 2, 2);
		bb::NeuralNetLoweringConvolution<bool>	bin_layer3_conv(&bin_sub3_net, 32, 12, 12, 32, 3, 3);
		bb::NeuralNetLoweringConvolution<bool>	bin_layer4_conv(&bin_sub4_net, 32, 10, 10, 32, 3, 3);
		bb::NeuralNetMaxPooling<bool>		bin_layer5_maxpol(32, 8, 8, 2, 2);
		bb::NeuralNetBinaryLut6<>			bin_layer6_lut(32 * 4 * 4, 480);
		bb::NeuralNetBinaryLut6<>			bin_layer7_lut(480, 80);

		bb::NeuralNetGroup<>			bin_mux_group;
		bin_mux_group.AddLayer(&bin_layer0_conv);
		bin_mux_group.AddLayer(&bin_layer1_conv);
		bin_mux_group.AddLayer(&bin_layer2_maxpol);
		bin_mux_group.AddLayer(&bin_layer3_conv);
		bin_mux_group.AddLayer(&bin_layer4_conv);
		bin_mux_group.AddLayer(&bin_layer5_maxpol);
		bin_mux_group.AddLayer(&bin_layer6_lut);
		bin_mux_group.AddLayer(&bin_layer7_lut);
		bb::NeuralNetBinaryMultiplex<>	bin_bin_mux(&bin_mux_group, 28 * 28, 10, 1, 8);

		// build network
		bb::NeuralNet<> bin_net;
		bin_net.AddLayer(&bin_bin_mux);

		// copy
		std::cout << "[parameter copy]" << std::endl;
		bin_sub0_lut0.ImportLayer(sub0_smm0);
		bin_sub0_lut1.ImportLayer(sub0_smm1);
		bin_sub1_lut0.ImportLayer(sub1_smm0);
		bin_sub1_lut1.ImportLayer(sub1_smm1);
		bin_sub3_lut0.ImportLayer(sub3_smm0);
		bin_sub3_lut1.ImportLayer(sub3_smm1);
		bin_sub4_lut0.ImportLayer(sub4_smm0);
		bin_sub4_lut1.ImportLayer(sub4_smm1);
		bin_layer6_lut.ImportLayer(layer6_smm);
		bin_layer7_lut.ImportLayer(layer7_smm);

		// Accuracy Function
		bb::NeuralNetAccuracyCategoricalClassification<>	bin_acc_func(num_class);

		// evaluation
		bin_bin_mux.SetMuxSize(1);
		auto test_accuracy = bin_net.RunCalculation(td.x_test, td.y_test, max_batch_size, 0, &bin_acc_func);
		std::cout << "copy test_accuracy : " << test_accuracy << std::endl;
		auto train_accuracy = bin_net.RunCalculation(td.x_train, td.y_train, max_batch_size, 0, &bin_acc_func);
		std::cout << "copy train_accuracy : " << train_accuracy << std::endl;

		// Write RTL
		std::ofstream ofs("lut_net_cnn.v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub0_lut0, "lutnet_layer0_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub0_lut1, "lutnet_layer0_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub1_lut0, "lutnet_layer1_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub1_lut1, "lutnet_layer1_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub3_lut0, "lutnet_layer3_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub3_lut1, "lutnet_layer3_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub4_lut0, "lutnet_layer4_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub4_lut1, "lutnet_layer4_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer6_lut, "lutnet_layer6");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer7_lut, "lutnet_layer7");
	}
#endif
}



// DenseAffine Real network
void MnistDenseAffineReal(int epoc_size, size_t max_batch_size)
{
	// run name
	std::string run_name = "MnistDenseAffineReal";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();
	
	// build layer
	bb::NeuralNetDenseAffine<>	layer0_affine(28 * 28, 256);
	bb::NeuralNetReLU<>			layer0_activation(256);
	bb::NeuralNetDenseAffine<>	layer1_affine(256, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_affine);
	net.AddLayer(&layer0_activation);
	net.AddLayer(&layer1_affine);
	// loss function has softmax layer

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);		
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}

// Binary DenseAffine network
void MnistDenseAffineBinary(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistDenseAffineBinary";

	// parameter
	int			num_class       = 10;
	int			binary_mux_size = 1;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetRealToBinary<float>	input_real2bin(28 * 28, 28 * 28);
	bb::NeuralNetDenseAffine<>			layer0_affine(28 * 28, 512);
	bb::NeuralNetBatchNormalization<>	layer0_batch_norm(512);
	bb::NeuralNetSigmoid<>				layer0_activation(512);
	bb::NeuralNetDenseAffine<>			layer1_affine(512, 256);
	bb::NeuralNetBatchNormalization<>	layer1_batch_norm(256);
	bb::NeuralNetSigmoid<>				layer1_activation(256);
	bb::NeuralNetDenseAffine<>			layer2_affine(256, 30);
	bb::NeuralNetBatchNormalization<>	layer2_batch_norm(30);
	bb::NeuralNetSigmoid<>				layer2_activation(30);
	bb::NeuralNetBinaryToReal<float>	output_bin2real(30, 10);

#if 1
	// build network
	bb::NeuralNet<> net;
	if (binary_mode) {
		net.AddLayer(&input_real2bin);
	}
	net.AddLayer(&layer0_affine);
	net.AddLayer(&layer0_batch_norm);
	net.AddLayer(&layer0_activation);
	net.AddLayer(&layer1_affine);
	net.AddLayer(&layer1_batch_norm);
	net.AddLayer(&layer1_activation);
	net.AddLayer(&layer2_affine);
	net.AddLayer(&layer2_batch_norm);
	net.AddLayer(&layer2_activation);
	net.AddLayer(&output_bin2real);

#else
	bb::NeuralNetGroup<>				mux_group;
	mux_group.AddLayer(&layer0_affine);
	mux_group.AddLayer(&layer0_batch_norm);
	mux_group.AddLayer(&layer0_activation);
	mux_group.AddLayer(&layer1_affine);
	mux_group.AddLayer(&layer1_batch_norm);
	mux_group.AddLayer(&layer1_activation);
	mux_group.AddLayer(&layer2_affine);
	mux_group.AddLayer(&layer2_batch_norm);
	mux_group.AddLayer(&layer2_activation);

	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&mux_group, 28*28, 10, 1, 3);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// set multiplexing size
	bin_mux.SetMuxSize(binary_mux_size);
#endif

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;


	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}



// Sparse Affine network
void MnistSparseAffineReal(int epoc_size, size_t max_batch_size)
{
	// run name
	std::string run_name = "MnistSparseAffineReal";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseAffine<6> layer0_affine(28 * 28, 360);
	bb::NeuralNetSigmoid<>		 layer0_sigmoid(360);
	bb::NeuralNetSparseAffine<6> layer1_affine(360, 60);
	bb::NeuralNetSigmoid<>		 layer1_sigmoid(60);
	bb::NeuralNetSparseAffine<6> layer2_affine(60, 10);
	// loss function has softmax layer

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_affine);
	net.AddLayer(&layer0_sigmoid);
	net.AddLayer(&layer1_affine);
	net.AddLayer(&layer1_sigmoid);
	net.AddLayer(&layer2_affine);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}


// Binary Sparse-Affine network
void MnistSparseAffineBinary2(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistSparseAffineBinary2";
	int			num_class = 10;
	int			binary_mux_size = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetRealToBinary<float>	input_real2bin(28 * 28, 28 * 28);

	bb::NeuralNetSparseAffine<6>		layer0_affine(28 * 28, 360);
	bb::NeuralNetBatchNormalization<>	layer0_batch_norm(360);
	bb::NeuralNetSigmoid<>				layer0_activation(360);

	bb::NeuralNetSparseAffine<6>		layer1_affine(360, 60);
	bb::NeuralNetBatchNormalization<>	layer1_batch_norm(60);
	bb::NeuralNetSigmoid<>				layer1_activation(60);

	bb::NeuralNetSparseAffine<6>		layer2_affine(60, 10);
	bb::NeuralNetBatchNormalization<>	layer2_batch_norm(10);
	bb::NeuralNetSigmoid<>				layer2_activation(10);

	bb::NeuralNetBinaryToReal<float>	output_bin2real(10, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&input_real2bin);
	net.AddLayer(&layer0_affine);
	net.AddLayer(&layer0_batch_norm);
	net.AddLayer(&layer0_activation);
	net.AddLayer(&layer1_affine);
	net.AddLayer(&layer1_batch_norm);
	net.AddLayer(&layer1_activation);
	net.AddLayer(&layer2_affine);
	net.AddLayer(&layer2_batch_norm);
	net.AddLayer(&layer2_activation);
	net.AddLayer(&output_bin2real);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
//	bb::NeuralNetOptimizerSgd<> optimizer(0.02);
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}


void MnistLut2(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistLut2";
	int			num_class = 10;
	int			binary_mux_size = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
#if 1
	bb::NeuralNetSigmoid<>			input_act(28 * 28);
	bb::NeuralNetSparseMicroMlp<6>	layer0_smm(28 * 28, 360);
	bb::NeuralNetSparseMicroMlp<6>	layer1_smm(360, 60);
	bb::NeuralNetSparseMicroMlp<6>	layer2_smm(60, 10);
#else
	bb::NeuralNetSigmoid<>				input_act(28 * 28);
	bb::NeuralNetSparseAffineSigmoid<6>	layer0_lut(28 * 28, 360);
	bb::NeuralNetSparseAffineSigmoid<6>	layer1_lut(360, 60);
	bb::NeuralNetSparseAffineSigmoid<6>	layer2_lut(60, 10);
#endif

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&input_act);
	net.AddLayer(&layer0_smm);
	net.AddLayer(&layer1_smm);
	net.AddLayer(&layer2_smm);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	//	bb::NeuralNetOptimizerSgd<> optimizer(0.02);
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}



void MnistLut3(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistLut4";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	int		m = 8;
	bb::NeuralNetSigmoid<>				input_act(28 * 28);
	bb::NeuralNetSparseMicroMlp<6>		layer0_smm(28 * 28, 2160 * m);
	bb::NeuralNetSparseMicroMlp<6>		layer1_smm(2160 * m, 360 * m);
	bb::NeuralNetSparseMicroMlp<6>		layer2_smm(360 * m, 60 * m);
	bb::NeuralNetSparseMicroMlp<6>		layer3_smm(60 * m, 10 * m);

	bb::NeuralNetGroup<>	mux_group;
	mux_group.AddLayer(&input_act);
	mux_group.AddLayer(&layer0_smm);
	mux_group.AddLayer(&layer1_smm);
	mux_group.AddLayer(&layer2_smm);
	mux_group.AddLayer(&layer3_smm);
	bb::NeuralNetBinaryMultiplex<float> bin_mux(&mux_group, 28 * 28, 10, 1, m);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	//	bb::NeuralNetOptimizerSgd<> optimizer(0.02);
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	bin_mux.SetMuxSize(1);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, true);
}



void MnistLut5(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistLut7";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseMicroMlp<6>		layer0_smm(28 * 28, 38880);
	bb::NeuralNetSparseMicroMlp<6>		layer1_smm(38880, 6480);
	bb::NeuralNetSparseMicroMlp<6>		layer2_smm(6480, 1080);
	bb::NeuralNetSparseMicroMlp<6>		layer3_smm(1080, 180);
	bb::NeuralNetSparseMicroMlp<6>		layer4_smm(180, 30);
	bb::NeuralNetBinaryToReal<float>	layer5_bin2real(30, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_smm);
	net.AddLayer(&layer1_smm);
	net.AddLayer(&layer2_smm);
	net.AddLayer(&layer3_smm);
	net.AddLayer(&layer4_smm);
	net.AddLayer(&layer5_bin2real);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, true);
}


void MnistLutA(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistLutB";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseMicroMlp<6>		layer0_smm(28 * 28, 4096);
	bb::NeuralNetSparseMicroMlp<6>		layer1_smm(4096, 4096);
	bb::NeuralNetSparseMicroMlp<6>		layer2_smm(4096, 1080);
	bb::NeuralNetSparseMicroMlp<6>		layer3_smm(1080, 180);
	bb::NeuralNetSparseMicroMlp<6>		layer4_smm(180, 30);
	bb::NeuralNetBinaryToReal<float>	layer5_bin2real(30, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_smm);
	net.AddLayer(&layer1_smm);
	net.AddLayer(&layer2_smm);
	net.AddLayer(&layer3_smm);
	net.AddLayer(&layer4_smm);
	net.AddLayer(&layer5_bin2real);

#if 1
	for (int i = 0; i < 4096; i++) {
		for (int j = 0; j < 6; j++) {
			layer1_smm.GetNodeInput(i, (i + j + 4096 - 3) % 4096);
			layer2_smm.GetNodeInput(i, (i + j + 4096 - 3) % 4096);
		}
	}
#endif

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, true);
}


void MnistLutD(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistLutD";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseMicroMlp<6, 16>		layer0_smm(28 * 28, 8192);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer1_smm(8192, 4096);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer2_smm(4096, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer3_smm(1024, 600);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer4_smm(600, 100);
	bb::NeuralNetBinaryToReal<float>	layer5_bin2real(100, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_smm);
	net.AddLayer(&layer1_smm);
	net.AddLayer(&layer2_smm);
	net.AddLayer(&layer3_smm);
	net.AddLayer(&layer4_smm);
	net.AddLayer(&layer5_bin2real);

#if 0
	for (int i = 0; i < 4096; i++) {
		for (int j = 0; j < 6; j++) {
			layer1_lut.GetNodeInput(i, (i + j + 4096 - 3) % 4096);
			layer2_lut.GetNodeInput(i, (i + j + 4096 - 3) % 4096);
		}
	}
#endif

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, true);
}



void MnistLutC(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistLutC";
	int			num_class = 10;

	// load MNIST data
	auto data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseMicroMlp<6, 16>		layer0_smm(28 * 28, 8192);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer1_smm(8192, 4096);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer2_smm(4096, 1080);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer3_smm(1080, 180);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer4_smm(180, 30);
	bb::NeuralNetBinaryToReal<float>	layer5_bin2real(30, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_smm);
	net.AddLayer(&layer1_smm);
	net.AddLayer(&layer2_smm);
	net.AddLayer(&layer3_smm);
	net.AddLayer(&layer4_smm);
	net.AddLayer(&layer5_bin2real);

#if 0
	for (auto& xx : data.x_train) {
		for (auto& x : xx) {
			x = (x >= 0.5f) ? 1.0f : 0.0f;
		}
	}

	for (auto& xx : data.x_test) {
		for (auto& x : xx) {
			x = (x >= 0.5f) ? 1.0f : 0.0f;
		}
	}
#endif

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, true);
}



void MnistLutE(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistLutE";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseMicroMlp<6, 16>	layer0_smm(28 * 28, 8192*2);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer1_smm(8192 * 2, 8192);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer2_smm(8192, 4096);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer3_smm(4096, 1024);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer4_smm(1024, 600);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer5_smm(600, 100);
	bb::NeuralNetBinaryToReal<float>	layer6_bin2real(100, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_smm);
	net.AddLayer(&layer1_smm);
	net.AddLayer(&layer2_smm);
	net.AddLayer(&layer3_smm);
	net.AddLayer(&layer4_smm);
	net.AddLayer(&layer5_smm);
	net.AddLayer(&layer6_bin2real);

#if 0
	for (int i = 0; i < 4096; i++) {
		for (int j = 0; j < 6; j++) {
			layer1_lut.GetNodeInput(i, (i + j + 4096 - 3) % 4096);
			layer2_lut.GetNodeInput(i, (i + j + 4096 - 3) % 4096);
		}
	}
#endif

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, true);
}


void MnistLutF(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistLutF";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseMicroMlp<6, 64>	layer0_smm(28 * 28, 8192 * 2);
	bb::NeuralNetSparseMicroMlp<6, 64>	layer1_smm(8192 * 2, 8192);
	bb::NeuralNetSparseMicroMlp<6, 64>	layer2_smm(8192, 4096);
	bb::NeuralNetSparseMicroMlp<6, 64>	layer3_smm(4096, 2048);
	bb::NeuralNetSparseMicroMlp<6, 64>	layer4_smm(2048, 600);
	bb::NeuralNetSparseMicroMlp<6, 64>	layer5_smm(600, 100);
	bb::NeuralNetBinaryToReal<float>	layer6_bin2real(100, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_smm);
	net.AddLayer(&layer1_smm);
	net.AddLayer(&layer2_smm);
	net.AddLayer(&layer3_smm);
	net.AddLayer(&layer4_smm);
	net.AddLayer(&layer5_smm);
	net.AddLayer(&layer6_bin2real);

#if 0
	for (int i = 0; i < 4096; i++) {
		for (int j = 0; j < 6; j++) {
			layer1_lut.GetNodeInput(i, (i + j + 4096 - 3) % 4096);
			layer2_lut.GetNodeInput(i, (i + j + 4096 - 3) % 4096);
		}
	}
#endif

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, true);
}


// Binary Sparse-Affine network
void MnistSparseAffineBinary(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistSparseAffineBinary";
	int			num_class = 10;
	int			binary_mux_size = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseAffine<6>		layer0_affine(28*28, 360);
	bb::NeuralNetBatchNormalization<>	layer0_batch_norm(360);
	bb::NeuralNetSigmoid<>				layer0_activation(360);

	bb::NeuralNetSparseAffine<6>		layer1_affine(360, 60);
	bb::NeuralNetBatchNormalization<>	layer1_batch_norm(60);
	bb::NeuralNetSigmoid<>				layer1_activation(60);

	bb::NeuralNetSparseAffine<6>		layer2_affine(60, 10);
	bb::NeuralNetBatchNormalization<>	layer2_batch_norm(10);
	bb::NeuralNetSigmoid<>				layer2_activation(10);

	bb::NeuralNetGroup<>				mux_group;
	mux_group.AddLayer(&layer0_affine);
	mux_group.AddLayer(&layer0_batch_norm);
	mux_group.AddLayer(&layer0_activation);
	mux_group.AddLayer(&layer1_affine);
	mux_group.AddLayer(&layer1_batch_norm);
	mux_group.AddLayer(&layer1_activation);
	mux_group.AddLayer(&layer2_affine);
	mux_group.AddLayer(&layer2_batch_norm);
	mux_group.AddLayer(&layer2_activation);

	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&mux_group, 28 * 28, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
//	bb::NeuralNetOptimizerSgd<> optimizer(0.02);
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// set multiplexing size
	bin_mux.SetMuxSize(binary_mux_size);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}

// 
void MnistLutBinary(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistLutBinary";
	int			num_class = 10;
	int			bin_mux_size = 1;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// define layer size
	size_t		input_node_size = 28 * 28;
	size_t		output_node_size = 10;
	size_t		input_hmux_size = 1;
	size_t		output_hmux_size = 3;

	size_t		layer0_node_size = 360;
	size_t		layer1_node_size = 60 * output_hmux_size;
	size_t		layer2_node_size = output_node_size * output_hmux_size;

	// build layer
#if 1
	bb::NeuralNetSparseMicroMlp<6, 64>	layer0_smm(input_node_size*input_hmux_size, layer0_node_size);
	bb::NeuralNetSparseMicroMlp<6, 64>	layer1_smm(layer0_node_size, layer1_node_size);
	bb::NeuralNetSparseMicroMlp<6, 64>	layer2_smm(layer1_node_size, layer2_node_size);
#else
	bb::NeuralNetSparseAffineSigmoid<6>		layer0_lut(28 * 28, 360);
	bb::NeuralNetSparseAffineSigmoid<6>		layer1_lut(360, 60);
	bb::NeuralNetSparseAffineSigmoid<6>		layer2_lut(60, 10);
#endif

	bb::NeuralNetGroup<>				mux_group;
	mux_group.AddLayer(&layer0_smm);
	mux_group.AddLayer(&layer1_smm);
	mux_group.AddLayer(&layer2_smm);
	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&mux_group, 28 * 28, 10, input_hmux_size, output_hmux_size);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// set multiplexing size
	bin_mux.SetMuxSize(bin_mux_size);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &acc_func, &loss_func, true, false);
}

// LUT6入力のバイナリ版の力技学習  with BruteForce training
void MnistSparseAffineLut6(int epoc_size, size_t max_batch_size)
{
	// run name
	std::string run_name = "MnistSparseAffineLut6";
	int			num_class = 10;
	int			max_train = -1;
	int			max_test = -1;

	// define layer size
	size_t		input_node_size = 28 * 28;
	size_t		output_node_size = 10;
	size_t		input_hmux_size = 1;
	size_t		output_hmux_size = 3;

	size_t		layer0_node_size = 360;
	size_t		layer1_node_size = 60 * output_hmux_size;
	size_t		layer2_node_size = output_node_size * output_hmux_size;


#ifdef _DEBUG
	std::cout << "!!!Debug mode!!!" << std::endl;
	max_train = 100;
	max_test = 50;
	max_batch_size = 16;
#endif

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load(num_class, max_train, max_test);
	auto& x_train = train_data.x_train;
	auto& y_train = train_data.y_train;
	auto& x_test = train_data.x_test;
	auto& y_test = train_data.y_test;
	auto label_train = bb::OnehotToLabel<std::uint8_t>(y_train);
	auto label_test = bb::OnehotToLabel<std::uint8_t>(y_test);
	auto train_size = x_train.size();
	auto test_size = x_test.size();
	auto x_node_size = x_test[0].size();

	std::cout << "start : " << run_name << std::endl;

	std::mt19937_64 mt(1);

	// 学習時と評価時で多重化数(乱数を変えて複数枚通して集計できるようにする)を変える
	int train_mux_size = 1;
	int test_mux_size = 1;

	// バイナリネットのGroup作成
	bb::NeuralNetBinaryLut6<>	lut_layer0_lut(input_node_size*input_hmux_size, layer0_node_size);
	bb::NeuralNetBinaryLut6<>	lut_layer1_lut(layer0_node_size, layer1_node_size);
	bb::NeuralNetBinaryLut6<>	lut_layer2_lut(layer1_node_size, layer2_node_size);
	bb::NeuralNetGroup<>		lut_mux_group;
	lut_mux_group.AddLayer(&lut_layer0_lut);
	lut_mux_group.AddLayer(&lut_layer1_lut);
	lut_mux_group.AddLayer(&lut_layer2_lut);

	// 多重化してパッキング
	bb::NeuralNetBinaryMultiplex<>	lut_mux(&lut_mux_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

	// ネット構築
	bb::NeuralNet<> lut_net;
	lut_net.AddLayer(&lut_mux);

	// 評価関数
	bb::NeuralNetAccuracyCategoricalClassification<>	lut_acc_func(num_class);

	// 初期評価
	lut_mux.SetMuxSize(test_mux_size);	// 評価用の多重化数にスイッチ
	auto test_accuracy = lut_net.RunCalculation(train_data.x_test, train_data.y_test, max_batch_size, 0, &lut_acc_func);
	std::cout << "initial test_accuracy : " << test_accuracy << std::endl;

	// 開始時間記録
	auto start_time = std::chrono::system_clock::now();

	// 学習ループ
	for (int epoc = 0; epoc < epoc_size; ++epoc) {
		int iteration = 0;
		for (size_t train_index = 0; train_index < train_size; train_index += max_batch_size) {
			// 末尾のバッチサイズクリップ
			size_t batch_size = std::min(max_batch_size, train_size - train_index);
			if (batch_size < max_batch_size) { break; }

			// 小サイズで演算すると劣化するので末尾スキップ
			if (batch_size < max_batch_size) {
				break;
			}

			// バッチサイズ設定
			lut_mux.SetMuxSize(train_mux_size);	// 学習の多重化数にスイッチ
			lut_net.SetBatchSize(batch_size);

			// データ格納
			auto in_sig_buf = lut_net.GetInputSignalBuffer();
			for (size_t frame = 0; frame < batch_size; ++frame) {
				for (size_t node = 0; node < x_node_size; ++node) {
					in_sig_buf.Set<float>(frame, node, x_train[train_index + frame][node]);
				}
			}

			// 予測
			lut_net.Forward(true);

			// バイナリ版フィードバック(力技学習)
			while (lut_mux.Feedback(lut_mux.GetOutputOnehotLoss<std::uint8_t, 10>(label_train, train_index)))
				;
	
	//		while (bin_mux.Feedback(bin_mux.GetOutputOnehotLoss(y_train, train_index)))
	//			;
	
	//		while (bin_mux.Feedback(bin_mux.CalcLoss(y_train, train_index)))
	//			;

			// 途中評価
			lut_mux.SetMuxSize(test_mux_size);	// 評価用の多重化数にスイッチ
			auto test_accuracy = lut_net.RunCalculation(x_test, y_test, max_batch_size, 0, &lut_acc_func);

			// 進捗表示
			auto progress = train_index + batch_size;
			auto rate = progress * 100 / train_size;
			std::cout << "[" << rate << "% (" << progress << "/" << train_size << ")]";
			std::cout << "  test_accuracy : " << test_accuracy << "                  ";
			std::cout << "\r" << std::flush;
		}

		// 評価
		lut_mux.SetMuxSize(test_mux_size);	// 評価用の多重化数にスイッチ
		auto test_accuracy = lut_net.RunCalculation(x_test, y_test, max_batch_size, 0, &lut_acc_func);
		auto train_accuracy = lut_net.RunCalculation(x_train,y_train, max_batch_size, 0, &lut_acc_func);
		auto now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
		std::cout << now_time << "s " << "epoc[" << epoc << "]"
			<< "  test_accuracy : " << test_accuracy
			<< "  train_accuracy : " << train_accuracy << std::endl;

		// Shuffle
		bb::ShuffleDataSet(mt(), x_train, y_train, label_train);
	}

	{
		// Write RTL
		std::ofstream ofs("lut_net.v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer0_lut, "lutnet_layer0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer1_lut, "lutnet_layer1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer2_lut, "lutnet_layer2");
	}

	std::cout << "end\n" << std::endl;
}

// Binary-Network copy to LUT-Network
void MnistSparseAffineBinToLut(int bin_epoc_size, size_t bin_max_batch_size, int lut_epoc_size, size_t lut_max_batch_size)
{
	// parameter
	std::string run_name = "MnistSparseAffineBinToLut";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// define layer size
	size_t		input_node_size = 28 * 28;
	size_t		output_node_size = 10;
	size_t		input_hmux_size = 1;
	size_t		output_hmux_size = 3;

	size_t		layer0_node_size = 360;
	size_t		layer1_node_size = 60 * output_hmux_size;
	size_t		layer2_node_size = output_node_size * output_hmux_size;



	// -------- Binary-Network --------

	int	bin_mux_size = 3;

	// build layer
	bb::NeuralNetSparseBinaryAffine<>	bin_layer0_affine(input_node_size*input_hmux_size, layer0_node_size);
	bb::NeuralNetSparseBinaryAffine<>	bin_layer1_affine(layer0_node_size, layer1_node_size);
	bb::NeuralNetSparseBinaryAffine<>	bin_layer2_affine(layer1_node_size, layer2_node_size);

	bb::NeuralNetGroup<>				bin_mux_group;
	bin_mux_group.AddLayer(&bin_layer0_affine);
	bin_mux_group.AddLayer(&bin_layer1_affine);
	bin_mux_group.AddLayer(&bin_layer2_affine);

	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&bin_mux_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

	// build network
	bb::NeuralNet<> bin_net;
	bin_net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	bin_net.SetOptimizer(&optimizer);

	// set binary mode
	bin_net.SetBinaryMode(true);

	// set multiplexing size
	bin_mux.SetMuxSize(bin_mux_size);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			bin_loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	bin_acc_func(num_class);
	bin_net.Fitting(run_name, train_data, bin_epoc_size, bin_max_batch_size, &bin_acc_func, &bin_loss_func, true, false);



	// -------- LUT-Network --------
	
	// load MNIST data
//	train_data = bb::LoadMnist<>::Load(num_class);
	auto& x_train = train_data.x_train;
	auto& y_train = train_data.y_train;
	auto& x_test = train_data.x_test;
	auto& y_test = train_data.y_test;
	auto label_train = bb::OnehotToLabel<std::uint8_t>(y_train);
	auto label_test = bb::OnehotToLabel<std::uint8_t>(y_test);
	auto train_size = x_train.size();
	auto test_size = x_test.size();
	auto x_node_size = x_test[0].size();

	std::cout << "start : " << run_name << std::endl;

	std::mt19937_64 mt(1);

	// 学習時と評価時で多重化数(乱数を変えて複数枚通して集計できるようにする)を変える
	int lut_train_mux_size = 1;
	int lut_test_mux_size = 3;

	// バイナリネットのGroup作成
	bb::NeuralNetBinaryLut6<>	lut_layer0_lut(input_node_size*input_hmux_size, layer0_node_size);
	bb::NeuralNetBinaryLut6<>	lut_layer1_lut(layer0_node_size, layer1_node_size);
	bb::NeuralNetBinaryLut6<>	lut_layer2_lut(layer1_node_size, layer2_node_size);
	bb::NeuralNetGroup<>		lut_mux_group;
	lut_mux_group.AddLayer(&lut_layer0_lut);
	lut_mux_group.AddLayer(&lut_layer1_lut);
	lut_mux_group.AddLayer(&lut_layer2_lut);

	// 多重化してパッキング
	bb::NeuralNetBinaryMultiplex<>	lut_mux(&lut_mux_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

	// ネット構築
	bb::NeuralNet<> lut_net;
	lut_net.AddLayer(&lut_mux);

	// 評価関数
	bb::NeuralNetAccuracyCategoricalClassification<>	lut_acc_func(num_class);


	// copy
	std::cout << "[parameter copy] Binary-Neteork -> LUT-Network" << std::endl;
	lut_layer0_lut.ImportLayer(bin_layer0_affine);
	lut_layer1_lut.ImportLayer(bin_layer1_affine);
	lut_layer2_lut.ImportLayer(bin_layer2_affine);
	

	// 初期評価
	lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
	auto test_accuracy = lut_net.RunCalculation(train_data.x_test, train_data.y_test, lut_max_batch_size, 0, &lut_acc_func);
	std::cout << "initial test_accuracy : " << test_accuracy << std::endl;

	// 開始時間記録
	auto start_time = std::chrono::system_clock::now();

	// 学習ループ
	for (int epoc = 0; epoc < lut_epoc_size; ++epoc) {
		int iteration = 0;
		for (size_t train_index = 0; train_index < train_size; train_index += lut_max_batch_size) {
			// 末尾のバッチサイズクリップ
			size_t batch_size = std::min(lut_max_batch_size, train_size - train_index);
			if (batch_size < lut_max_batch_size) { break; }

			// 小サイズで演算すると劣化するので末尾スキップ
			if (batch_size < lut_max_batch_size) {
				break;
			}

			// バッチサイズ設定
			lut_mux.SetMuxSize(lut_train_mux_size);	// 学習の多重化数にスイッチ
			lut_net.SetBatchSize(batch_size);

			// データ格納
			auto in_sig_buf = lut_net.GetInputSignalBuffer();
			for (size_t frame = 0; frame < batch_size; ++frame) {
				for (size_t node = 0; node < x_node_size; ++node) {
					in_sig_buf.Set<float>(frame, node, x_train[train_index + frame][node]);
				}
			}

			// 予測
			lut_net.Forward(true);

			// バイナリ版フィードバック(力技学習)
			while (lut_mux.Feedback(lut_mux.GetOutputOnehotLoss<std::uint8_t, 10>(label_train, train_index)))
				;

			//		while (bin_mux.Feedback(bin_mux.GetOutputOnehotLoss(y_train, train_index)))
			//			;

			//		while (bin_mux.Feedback(bin_mux.CalcLoss(y_train, train_index)))
			//			;

			// 途中評価
			lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
			auto test_accuracy = lut_net.RunCalculation(x_test, y_test, lut_max_batch_size, 0, &lut_acc_func);

			// 進捗表示
			auto progress = train_index + batch_size;
			auto rate = progress * 100 / train_size;
			std::cout << "[" << rate << "% (" << progress << "/" << train_size << ")]";
			std::cout << "  test_accuracy : " << test_accuracy << "                  ";
			std::cout << "\r" << std::flush;
		}

		// 評価
		lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
		auto test_accuracy = lut_net.RunCalculation(x_test, y_test, lut_max_batch_size, 0, &lut_acc_func);
		auto train_accuracy = lut_net.RunCalculation(x_train, y_train, lut_max_batch_size, 0, &lut_acc_func);
		auto now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
		std::cout << now_time << "s " << "epoc[" << epoc << "]"
			<< "  test_accuracy : " << test_accuracy
			<< "  train_accuracy : " << train_accuracy << std::endl;

		// Shuffle
		bb::ShuffleDataSet(mt(), x_train, y_train, label_train);
	}

	{
		// Write RTL
		std::ofstream ofs("lut_net.v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer0_lut, "lutnet_layer0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer1_lut, "lutnet_layer1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer2_lut, "lutnet_layer2");
	}

	std::cout << "end\n" << std::endl;
}





// DenseSimpleConvolution  network
void MnistDenseSimpleConvolution(int epoc_size, size_t max_batch_size)
{
	// run name
	std::string run_name = "MnistDenseSimpleConvolution";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetDenseConvolution<> layer0_conv(1, 28, 28, 32, 3, 3);	// c:1  w:28 h:28  --(filter:3x3)--> c:32 w:26 h:26 
	bb::NeuralNetDenseConvolution<> layer1_conv(32, 26, 26, 32, 3, 3);	// c:32 w:26 h:26  --(filter:3x3)--> c:32 w:24 h:24 
	bb::NeuralNetMaxPooling<>		layer2_maxpol(32, 24, 24, 2, 2);	// c:32 w:24 h:24  --(filter:2x2)--> c:32 w:12 h:12 
	bb::NeuralNetDenseAffine<>		layer3_affine(32 * 12 * 12, 128);
	bb::NeuralNetSigmoid<>			layer4_sigmoid(128);
	bb::NeuralNetDenseAffine<>		layer5_affine(128, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer2_maxpol);
	net.AddLayer(&layer3_affine);
	net.AddLayer(&layer4_sigmoid);
	net.AddLayer(&layer5_affine);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}




void MnistSparseSimpleConvolutionBinary(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistSparseSimpleConvolutionBinary";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();


	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseBinaryAffine<>	sub0_affine0(1 * 3 * 3, 96);
	bb::NeuralNetSparseBinaryAffine<>	sub0_affine1(96, 16);
	bb::NeuralNetGroup<>				sub0_net;
	sub0_net.AddLayer(&sub0_affine0);
	sub0_net.AddLayer(&sub0_affine1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseBinaryAffine<>	sub1_affine0(16 * 3 * 3, 256);
	bb::NeuralNetSparseBinaryAffine<>	sub1_affine1(256, 96);
	bb::NeuralNetSparseBinaryAffine<>	sub1_affine2(96, 16);
	bb::NeuralNetGroup<>				sub1_net;
	sub1_net.AddLayer(&sub1_affine0);
	sub1_net.AddLayer(&sub1_affine1);
	sub1_net.AddLayer(&sub1_affine2);

	bb::NeuralNetLoweringConvolution<>		layer0_conv(&sub0_net, 1, 28, 28, 16, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer1_conv(&sub1_net, 16, 26, 26, 16, 3, 3);
	bb::NeuralNetMaxPooling<>			layer2_maxpol(16, 24, 24, 2, 2);
	bb::NeuralNetSparseBinaryAffine<>	layer3_affine(16 * 12 * 12, 180);
	bb::NeuralNetSparseBinaryAffine<>	layer4_affine(180, 30);

	bb::NeuralNetGroup<>				mux_group;
	mux_group.AddLayer(&layer0_conv);
	mux_group.AddLayer(&layer1_conv);
	mux_group.AddLayer(&layer2_maxpol);
	mux_group.AddLayer(&layer3_affine);
	mux_group.AddLayer(&layer4_affine);
	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&mux_group, 28 * 28, 10, 1, 3);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// set multiplexing size
	bin_mux.SetMuxSize(1);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}



void MnistLutSimpleConvolutionBinary(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistLutSimpleConvolutionBinary";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// 入力のバイナリ化
	for (auto& xx : train_data.x_train) {
		for (auto& x : xx) {
			x = (x >= 0.5) ? 1.0f : 0.0f;
		}
	}
	for (auto& xx : train_data.x_test) {
		for (auto& x : xx) {
			x = (x >= 0.5) ? 1.0f : 0.0f;
		}
	}


	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm0(1 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub0_smm1(192, 32);
	bb::NeuralNetGroup<>				sub0_net;
	sub0_net.AddLayer(&sub0_smm0);
	sub0_net.AddLayer(&sub0_smm1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm0(32 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm1(192, 32);
	bb::NeuralNetGroup<>				sub1_net;
	sub1_net.AddLayer(&sub1_smm0);
	sub1_net.AddLayer(&sub1_smm1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm0(32 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub3_smm1(192, 32);
	bb::NeuralNetGroup<>				sub3_net;
	sub3_net.AddLayer(&sub3_smm0);
	sub3_net.AddLayer(&sub3_smm1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm0(32 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub4_smm1(192, 32);
	bb::NeuralNetGroup<>				sub4_net;
	sub4_net.AddLayer(&sub4_smm0);
	sub4_net.AddLayer(&sub4_smm1);

	bb::NeuralNetLoweringConvolution<>		layer0_conv(&sub0_net, 1, 28, 28, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer1_conv(&sub1_net, 32, 26, 26, 32, 3, 3);
	bb::NeuralNetMaxPooling<>				layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<>		layer3_conv(&sub3_net, 32, 12, 12, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer4_conv(&sub4_net, 32, 10, 10, 32, 3, 3);
	bb::NeuralNetMaxPooling<>				layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer6_smm(32 * 4 * 4, 480);
	bb::NeuralNetSparseMicroMlp<6, 16>		layer7_smm(480, 80);
	bb::NeuralNetBinaryToReal<float>		output_bin2real(80, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer2_maxpol);
	net.AddLayer(&layer3_conv);
	net.AddLayer(&layer4_conv);
	net.AddLayer(&layer5_maxpol);
	net.AddLayer(&layer6_smm);
	net.AddLayer(&layer7_smm);
	net.AddLayer(&output_bin2real);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// set multiplexing size
	//	bin_mux.SetMuxSize(1);


#if 1
	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
#else
	{
		std::string net_file_name = "MnistLutSimpleConvolutionBinary_net.json";
		std::ifstream ifs(net_file_name);
		cereal::JSONInputArchive ar(ifs);
		int epoc;
		ar(cereal::make_nvp("epoc", epoc));
		net.Load(ar);
		std::cout << "[load] " << net_file_name << std::endl;
	}
#endif


	// Binary Network
#if 1

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<>	bin_sub0_lut0(1 * 3 * 3, 192);
	bb::NeuralNetBinaryLut6<>	bin_sub0_lut1(192, 32);
	bb::NeuralNetGroup<>		bin_sub0_net;
	bin_sub0_net.AddLayer(&bin_sub0_lut0);
	bin_sub0_net.AddLayer(&bin_sub0_lut1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<>	bin_sub1_lut0(32 * 3 * 3, 192);
	bb::NeuralNetBinaryLut6<>	bin_sub1_lut1(192, 32);
	bb::NeuralNetGroup<>		bin_sub1_net;
	bin_sub1_net.AddLayer(&bin_sub1_lut0);
	bin_sub1_net.AddLayer(&bin_sub1_lut1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<>	bin_sub3_lut0(32 * 3 * 3, 192);
	bb::NeuralNetBinaryLut6<>	bin_sub3_lut1(192, 32);
	bb::NeuralNetGroup<>		bin_sub3_net;
	bin_sub3_net.AddLayer(&bin_sub3_lut0);
	bin_sub3_net.AddLayer(&bin_sub3_lut1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<>	bin_sub4_lut0(32 * 3 * 3, 192);
	bb::NeuralNetBinaryLut6<>	bin_sub4_lut1(192, 32);
	bb::NeuralNetGroup<>		bin_sub4_net;
	bin_sub4_net.AddLayer(&bin_sub4_lut0);
	bin_sub4_net.AddLayer(&bin_sub4_lut1);

	bb::NeuralNetLoweringConvolution<bool>	bin_layer0_conv(&bin_sub0_net, 1, 28, 28, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<bool>	bin_layer1_conv(&bin_sub1_net, 32, 26, 26, 32, 3, 3);
	bb::NeuralNetMaxPooling<bool>		bin_layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<bool>	bin_layer3_conv(&bin_sub3_net, 32, 12, 12, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<bool>	bin_layer4_conv(&bin_sub4_net, 32, 10, 10, 32, 3, 3);
	bb::NeuralNetMaxPooling<bool>		bin_layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetBinaryLut6<>			bin_layer6_lut(32 * 4 * 4, 480);
	bb::NeuralNetBinaryLut6<>			bin_layer7_lut(480, 80);

	bb::NeuralNetGroup<>			bin_mux_group;
	bin_mux_group.AddLayer(&bin_layer0_conv);
	bin_mux_group.AddLayer(&bin_layer1_conv);
	bin_mux_group.AddLayer(&bin_layer2_maxpol);
	bin_mux_group.AddLayer(&bin_layer3_conv);
	bin_mux_group.AddLayer(&bin_layer4_conv);
	bin_mux_group.AddLayer(&bin_layer5_maxpol);
	bin_mux_group.AddLayer(&bin_layer6_lut);
	bin_mux_group.AddLayer(&bin_layer7_lut);
	bb::NeuralNetBinaryMultiplex<>	bin_bin_mux(&bin_mux_group, 28 * 28, 10, 1, 8);

	// build network
	bb::NeuralNet<> bin_net;
	bin_net.AddLayer(&bin_bin_mux);

	// copy
#if 1
	std::cout << "[parameter copy]" << std::endl;
	bin_sub0_lut0.ImportLayer(sub0_smm0);
	bin_sub0_lut1.ImportLayer(sub0_smm1);
	bin_sub1_lut0.ImportLayer(sub1_smm0);
	bin_sub1_lut1.ImportLayer(sub1_smm1);
	bin_sub3_lut0.ImportLayer(sub3_smm0);
	bin_sub3_lut1.ImportLayer(sub3_smm1);
	bin_sub4_lut0.ImportLayer(sub4_smm0);
	bin_sub4_lut1.ImportLayer(sub4_smm1);
	bin_layer6_lut.ImportLayer(layer6_smm);
	bin_layer7_lut.ImportLayer(layer7_smm);
#endif

	// 評価関数
	bb::NeuralNetAccuracyCategoricalClassification<>	bin_acc_func(num_class);

	// 初期評価
	bin_bin_mux.SetMuxSize(1);	// 評価用の多重化数にスイッチ
	auto test_accuracy = bin_net.RunCalculation(train_data.x_test, train_data.y_test, max_batch_size, 0, &bin_acc_func);
	std::cout << "copy test_accuracy : " << test_accuracy << std::endl;
	auto train_accuracy = bin_net.RunCalculation(train_data.x_train, train_data.y_train, max_batch_size, 0, &bin_acc_func);
	std::cout << "copy train_accuracy : " << train_accuracy << std::endl;

	{
		// Write RTL
		std::ofstream ofs("lut_net_simple_cnv.v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub0_lut0, "lutnet_layer0_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub0_lut1, "lutnet_layer0_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub1_lut0, "lutnet_layer1_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub1_lut1, "lutnet_layer1_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub3_lut0, "lutnet_layer3_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub3_lut1, "lutnet_layer3_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub4_lut0, "lutnet_layer4_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub4_lut1, "lutnet_layer4_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer6_lut, "lutnet_layer6");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer7_lut, "lutnet_layer7");
	}
#endif
}






void MnistLutSimpleConvolutionBinary2(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistLutSimpleConvolutionBinary2";
	int			num_class = 10;

	// load MNIST data
	auto data = bb::LoadMnist<>::Load();

	// 入力のバイナリ化
	for (auto& xx : data.x_train) {
		for (auto& x : xx) {
			x = (x >= 0.5) ? 1.0f : 0.0f;
		}
	}
	for (auto& xx : data.x_test) {
		for (auto& x : xx) {
			x = (x >= 0.5) ? 1.0f : 0.0f;
		}
	}

	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>		sub0_smm0(1 * 3 * 3, 256);
	bb::NeuralNetSparseMicroMlp<6, 16>		sub0_smm1(256, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>		sub0_smm2(192, 32);
	bb::NeuralNetGroup<>		sub0_net;
	sub0_net.AddLayer(&sub0_smm0);
	sub0_net.AddLayer(&sub0_smm1);
	sub0_net.AddLayer(&sub0_smm2);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>		sub1_smm0(32 * 3 * 3, 512);
	bb::NeuralNetSparseMicroMlp<6, 16>		sub1_smm1(512, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>		sub1_smm2(192, 32);
	bb::NeuralNetGroup<>		sub1_net;
	sub1_net.AddLayer(&sub1_smm0);
	sub1_net.AddLayer(&sub1_smm1);
	sub1_net.AddLayer(&sub1_smm2);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>		sub3_smm0(32 * 3 * 3, 512);
	bb::NeuralNetSparseMicroMlp<6, 16>		sub3_smm1(512, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>		sub3_smm2(192, 32);
	bb::NeuralNetGroup<>		sub3_net;
	sub3_net.AddLayer(&sub3_smm0);
	sub3_net.AddLayer(&sub3_smm1);
	sub3_net.AddLayer(&sub3_smm2);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseMicroMlp<6, 16>		sub4_smm0(32 * 3 * 3, 512);
	bb::NeuralNetSparseMicroMlp<6, 16>		sub4_smm1(512, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>		sub4_smm2(192, 32);
	bb::NeuralNetGroup<>		sub4_net;
	sub4_net.AddLayer(&sub4_smm0);
	sub4_net.AddLayer(&sub4_smm1);
	sub4_net.AddLayer(&sub4_smm2);

	bb::NeuralNetLoweringConvolution<>		layer0_conv(&sub0_net, 1, 28, 28, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer1_conv(&sub1_net, 32, 26, 26, 32, 3, 3);
	bb::NeuralNetMaxPooling<>				layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<>		layer3_conv(&sub3_net, 32, 12, 12, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>		layer4_conv(&sub4_net, 32, 10, 10, 32, 3, 3);
	bb::NeuralNetMaxPooling<>				layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetSparseMicroMlp<6, 32>		layer6_lut(32 * 4 * 4, 1024);
	bb::NeuralNetSparseMicroMlp<6, 32>		layer7_lut(1024, 1024);
	bb::NeuralNetSparseMicroMlp<6, 32>		layer8_lut(1024, 512);
	bb::NeuralNetSparseMicroMlp<6, 32>		layer9_lut(512, 100);
	bb::NeuralNetBinaryToReal<float>		output_bin2real(100, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer2_maxpol);
	net.AddLayer(&layer3_conv);
	net.AddLayer(&layer4_conv);
	net.AddLayer(&layer5_maxpol);
	net.AddLayer(&layer6_lut);
	net.AddLayer(&layer7_lut);
	net.AddLayer(&layer8_lut);
	net.AddLayer(&layer9_lut);
	net.AddLayer(&output_bin2real);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	bb::NeuralNetLossCrossEntropyWithSoftmax<>			loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	acc_func(num_class);

#if 1
	// run fitting
//	net.Fitting(run_name + "_real", data, 2, max_batch_size, &acc_func, &loss_func, true, true);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// run fitting
	net.Fitting(run_name, data, epoc_size, max_batch_size, &acc_func, &loss_func, true, true);

#else
	{
		std::string net_file_name = "MnistLutSimpleConvolutionBinary_net.json";
		std::ifstream ifs(net_file_name);
		cereal::JSONInputArchive ar(ifs);
		int epoc;
		ar(cereal::make_nvp("epoc", epoc));
		net.Load(ar);
		std::cout << "[load] " << net_file_name << std::endl;
	}
#endif

	// Binary Network
#if 0

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<>	bin_sub0_lut0(1 * 3 * 3, 192);
	bb::NeuralNetBinaryLut6<>	bin_sub0_lut1(192, 32);
	bb::NeuralNetGroup<>		bin_sub0_net;
	bin_sub0_net.AddLayer(&bin_sub0_lut0);
	bin_sub0_net.AddLayer(&bin_sub0_lut1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<>	bin_sub1_lut0(32 * 3 * 3, 192);
	bb::NeuralNetBinaryLut6<>	bin_sub1_lut1(192, 32);
	bb::NeuralNetGroup<>		bin_sub1_net;
	bin_sub1_net.AddLayer(&bin_sub1_lut0);
	bin_sub1_net.AddLayer(&bin_sub1_lut1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<>	bin_sub3_lut0(32 * 3 * 3, 192);
	bb::NeuralNetBinaryLut6<>	bin_sub3_lut1(192, 32);
	bb::NeuralNetGroup<>		bin_sub3_net;
	bin_sub3_net.AddLayer(&bin_sub3_lut0);
	bin_sub3_net.AddLayer(&bin_sub3_lut1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<>	bin_sub4_lut0(32 * 3 * 3, 192);
	bb::NeuralNetBinaryLut6<>	bin_sub4_lut1(192, 32);
	bb::NeuralNetGroup<>		bin_sub4_net;
	bin_sub4_net.AddLayer(&bin_sub4_lut0);
	bin_sub4_net.AddLayer(&bin_sub4_lut1);

	bb::NeuralNetLoweringConvolution<bool>	bin_layer0_conv(&bin_sub0_net, 1, 28, 28, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<bool>	bin_layer1_conv(&bin_sub1_net, 32, 26, 26, 32, 3, 3);
	bb::NeuralNetMaxPooling<bool>		bin_layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<bool>	bin_layer3_conv(&bin_sub3_net, 32, 12, 12, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<bool>	bin_layer4_conv(&bin_sub4_net, 32, 10, 10, 32, 3, 3);
	bb::NeuralNetMaxPooling<bool>		bin_layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetBinaryLut6<>			bin_layer6_lut(32 * 4 * 4, 480);
	bb::NeuralNetBinaryLut6<>			bin_layer7_lut(480, 80);

	bb::NeuralNetGroup<>			bin_mux_group;
	bin_mux_group.AddLayer(&bin_layer0_conv);
	bin_mux_group.AddLayer(&bin_layer1_conv);
	bin_mux_group.AddLayer(&bin_layer2_maxpol);
	bin_mux_group.AddLayer(&bin_layer3_conv);
	bin_mux_group.AddLayer(&bin_layer4_conv);
	bin_mux_group.AddLayer(&bin_layer5_maxpol);
	bin_mux_group.AddLayer(&bin_layer6_lut);
	bin_mux_group.AddLayer(&bin_layer7_lut);
	bb::NeuralNetBinaryMultiplex<>	bin_bin_mux(&bin_mux_group, 28 * 28, 10, 1, 8);

	// build network
	bb::NeuralNet<> bin_net;
	bin_net.AddLayer(&bin_bin_mux);

	// copy
#if 1
	std::cout << "[parameter copy]" << std::endl;
	bin_sub0_lut0.ImportLayer(sub0_lut0);
	bin_sub0_lut1.ImportLayer(sub0_lut1);
	bin_sub1_lut0.ImportLayer(sub1_lut0);
	bin_sub1_lut1.ImportLayer(sub1_lut1);
	bin_sub3_lut0.ImportLayer(sub3_lut0);
	bin_sub3_lut1.ImportLayer(sub3_lut1);
	bin_sub4_lut0.ImportLayer(sub4_lut0);
	bin_sub4_lut1.ImportLayer(sub4_lut1);
	bin_layer6_lut.ImportLayer(layer6_lut);
	bin_layer7_lut.ImportLayer(layer7_lut);
#endif

	// 評価関数
	bb::NeuralNetAccuracyCategoricalClassification<>	bin_acc_func(num_class);

	// 初期評価
	bin_bin_mux.SetMuxSize(1);	// 評価用の多重化数にスイッチ
	auto test_accuracy = bin_net.RunCalculation(train_data.x_test, train_data.y_test, max_batch_size, 0, &bin_acc_func);
	std::cout << "copy test_accuracy : " << test_accuracy << std::endl;
	auto train_accuracy = bin_net.RunCalculation(train_data.x_train, train_data.y_train, max_batch_size, 0, &bin_acc_func);
	std::cout << "copy train_accuracy : " << train_accuracy << std::endl;

	{
		// Write RTL
		std::ofstream ofs("lut_net_simple_cnv.v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub0_lut0, "lutnet_layer0_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub0_lut1, "lutnet_layer0_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub1_lut0, "lutnet_layer1_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub1_lut1, "lutnet_layer1_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub3_lut0, "lutnet_layer3_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub3_lut1, "lutnet_layer3_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub4_lut0, "lutnet_layer4_sub0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_sub4_lut1, "lutnet_layer4_sub1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer6_lut, "lutnet_layer6");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer7_lut, "lutnet_layer7");
	}
#endif
}



