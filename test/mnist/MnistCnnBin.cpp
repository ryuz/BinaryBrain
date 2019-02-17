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
#include "bb/NeuralNetOutputVerilog.h"

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
#include "bb/NeuralNetDenseToSparseAffine.h"

#include "bb/ShuffleSet.h"



// MNIST CNN with LUT networks
void MnistCnnBin(int epoch_size, size_t mini_batch_size, bool binary_mode)
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
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm0(32 * 3 * 3, 192);
	bb::NeuralNetSparseMicroMlp<6, 16>	sub1_smm1(192, 32);
	bb::NeuralNetGroup<>		sub1_net;
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

	bb::NeuralNetRealToBinary<float>	input_real2bin(28 * 28, 28 * 28);
	bb::NeuralNetLoweringConvolution<>	layer0_conv(&sub0_net, 1, 28, 28, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>	layer1_conv(&sub1_net, 32, 26, 26, 32, 3, 3);
	bb::NeuralNetMaxPooling<>			layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<>	layer3_conv(&sub3_net, 32, 12, 12, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<>	layer4_conv(&sub4_net, 32, 10, 10, 32, 3, 3);
	bb::NeuralNetMaxPooling<>			layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer6_smm(32 * 4 * 4, 480);
	bb::NeuralNetSparseMicroMlp<6, 16>	layer7_smm(480, 80);
	bb::NeuralNetBinaryToReal<float>	output_bin2real(80, 10);

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
	net.Fitting(run_name, td, epoch_size, mini_batch_size, &acc_func, &loss_func, true, true, false, false);



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

	bb::NeuralNetBinaryLut6<>	lut_sub6_lut0(32 * 4 * 4, 480);
	bb::NeuralNetBinaryLut6<>	lut_sub6_lut1(480, 80);
	bb::NeuralNetGroup<>		lut_sub6_net;
	lut_sub6_net.AddLayer(&lut_sub6_lut0);
	lut_sub6_net.AddLayer(&lut_sub6_lut1);

	bb::NeuralNetLoweringConvolution<bool>	lut_layer0_conv(&lut_sub0_net, 1, 28, 28, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<bool>	lut_layer1_conv(&lut_sub1_net, 32, 26, 26, 32, 3, 3);
	bb::NeuralNetMaxPooling<bool>			lut_layer2_maxpol(32, 24, 24, 2, 2);
	bb::NeuralNetLoweringConvolution<bool>	lut_layer3_conv(&lut_sub3_net, 32, 12, 12, 32, 3, 3);
	bb::NeuralNetLoweringConvolution<bool>	lut_layer4_conv(&lut_sub4_net, 32, 10, 10, 32, 3, 3);
	bb::NeuralNetMaxPooling<bool>			lut_layer5_maxpol(32, 8, 8, 2, 2);
	bb::NeuralNetLoweringConvolution<bool>	lut_layer6_conv(&lut_sub6_net, 32, 4, 4, 80, 4, 4);

//	bb::NeuralNetBinaryLut6<>				lut_layer6_lut(32 * 4 * 4, 480);
//	bb::NeuralNetBinaryLut6<>				lut_layer7_lut(480, 80);

	bb::NeuralNetGroup<>			lut_mux_group;
	lut_mux_group.AddLayer(&lut_layer0_conv);
	lut_mux_group.AddLayer(&lut_layer1_conv);
	lut_mux_group.AddLayer(&lut_layer2_maxpol);
	lut_mux_group.AddLayer(&lut_layer3_conv);
	lut_mux_group.AddLayer(&lut_layer4_conv);
	lut_mux_group.AddLayer(&lut_layer5_maxpol);
	lut_mux_group.AddLayer(&lut_layer6_conv);
//	lut_mux_group.AddLayer(&lut_layer6_lut);
//	lut_mux_group.AddLayer(&lut_layer7_lut);
	bb::NeuralNetBinaryMultiplex<>	lut_bin_mux(&lut_mux_group, 28 * 28, 10, 1, 8);

	// build network
	bb::NeuralNet<> lut_net;
	lut_net.AddLayer(&lut_bin_mux);

	// copy
	std::cout << "[copy] BSMM-Network -> LUT-Network" << std::endl;
	lut_sub0_lut0.ImportLayer(sub0_smm0);
	lut_sub0_lut1.ImportLayer(sub0_smm1);
	lut_sub1_lut0.ImportLayer(sub1_smm0);
	lut_sub1_lut1.ImportLayer(sub1_smm1);
	lut_sub3_lut0.ImportLayer(sub3_smm0);
	lut_sub3_lut1.ImportLayer(sub3_smm1);
	lut_sub4_lut0.ImportLayer(sub4_smm0);
	lut_sub4_lut1.ImportLayer(sub4_smm1);
	lut_sub6_lut0.ImportLayer(layer6_smm);
	lut_sub6_lut1.ImportLayer(layer7_smm);

//	lut_layer6_lut.ImportLayer(layer6_smm);
//	lut_layer7_lut.ImportLayer(layer7_smm);

	// Accuracy Function
	bb::NeuralNetAccuracyCategoricalClassification<>	lut_acc_func(num_class);

	// evaluation
	lut_bin_mux.SetMuxSize(1);
//	auto test_accuracy = lut_net.RunCalculation(td.x_test, td.y_test, mini_batch_size, 0, &lut_acc_func);
//	std::cout << "[copied LUT-Network] test_accuracy : " << test_accuracy << std::endl;
//	auto train_accuracy = lut_net.RunCalculation(td.x_train, td.y_train, mini_batch_size, 0, &lut_acc_func);
//	std::cout << "[copied LUT-Network] train_accuracy : " << train_accuracy << std::endl;

	// Write RTL
	std::string rtl_fname = run_name + ".v";
	std::ofstream ofs(rtl_fname);

	ofs << "`timescale 1ns / 1ps"  << "\n";
	ofs << "`default_nettype none" << "\n\n\n";

//	bb::OutputVerilogLutGroup<float>(ofs, "lut_net_cnn_mnist", lut_sub0_net);
//	bb::OutputVerilogLoweringConvolution<>(ofs, "lut_net_cnn_mnist", lut_layer0_conv);

	std::vector< bb::NeuralNetFilter2d<>* > layers0;
	layers0.push_back(&lut_layer0_conv);
	layers0.push_back(&lut_layer1_conv);
	layers0.push_back(&lut_layer2_maxpol);
	bb::OutputVerilogCnnAxi4s<>(ofs, "lut_net_cnn_mnist_l0", layers0);

	std::vector< bb::NeuralNetFilter2d<>* > layers1;
	layers1.push_back(&lut_layer3_conv);
	layers1.push_back(&lut_layer4_conv);
	layers1.push_back(&lut_layer5_maxpol);
	bb::OutputVerilogCnnAxi4s<>(ofs, "lut_net_cnn_mnist_l1", layers1);

	std::vector< bb::NeuralNetFilter2d<>* > layers2;
	layers2.push_back(&lut_layer6_conv);
	bb::OutputVerilogCnnAxi4s<>(ofs, "lut_net_cnn_mnist_l2", layers2);


	/*
	bb::NeuralNetBinaryLutVerilog(ofs, lut_sub0_lut0, "lutnet_layer0_sub0");
	bb::NeuralNetBinaryLutVerilog(ofs, lut_sub0_lut1, "lutnet_layer0_sub1");
	bb::NeuralNetBinaryLutVerilog(ofs, lut_sub1_lut0, "lutnet_layer1_sub0");
	bb::NeuralNetBinaryLutVerilog(ofs, lut_sub1_lut1, "lutnet_layer1_sub1");
	bb::NeuralNetBinaryLutVerilog(ofs, lut_sub3_lut0, "lutnet_layer3_sub0");
	bb::NeuralNetBinaryLutVerilog(ofs, lut_sub3_lut1, "lutnet_layer3_sub1");
	bb::NeuralNetBinaryLutVerilog(ofs, lut_sub4_lut0, "lutnet_layer4_sub0");
	bb::NeuralNetBinaryLutVerilog(ofs, lut_sub4_lut1, "lutnet_layer4_sub1");
	bb::NeuralNetBinaryLutVerilog(ofs, lut_layer6_lut, "lutnet_layer6");
	bb::NeuralNetBinaryLutVerilog(ofs, lut_layer7_lut, "lutnet_layer7");
	*/
	ofs << "\n\n";
	ofs << "`default_nettype wire" << "\n";
	std::cout << "write : " << rtl_fname << std::endl;

	std::cout << "end\n" << std::endl;
}


