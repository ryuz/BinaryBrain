#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/NeuralNetGroup.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryFilter.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputValueBuffer (net.CreateInputValueBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputValueBuffer(net.CreateOutputValueBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetBinaryFilterTest, testNeuralNetBinaryFilter)
{
	size_t batch_size = 3;
	size_t input_c_size = 3;
	size_t input_h_size = 16;
	size_t input_w_size = 24;
	size_t output_c_size = 3;
	size_t filter_h_size = 7;
	size_t filter_w_size = 5;
	size_t y_step = 2;
	size_t x_step = 3;
	size_t mux_size = 7;

	// ÉtÉBÉãÉ^ëwê∂ê¨
	size_t filter_input_node_size  = input_c_size * input_h_size * input_w_size;
	size_t filter_layer0_node_size = output_c_size * 6;
	size_t filter_output_node_size = output_c_size;
	bb::NeuralNetGroup<> filter_net;
	bb::NeuralNetBinaryLut6<> filter_lut0(filter_input_node_size, filter_layer0_node_size, mux_size);
	bb::NeuralNetBinaryLut6<> filter_lut1(filter_layer0_node_size, filter_output_node_size, mux_size);
	filter_net.AddLayer(&filter_lut0);
	filter_net.AddLayer(&filter_lut1);
	
	// èÙÇ›çûÇ›ëwç\ê¨
	bb::NeuralNetBinaryFilter<> net(&filter_net, input_c_size, input_h_size, input_w_size, output_c_size, filter_h_size, filter_w_size, y_step, x_step, mux_size);
	testSetupLayerBuffer(net);

	net.SetBatchSize(batch_size);
	net.Forward();

	EXPECT_EQ(true, true);
}


