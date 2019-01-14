#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

#include "gtest/gtest.h"

#include "bb/NeuralNetStackedMicroAffine.h"
#include "cubb/MicroMlp.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}



#define	N					6
#define	M					16

#if 0

#define FRAME_SIZE			8
#define OUTPUT_NODE_SIZE	2
#define INPUT_NODE_SIZE		6

#else

#define FRAME_SIZE			(2*64*28*28)
#define INPUT_NODE_SIZE		(256*3*3)
#define OUTPUT_NODE_SIZE	(256)
#endif


float	in_sig[INPUT_NODE_SIZE*FRAME_SIZE];
float	out_sig[OUTPUT_NODE_SIZE*FRAME_SIZE];
int		input_index[INPUT_NODE_SIZE*N];
float	hidden_W[OUTPUT_NODE_SIZE*M*N];
float	hidden_b[OUTPUT_NODE_SIZE*M];
float	output_W[OUTPUT_NODE_SIZE*M];
float	output_b[OUTPUT_NODE_SIZE];


#if 1

TEST(cudaMicroMlpTest, test_cudaMicroMlp2)
{
	bb::NeuralNetStackedMicroAffine<N, M> umlp_cpu(INPUT_NODE_SIZE, OUTPUT_NODE_SIZE);
	umlp_cpu.SetBatchSize(FRAME_SIZE);
	testSetupLayerBuffer(umlp_cpu);

	std::mt19937_64 mt(1);
	std::uniform_int_distribution<int>	index_rand(0, INPUT_NODE_SIZE-1);
	std::uniform_int_distribution<int>	norm_rand(-10, 10);
//	std::normal_distribution<float>		norm_rand(0, 1);
	
	
	for (int i = 0; i < sizeof(in_sig) / sizeof(float); ++i) { in_sig[i] = norm_rand(mt); }

	for (int i = 0; i < sizeof(input_index) / sizeof(int); ++i) { input_index[i] = index_rand(mt); }
	for (int i = 0; i < sizeof(hidden_W) / sizeof(float); ++i) { hidden_W[i] = norm_rand(mt); }
	for (int i = 0; i < sizeof(hidden_b) / sizeof(float); ++i) { hidden_b[i] = norm_rand(mt); }
	for (int i = 0; i < sizeof(output_W) / sizeof(float); ++i) { output_W[i] = norm_rand(mt); }
	for (int i = 0; i < sizeof(output_b) / sizeof(float); ++i) { output_b[i] = -norm_rand(mt); }
	

	std::cout << "\n\n";
	std::cout << "input size : " << INPUT_NODE_SIZE << "\n";
	std::cout <<  OUTPUT_NODE_SIZE << " node[6x16] x " << FRAME_SIZE << " frames\n\n";

	std::cout << "[GPU GT1030]" << std::endl;
	MicroMlp6x16_Forward
		(
			INPUT_NODE_SIZE,
			OUTPUT_NODE_SIZE,
			FRAME_SIZE,
			in_sig,
			out_sig,
			input_index,
			hidden_W,
			hidden_b,
			output_W,
			output_b
		);
	
	for (int i = 0; i < OUTPUT_NODE_SIZE; i++) {
		for (int j = 0; j < N; j++) {
			umlp_cpu.SetNodeInput(i, j, input_index[i*N+j]);
		}
	}

	for (int i = 0; i < OUTPUT_NODE_SIZE; i++) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k < N; k++) {
				umlp_cpu.W0(i, j, k) = hidden_W[i*(M*N) + j*N + k];
			}
			umlp_cpu.b0(i, j) = hidden_b[i*M + j];
		}

		for (int j = 0; j < M; j++) {
			umlp_cpu.W1(i, j) = output_W[i*M + j];
		}
		umlp_cpu.b1(i) = output_b[i];
	}

	auto in_sig_buf  = umlp_cpu.GetInputSignalBuffer();
	auto out_sig_buf = umlp_cpu.GetOutputSignalBuffer();

	for (int i = 0; i < INPUT_NODE_SIZE; i++) {
		for (int j = 0; j < FRAME_SIZE; j++) {
			in_sig_buf.SetReal(j, i, in_sig[FRAME_SIZE*i + j]);
		}
	}

	auto time0 = std::chrono::system_clock::now();

	umlp_cpu.Forward();

	auto time1 = std::chrono::system_clock::now();

	std::cout << "\n\n[CPU Core i7-4770]" << std::endl;
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count();
	std::cout << "OpenMP + AVX2 : " << elapsed << " [msec]" << std::endl;
	double flops = OUTPUT_NODE_SIZE * FRAME_SIZE * (16+6)*2 / elapsed / 1000000.0;
	std::cout << "      " << flops << " [GFLOPS]  (" << flops / 435.2 * 100.0 << "% [peak 435.2 GFLOPS])" << std::endl;

	std::cout << "\n\n";
	
	for (int i = 0; i < OUTPUT_NODE_SIZE; i++) {
		for (int j = 0; j < FRAME_SIZE; j++) {
			EXPECT_EQ(out_sig_buf.GetReal(j, i), out_sig[FRAME_SIZE*i + j]);
//			std::cout << out_sig_buf.GetReal(j, i) << " " << out_sig[FRAME_SIZE*i + j] << std::endl;
		}
	}
}


#else

TEST(cudaMicroMlpTest, test_cudaMicroMlp1)
{
	const int frame_size = FRAME_SIZE;
	const int input_node_size = INPUT_NODE_SIZE;
	const int output_node_size = OUTPUT_NODE_SIZE;

	memset(in_sig, 0, sizeof(in_sig));
	memset(out_sig, 0, sizeof(out_sig));
	memset(input_index, 0, sizeof(input_index));
	memset(hidden_W, 0, sizeof(hidden_W));
	memset(hidden_b, 0, sizeof(hidden_b));
	memset(output_W, 0, sizeof(output_W));
	memset(output_b, 0, sizeof(output_b));

	for ( int i = 0; i < frame_size; ++i ) {
		for ( int j = 0; j < input_node_size; ++j ) {
			in_sig[frame_size*j + i] = i * 100 + j + 10000;
		}
	}

//	in_sig[frame_size*0 + 0] = 2;
//	in_sig[frame_size*1 + 0] = 2;
//	in_sig[frame_size*2 + 0] = 3;
//	in_sig[frame_size*3 + 0] = 4;
//	in_sig[frame_size*4 + 0] = 5;
//	in_sig[frame_size*5 + 0] = 6;
//	in_sig[frame_size*0 + 1] = 11;
//	in_sig[frame_size*1 + 1] = 12;
//	in_sig[frame_size*2 + 1] = 13;
//	in_sig[frame_size*3 + 1] = 14;
//	in_sig[frame_size*4 + 1] = 15;
//	in_sig[frame_size*5 + 1] = 16;

	input_index[0] = 5;
	input_index[1] = 4;
	input_index[2] = 3;
	input_index[3] = 2;
	input_index[4] = 1;
	input_index[5] = 0;

	input_index[6+0] = 0;
	input_index[6+1] = 1;
	input_index[6+2] = 2;
	input_index[6+3] = 3;
	input_index[6+4] = 4;
	input_index[6+5] = 5;

	hidden_W[6*0+0] = -1;
	hidden_W[6*0+1] = 3;
	hidden_W[6*0+2] = 4;
	hidden_W[6*0+3] = 5;
	hidden_W[6*0+4] = 0;
	hidden_W[6*0+5] = 0;

	hidden_W[6*1+0] = 3;
	hidden_W[6*1+1] = 0;
	hidden_W[6*1+2] = 0;
	hidden_W[6*1+3] = 0;
	hidden_W[6*1+4] = 0;
	hidden_W[6*1+5] = 0;

	hidden_W[6*2+0] = 0;
	hidden_W[6*3+0] = 0;
	hidden_W[6*4+0] = 0;
	hidden_W[6*5+0] = 0;

	hidden_b[2] = 0;

	output_W[0] = 1;
	output_W[1] = 1;
	output_W[2] = 1;
	output_W[3] = 0;
	output_W[4] = 0;
	output_W[5] = 0;
	output_W[6] = 0;
	output_W[7] = 0;
	output_W[8] = 0;
	output_W[9] = 0;
	output_W[10] = 0;
	output_W[11] = 0;
	output_W[12] = 0;
	output_W[13] = 0;
	output_W[14] = 0;
	output_W[15] = 0;

	output_b[0] = 0;

	MicroMlp6x16_Forward
		(
			input_node_size,
			output_node_size,
			frame_size,
			in_sig,
			out_sig,
			input_index,
			hidden_W,
			hidden_b,
			output_W,
			output_b
		);
	
	for ( int i = 0; i < frame_size; ++ i ) {
		std::cout << "out[" << i << "] : " << out_sig[i] << std::endl;
		std::cout << "out[" << i << "] : " << out_sig[frame_size+i] << std::endl;
	}
}


#endif

