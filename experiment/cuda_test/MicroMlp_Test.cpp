
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <random>

#include <stdio.h>
#include <stdlib.h>

#include "bbcu/bbcu.h"



#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(1); \
     } \
} while(0)



#define	N					6
#define	M					16

#if _DEBUG
#define FRAME_SIZE			(16*28*28)
#define OUTPUT_NODE_SIZE	(8)
#define INPUT_NODE_SIZE		(8*3*3)
#else
#define FRAME_SIZE			(64*28*28)
#define OUTPUT_NODE_SIZE	(256)
#define INPUT_NODE_SIZE		(256*9)
#endif


float	in_sig[INPUT_NODE_SIZE*FRAME_SIZE];
float	out_sig[OUTPUT_NODE_SIZE*FRAME_SIZE];
float	in_err[INPUT_NODE_SIZE*FRAME_SIZE];
float	out_err[OUTPUT_NODE_SIZE*FRAME_SIZE];
int		input_index[INPUT_NODE_SIZE*N];
float	hidden_W[OUTPUT_NODE_SIZE*M*N];
float	hidden_b[OUTPUT_NODE_SIZE*M];
float	output_W[OUTPUT_NODE_SIZE*M];
float	output_b[OUTPUT_NODE_SIZE];
float	hidden_dW[OUTPUT_NODE_SIZE*M*N];
float	hidden_db[OUTPUT_NODE_SIZE*M];
float	output_dW[OUTPUT_NODE_SIZE*M];
float	output_db[OUTPUT_NODE_SIZE];


int Test_MicroMlp_Forward(void)
{
	cudaDeviceProp dev;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));
//	printf(" shared memory / block : %d (KB)\n", dev.sharedMemPerBlock/1024);

	std::mt19937_64 mt(1);
	std::uniform_int_distribution<int>	index_rand(0, INPUT_NODE_SIZE-1);
	std::normal_distribution<float>		norm_rand(0, 1);

	for (int i = 0; i < sizeof(in_sig) / sizeof(float); ++i) { in_sig[i] = norm_rand(mt); }

	for (int i = 0; i < sizeof(input_index) / sizeof(int); ++i) { input_index[i] = index_rand(mt); }
	for (int i = 0; i < sizeof(hidden_W) / sizeof(float); ++i) { hidden_W[i] = norm_rand(mt); }
	for (int i = 0; i < sizeof(hidden_b) / sizeof(float); ++i) { hidden_b[i] = norm_rand(mt); }
	for (int i = 0; i < sizeof(output_W) / sizeof(float); ++i) { output_W[i] = norm_rand(mt); }
	for (int i = 0; i < sizeof(output_b) / sizeof(float); ++i) { output_b[i] = norm_rand(mt); }

//	std::cout << "start" << std::endl;

	std::chrono::system_clock::time_point  start, end; // 型は auto で可
	start = std::chrono::system_clock::now(); // 計測開始時間

	MicroMlp6x16_Forward
		(
			in_sig,
			out_sig,
			INPUT_NODE_SIZE,
			OUTPUT_NODE_SIZE,
			FRAME_SIZE,
			input_index,
			hidden_W,
			hidden_b,
			output_W,
			output_b
		);
	
	end = std::chrono::system_clock::now();  // 計測終了時間
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
	std::cout << "totel : " << elapsed << " [msec]" << std::endl;

	return 0;
}





int Test_MicroMlp_Backward(void)
{
	cudaDeviceProp dev;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));
//	printf(" shared memory / block : %d (KB)\n", dev.sharedMemPerBlock/1024);

	std::mt19937_64 mt(1);
	std::uniform_int_distribution<int>	index_rand(0, INPUT_NODE_SIZE-1);
	std::normal_distribution<float>		norm_rand(0, 1);

	for (int i = 0; i < sizeof(in_sig) / sizeof(float); ++i)    { in_sig[i]  = norm_rand(mt); }
	for (int i = 0; i < sizeof(out_err) / sizeof(float); ++i)   { out_err[i] = norm_rand(mt); }

	for (int i = 0; i < sizeof(input_index) / sizeof(int); ++i) { input_index[i] = index_rand(mt); }
	for (int i = 0; i < sizeof(hidden_W) / sizeof(float); ++i)  { hidden_W[i] = norm_rand(mt); }
	for (int i = 0; i < sizeof(hidden_b) / sizeof(float); ++i)  { hidden_b[i] = norm_rand(mt); }
	for (int i = 0; i < sizeof(output_W) / sizeof(float); ++i)  { output_W[i] = norm_rand(mt); }
	for (int i = 0; i < sizeof(output_b) / sizeof(float); ++i)  { output_b[i] = norm_rand(mt); }
//	for (int i = 0; i < sizeof(hidden_dW) / sizeof(float); ++i) { hidden_W[i] = norm_rand(mt); }
//	for (int i = 0; i < sizeof(hidden_db) / sizeof(float); ++i) { hidden_b[i] = norm_rand(mt); }
//	for (int i = 0; i < sizeof(output_dW) / sizeof(float); ++i) { output_W[i] = norm_rand(mt); }
//	for (int i = 0; i < sizeof(output_db) / sizeof(float); ++i) { output_b[i] = norm_rand(mt); }

//	std::cout << "start" << std::endl;

	for (int i = 0; i < sizeof(in_sig) / sizeof(float); ++i)    { in_sig[i]  = 0; }
	for (int i = 0; i < sizeof(out_err) / sizeof(float); ++i)   { out_err[i] = 0; }

	for (int i = 0; i < sizeof(input_index) / sizeof(int); ++i) { input_index[i] = i % N; }
	for (int i = 0; i < sizeof(hidden_W) / sizeof(float); ++i)  { hidden_W[i] = 0; }
	for (int i = 0; i < sizeof(hidden_b) / sizeof(float); ++i)  { hidden_b[i] = 0; }
	for (int i = 0; i < sizeof(output_W) / sizeof(float); ++i)  { output_W[i] = 0; }
	for (int i = 0; i < sizeof(output_b) / sizeof(float); ++i)  { output_b[i] = 0; }

	in_sig[0*N + 0] = 1;
	in_sig[0*N + 1] = 2;
	in_sig[0*N + 2] = 3;
	in_sig[0*N + 3] = 4;
	in_sig[0*N + 4] = 5;
	in_sig[0*N + 5] = 6;

	hidden_W[0*N + 0] = 1;
	hidden_W[0*N + 1] = 2;
	hidden_W[0*N + 2] = 3;
	hidden_W[0*N + 3] = 4;
	hidden_W[0*N + 4] = 5;
	hidden_W[0*N + 5] = 6;

	output_W[0*N + 0] = 1;

	out_err[0] = -1;

	std::chrono::system_clock::time_point  start, end; // 型は auto で可
	start = std::chrono::system_clock::now(); // 計測開始時間

	MicroMlp6x16_Backward
		(
			in_sig,
			in_err,
			out_err,
			INPUT_NODE_SIZE,
			OUTPUT_NODE_SIZE,
			FRAME_SIZE,
			input_index,
			hidden_W,
			hidden_b,
			hidden_dW,
			hidden_db,
			output_W,
			output_b,
			output_dW,
			output_db
		);

	std::cout << "output_dW[0] : " << output_dW[0] << std::endl;
	std::cout << "output_dW[1] : " << output_dW[1] << std::endl;
	std::cout << "output_dW[2] : " << output_dW[2] << std::endl;
	std::cout << "output_db[0] : " << output_db[0] << std::endl;
	std::cout << "hidden_dW[0] : " << hidden_dW[0] << std::endl;
	std::cout << "hidden_dW[1] : " << hidden_dW[1] << std::endl;
	std::cout << "hidden_db[2] : " << hidden_dW[2] << std::endl;
	std::cout << "hidden_db[0] : " << hidden_db[0] << std::endl;

	std::cout << "in_err[0] : " << in_err[0] << std::endl;
	std::cout << "in_err[1] : " << in_err[1] << std::endl;
	std::cout << "in_err[2] : " << in_err[2] << std::endl;
	

	end = std::chrono::system_clock::now();  // 計測終了時間
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
	std::cout << "totel : " << elapsed << " [msec]" << std::endl;

	return 0;
}


