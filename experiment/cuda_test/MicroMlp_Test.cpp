
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <random>

#include <stdio.h>
#include <stdlib.h>

#include "MicroMlp.h"


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

//#define FRAME_SIZE			(128*28*28)
//#define OUTPUT_NODE_SIZE	(128)
//#define INPUT_NODE_SIZE		(128*3*3)

#define FRAME_SIZE			(64*28*28)
#define OUTPUT_NODE_SIZE	(256)
#define INPUT_NODE_SIZE		(256*9)


float	in_sig[INPUT_NODE_SIZE*FRAME_SIZE];
float	out_sig[OUTPUT_NODE_SIZE*FRAME_SIZE];
int		input_index[INPUT_NODE_SIZE*N];
float	hidden_W[OUTPUT_NODE_SIZE*M*N];
float	hidden_b[OUTPUT_NODE_SIZE*M];
float	output_W[OUTPUT_NODE_SIZE*M];
float	output_b[OUTPUT_NODE_SIZE];


int MicroMlp_Test(void)
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
	
	end = std::chrono::system_clock::now();  // 計測終了時間
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
	std::cout << "totel : " << elapsed << " [msec]" << std::endl;

	return 0;
}


