
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <random>

#include <stdio.h>
#include <stdlib.h>

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



int bbcu_eva_MicroMlp6x16_Forward
		(
			const float*	in_sig_buf,
			float*			out_sig_buf,
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			const int*		input_index,
			const float*	hidden_W,
			const float*	hidden_b,
			const float*	output_W,
			const float*	output_b
		)
{
    int const N = 6;
    int const M = 16;

    cudaDeviceSynchronize();
	auto time0 = std::chrono::system_clock::now();

	float* dev_in_sig;
	float* dev_out_sig;
	int*   dev_input_index;
	float* dev_hidden_W;
	float* dev_hidden_b;
	float* dev_output_W;
	float* dev_output_b;

	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_in_sig,   input_node_size * frame_size * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_out_sig,  output_node_size * frame_size * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_input_index, output_node_size * N * sizeof(int)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_hidden_W, output_node_size * M * N * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_hidden_b, output_node_size * M * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_output_W, output_node_size * M * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_output_b, output_node_size * sizeof(float)));
	
	cudaDeviceSynchronize();
	auto time1 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_input_index, input_index, output_node_size * N * sizeof(int), cudaMemcpyHostToDevice));
	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_hidden_W, hidden_W, output_node_size * M * N * sizeof(float), cudaMemcpyHostToDevice));
	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_hidden_b, hidden_b, output_node_size * M * sizeof(float), cudaMemcpyHostToDevice));
	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_output_W, output_W, output_node_size * M * sizeof(float), cudaMemcpyHostToDevice));
	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_output_b, output_b, output_node_size * sizeof(float), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
	auto time2 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_in_sig, in_sig_buf, input_node_size * frame_size * sizeof(float), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
	auto time3 = std::chrono::system_clock::now();
	
	bbcu_MicroMlp6x16_Forward(
			dev_in_sig,
			dev_out_sig,
			input_node_size,
			output_node_size,
			frame_size,
			dev_input_index,
			dev_hidden_W,
			dev_hidden_b,
			dev_output_W,
			dev_output_b
		);
	cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(1);
    }


	cudaDeviceSynchronize();
	auto time4 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaMemcpy(out_sig_buf, dev_out_sig, output_node_size * frame_size * sizeof(float), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
	auto time5 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaFree(dev_in_sig));
	BB_CUDA_SAFE_CALL(cudaFree(dev_out_sig));
	BB_CUDA_SAFE_CALL(cudaFree(dev_hidden_W));
	BB_CUDA_SAFE_CALL(cudaFree(dev_hidden_b));
	BB_CUDA_SAFE_CALL(cudaFree(dev_output_W));
	BB_CUDA_SAFE_CALL(cudaFree(dev_output_b));

	cudaDeviceSynchronize();
	auto time6 = std::chrono::system_clock::now();

	double elapsed_malloc       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count();
	double elapsed_cpu_to_gpu_p = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
	double elapsed_cpu_to_gpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
	double elapsed_kernel       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
	double elapsed_gpu_to_cpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
	double elapsed_free         = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time6-time5).count();

	double kernel_flops = (double)output_node_size *(double) frame_size * (M*N+M+M)*2.0 / elapsed_kernel / 1000000.0;

	std::cout << "malloc               : " << elapsed_malloc       << " [msec]" << std::endl;
	std::cout << "param copy(cpu->gpu) : " << elapsed_cpu_to_gpu_p << " [msec]" << std::endl;
	std::cout << "data copy(cpu->gpu)  : " << elapsed_cpu_to_gpu   << " [msec]" << std::endl;
	std::cout << "kernel               : " << elapsed_kernel       << " [msec]  " << kernel_flops << " [GFLOPS]" << std::endl;
	std::cout << "data copy(gpu->cpu)  : " << elapsed_gpu_to_cpu   << " [msec]" << std::endl;
	std::cout << "free                 : " << elapsed_free         << " [msec]" << std::endl;
	
	return 0;
}


int bbcu_eva_MicroMlp6x16_Backward
		(
			const float*	in_sig_buf,
			float*			in_err_buf,
			float*			out_err_buf,
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			const int*		input_index,
			const float*	hidden_W,
			const float*	hidden_b,
			float*			hidden_dW,
			float*			hidden_db,
			const float*	output_W,
			const float*	output_b,
			float*			output_dW,
			float*			output_db
		)
{
    int const N = 6;
    int const M = 16;

    cudaDeviceProp dev;
	BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));

	cudaError_t cudaStatus0 = cudaGetLastError();
    if (cudaStatus0 != cudaSuccess) {
        fprintf(stderr, "start failed: %s\n", cudaGetErrorString(cudaStatus0));
		exit(1);
    }

	cudaDeviceSynchronize();
	auto time0 = std::chrono::system_clock::now();

	float* dev_in_sig_buf;
	float* dev_in_err_buf;
	float* dev_in_err_tmp;
	float* dev_out_err_buf;

	int*   dev_input_index;
	float* dev_hidden_W;
	float* dev_hidden_b;
	float* dev_output_W;
	float* dev_output_b;
	float* dev_hidden_dW;
	float* dev_hidden_db;
	float* dev_output_dW;
	float* dev_output_db;

	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_in_sig_buf,  input_node_size * frame_size * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_in_err_buf,  input_node_size * frame_size * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_in_err_tmp,  output_node_size * N * frame_size * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_out_err_buf, output_node_size * frame_size * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_input_index, output_node_size * N * sizeof(int)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_hidden_W, output_node_size * M * N * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_hidden_b, output_node_size * M * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_output_W, output_node_size * M * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_output_b, output_node_size * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_hidden_dW, output_node_size * M * N * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_hidden_db, output_node_size * M * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_output_dW, output_node_size * M * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_output_db, output_node_size * sizeof(float)));
	
	cudaDeviceSynchronize();
	auto time1 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_input_index, input_index, output_node_size * N * sizeof(int), cudaMemcpyHostToDevice));
	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_hidden_W, hidden_W, output_node_size * M * N * sizeof(float), cudaMemcpyHostToDevice));
	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_hidden_b, hidden_b, output_node_size * M * sizeof(float), cudaMemcpyHostToDevice));
	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_output_W, output_W, output_node_size * M * sizeof(float), cudaMemcpyHostToDevice));
	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_output_b, output_b, output_node_size * sizeof(float), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
	auto time2 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_in_sig_buf,  in_sig_buf,  input_node_size * frame_size * sizeof(float), cudaMemcpyHostToDevice));
	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_out_err_buf, out_err_buf, output_node_size * frame_size * sizeof(float), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
	auto time3 = std::chrono::system_clock::now();

	bbcu_MicroMlp6x16_Backward(
			dev_in_sig_buf,
			dev_in_err_buf,
			dev_in_err_tmp,
			dev_out_err_buf,
			input_node_size,
			output_node_size,
			frame_size,
			dev_input_index,
			dev_hidden_W,
			dev_hidden_b,
			dev_hidden_dW,
			dev_hidden_db,
			dev_output_W,
			dev_output_b,
			dev_output_dW,
			dev_output_db
		);


	cudaDeviceSynchronize();
	auto time4 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaMemcpy(in_err_buf, dev_in_err_buf, input_node_size * frame_size * sizeof(float), cudaMemcpyDeviceToHost));

	BB_CUDA_SAFE_CALL(cudaMemcpy(hidden_dW, dev_hidden_dW, output_node_size * M * N * sizeof(float), cudaMemcpyDeviceToHost));
	BB_CUDA_SAFE_CALL(cudaMemcpy(hidden_db, dev_hidden_db, output_node_size * M * sizeof(float), cudaMemcpyDeviceToHost));
	BB_CUDA_SAFE_CALL(cudaMemcpy(output_dW, dev_output_dW, output_node_size * M * sizeof(float), cudaMemcpyDeviceToHost));
	BB_CUDA_SAFE_CALL(cudaMemcpy(output_db, dev_output_db, output_node_size * sizeof(float), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
	auto time5 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaFree(dev_in_sig_buf));
	BB_CUDA_SAFE_CALL(cudaFree(dev_in_err_buf));
	BB_CUDA_SAFE_CALL(cudaFree(dev_in_err_tmp));
	BB_CUDA_SAFE_CALL(cudaFree(dev_out_err_buf));
	BB_CUDA_SAFE_CALL(cudaFree(dev_input_index));
	BB_CUDA_SAFE_CALL(cudaFree(dev_hidden_W));
	BB_CUDA_SAFE_CALL(cudaFree(dev_hidden_b));
	BB_CUDA_SAFE_CALL(cudaFree(dev_output_W));
	BB_CUDA_SAFE_CALL(cudaFree(dev_output_b));
	BB_CUDA_SAFE_CALL(cudaFree(dev_hidden_dW));
	BB_CUDA_SAFE_CALL(cudaFree(dev_hidden_db));
	BB_CUDA_SAFE_CALL(cudaFree(dev_output_dW));
	BB_CUDA_SAFE_CALL(cudaFree(dev_output_db));

	cudaDeviceSynchronize();
	auto time6 = std::chrono::system_clock::now();

	double elapsed_malloc       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count();
	double elapsed_cpu_to_gpu_p = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
	double elapsed_cpu_to_gpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
	double elapsed_kernel       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
	double elapsed_gpu_to_cpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
	double elapsed_free         = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time6-time5).count();
//	double kernel_flops = (double)output_node_size *(double) frame_size * (16.0*6.0+16.0+16.0)*2.0 / elapsed_kernel / 1000000.0;
	std::cout << "malloc               : " << elapsed_malloc       << " [msec]" << std::endl;
	std::cout << "param copy(cpu->gpu) : " << elapsed_cpu_to_gpu_p << " [msec]" << std::endl;
	std::cout << "data copy(cpu->gpu)  : " << elapsed_cpu_to_gpu   << " [msec]" << std::endl;
	std::cout << "kernel               : " << elapsed_kernel       << " [msec]" << std::endl;
//	 << kernel_flops << " [GFLOPS])" << std::endl;
	std::cout << "data copy(gpu->cpu)  : " << elapsed_gpu_to_cpu   << " [msec]" << std::endl;
	std::cout << "free                 : " << elapsed_free         << " [msec]" << std::endl;
	
	return 0;
}






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

	bbcu_eva_MicroMlp6x16_Forward
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

	bbcu_eva_MicroMlp6x16_Backward
		(   in_sig,
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


