
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <random>

#include <stdio.h>
#include <stdlib.h>

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


int bbcu_eva_fp32_StochasticLut6_Forward
        (
            const float     *x_buf,
            float           *y_buf,
            int   const     *input_index,
            float const     *W,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             binary_mode
        )
{
    int const N = 6;
    int const M = (1 << N);


    cudaDeviceSynchronize();
    auto time0 = std::chrono::system_clock::now();

    float* dev_x_buf;
    float* dev_y_buf;
    int*   dev_input_index;
    float* dev_W;

    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_x_buf,       input_node_size * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_y_buf,       output_node_size * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_input_index, output_node_size * N * sizeof(int)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_W,           output_node_size * M * sizeof(float)));
    
    cudaDeviceSynchronize();
    auto time1 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_input_index, input_index, output_node_size * N * sizeof(int), cudaMemcpyHostToDevice));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_W, W, output_node_size * M * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time2 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_x_buf, x_buf, input_node_size * frame_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time3 = std::chrono::system_clock::now();
    
    bbcu_fp32_StochasticLut6_Forward
        (
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_W,
            output_node_size,
            frame_size,
            frame_stride,
            binary_mode
        );

    cudaDeviceSynchronize();
    auto time4 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(y_buf, dev_y_buf, output_node_size * frame_size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    auto time5 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaFree(dev_x_buf));
    BB_CUDA_SAFE_CALL(cudaFree(dev_y_buf));
    BB_CUDA_SAFE_CALL(cudaFree(dev_input_index));
    BB_CUDA_SAFE_CALL(cudaFree(dev_W));

    cudaDeviceSynchronize();
    auto time6 = std::chrono::system_clock::now();

    double elapsed_malloc       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count();
    double elapsed_cpu_to_gpu_p = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
    double elapsed_cpu_to_gpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
    double elapsed_kernel       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
    double elapsed_gpu_to_cpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
    double elapsed_free         = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time6-time5).count();

    std::cout << "malloc               : " << elapsed_malloc       << " [msec]" << std::endl;
    std::cout << "param copy(cpu->gpu) : " << elapsed_cpu_to_gpu_p << " [msec]" << std::endl;
    std::cout << "data copy(cpu->gpu)  : " << elapsed_cpu_to_gpu   << " [msec]" << std::endl;
    std::cout << "kernel               : " << elapsed_kernel       << " [msec]" << std::endl;
    std::cout << "data copy(gpu->cpu)  : " << elapsed_gpu_to_cpu   << " [msec]" << std::endl;
    std::cout << "free                 : " << elapsed_free         << " [msec]" << std::endl;
    
    return 0;
}



int bbcu_eva_fp32_StochasticLut6_Backward
        (
            float const     *x_buf,
            float const     *dy_buf,
            float           *dx_buf,
            int   const     *input_index,
            float const     *W,
            float           *dW,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             binary_mode
        )
{
    int const N = 6;
    int const M = (1 << N);

    cudaDeviceProp dev;
    BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));

    cudaError_t cudaStatus0 = cudaGetLastError();
    if (cudaStatus0 != cudaSuccess) {
        fprintf(stderr, "start failed: %s\n", cudaGetErrorString(cudaStatus0));
        exit(1);
    }

    cudaDeviceSynchronize();
    auto time0 = std::chrono::system_clock::now();

    float* dev_x_buf;
    float* dev_dx_buf;
    float* dev_dx_tmp;
    float* dev_dy_buf;

    int*   dev_input_index;
    float* dev_W;
    float* dev_dW;

    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_x_buf,   input_node_size * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_dx_buf,  input_node_size * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_dx_tmp,  output_node_size * N * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_dy_buf,  output_node_size * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_input_index, output_node_size * N * sizeof(int)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_W, output_node_size * M * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_dW, output_node_size * M * sizeof(float)));
    
    cudaDeviceSynchronize();
    auto time1 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_input_index, input_index, output_node_size * N * sizeof(int), cudaMemcpyHostToDevice));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_W, W, output_node_size * M * sizeof(float), cudaMemcpyHostToDevice));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_dW, dW, output_node_size * M * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time2 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_dx_buf,  x_buf,  input_node_size * frame_size * sizeof(float), cudaMemcpyHostToDevice));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_dy_buf, dy_buf, output_node_size * frame_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time3 = std::chrono::system_clock::now();

    bbcu_fp32_StochasticLut6_Backward(
            dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
            dev_dx_tmp,
            dev_input_index,
            dev_W,
            dev_dW,
            input_node_size,
            output_node_size,
            frame_size,
            frame_stride,
            binary_mode
        );

    cudaDeviceSynchronize();
    auto time4 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(dx_buf, dev_dx_buf, input_node_size * frame_size * sizeof(float), cudaMemcpyDeviceToHost));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dW, dev_dW, output_node_size * M * sizeof(float), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    auto time5 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaFree(dev_x_buf));
    BB_CUDA_SAFE_CALL(cudaFree(dev_dx_buf));
    BB_CUDA_SAFE_CALL(cudaFree(dev_dx_tmp));
    BB_CUDA_SAFE_CALL(cudaFree(dev_dy_buf));
    BB_CUDA_SAFE_CALL(cudaFree(dev_input_index));
    BB_CUDA_SAFE_CALL(cudaFree(dev_W));
    BB_CUDA_SAFE_CALL(cudaFree(dev_dW));

    cudaDeviceSynchronize();
    auto time6 = std::chrono::system_clock::now();

    double elapsed_malloc       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count();
    double elapsed_cpu_to_gpu_p = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
    double elapsed_cpu_to_gpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
    double elapsed_kernel       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
    double elapsed_gpu_to_cpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
    double elapsed_free         = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time6-time5).count();
    std::cout << "malloc               : " << elapsed_malloc       << " [msec]" << std::endl;
    std::cout << "param copy(cpu->gpu) : " << elapsed_cpu_to_gpu_p << " [msec]" << std::endl;
    std::cout << "data copy(cpu->gpu)  : " << elapsed_cpu_to_gpu   << " [msec]" << std::endl;
    std::cout << "kernel               : " << elapsed_kernel       << " [msec]" << std::endl;
    std::cout << "data copy(gpu->cpu)  : " << elapsed_gpu_to_cpu   << " [msec]" << std::endl;
    std::cout << "free                 : " << elapsed_free         << " [msec]" << std::endl;
    
    return 0;
}




#define N                   6
#define M                   (1 << N)

#if _DEBUG
#define FRAME_SIZE          (16*28*28)
#define OUTPUT_NODE_SIZE    (8)
#define INPUT_NODE_SIZE     (8*3*3)
#else
#define FRAME_SIZE          (64*28*28)
#define OUTPUT_NODE_SIZE    (256)
#define INPUT_NODE_SIZE     (256*9)
#endif


static float   x_buf[INPUT_NODE_SIZE*FRAME_SIZE];
static float   y_buf[OUTPUT_NODE_SIZE*FRAME_SIZE];
static float   dx_buf[INPUT_NODE_SIZE*FRAME_SIZE];
static float   dy_buf[OUTPUT_NODE_SIZE*FRAME_SIZE];
static int     input_index[INPUT_NODE_SIZE*N];
static float   W[OUTPUT_NODE_SIZE*M];
static float   dW[OUTPUT_NODE_SIZE*M];


int Test_StochasticLut6_Forward(void)
{
    std::mt19937_64 mt(1);
    std::uniform_int_distribution<int>  index_rand(0, INPUT_NODE_SIZE-1);
    std::normal_distribution<float>     norm_rand(0, 1);

    for (int i = 0; i < sizeof(input_index) / sizeof(int);   ++i) { input_index[i] = index_rand(mt); }
#if 0
    for (int i = 0; i < sizeof(x_buf)       / sizeof(float); ++i) { x_buf[i]       = norm_rand(mt); }
    for (int i = 0; i < sizeof(hidden_W)    / sizeof(float); ++i) { hidden_W[i]    = norm_rand(mt); }
    for (int i = 0; i < sizeof(hidden_b)    / sizeof(float); ++i) { hidden_b[i]    = norm_rand(mt); }
    for (int i = 0; i < sizeof(output_W)    / sizeof(float); ++i) { output_W[i]    = norm_rand(mt); }
    for (int i = 0; i < sizeof(output_b)    / sizeof(float); ++i) { output_b[i]    = norm_rand(mt); }
#endif

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間

    bbcu_eva_fp32_StochasticLut6_Forward
        (
            x_buf,
            y_buf,
            input_index,
            W,
            INPUT_NODE_SIZE,
            OUTPUT_NODE_SIZE,
            FRAME_SIZE,
            FRAME_SIZE,
            1
        );
    
    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    std::cout << "totel : " << elapsed << " [msec]" << std::endl;

    return 0;
}



int Test_StochasticLut6_Backward(void)
{
    cudaDeviceProp dev;
    BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));
//  printf(" shared memory / block : %d (KB)\n", dev.sharedMemPerBlock/1024);

    std::mt19937_64 mt(1);
    std::uniform_int_distribution<int>  index_rand(0, INPUT_NODE_SIZE-1);
    std::normal_distribution<float>     norm_rand(0, 1);

    for (int i = 0; i < sizeof(input_index) / sizeof(int); ++i) { input_index[i] = index_rand(mt); }
#if 0
    for (int i = 0; i < sizeof(x_buf)  / sizeof(float); ++i)    { x_buf[i]  = norm_rand(mt); }
    for (int i = 0; i < sizeof(dy_buf) / sizeof(float); ++i)    { dy_buf[i] = norm_rand(mt); }
    for (int i = 0; i < sizeof(hidden_W) / sizeof(float); ++i)  { hidden_W[i] = norm_rand(mt); }
    for (int i = 0; i < sizeof(hidden_b) / sizeof(float); ++i)  { hidden_b[i] = norm_rand(mt); }
    for (int i = 0; i < sizeof(output_W) / sizeof(float); ++i)  { output_W[i] = norm_rand(mt); }
    for (int i = 0; i < sizeof(output_b) / sizeof(float); ++i)  { output_b[i] = norm_rand(mt); }
//  for (int i = 0; i < sizeof(hidden_dW) / sizeof(float); ++i) { hidden_W[i] = norm_rand(mt); }
//  for (int i = 0; i < sizeof(hidden_db) / sizeof(float); ++i) { hidden_b[i] = norm_rand(mt); }
//  for (int i = 0; i < sizeof(output_dW) / sizeof(float); ++i) { output_W[i] = norm_rand(mt); }
//  for (int i = 0; i < sizeof(output_db) / sizeof(float); ++i) { output_b[i] = norm_rand(mt); }
#endif

//  std::cout << "start" << std::endl;

    for (int i = 0; i < sizeof(x_buf) / sizeof(float); ++i)    { x_buf[i]  = 0; }
    for (int i = 0; i < sizeof(dy_buf) / sizeof(float); ++i)   { dy_buf[i] = 0; }

    for (int i = 0; i < sizeof(input_index) / sizeof(int); ++i) { input_index[i] = i % N; }
    for (int i = 0; i < sizeof(W) / sizeof(float); ++i)  { W[i] = 0; }

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間

    bbcu_eva_fp32_StochasticLut6_Backward
        (
            x_buf,
            dy_buf,
            dx_buf,
            input_index,
            W,
            dW,
            INPUT_NODE_SIZE,
            OUTPUT_NODE_SIZE,
            FRAME_SIZE,
            FRAME_SIZE,
            1
        );

    /*
    std::cout << "output_dW[0] : " << output_dW[0] << std::endl;
    std::cout << "output_dW[1] : " << output_dW[1] << std::endl;
    std::cout << "output_dW[2] : " << output_dW[2] << std::endl;
    std::cout << "output_db[0] : " << output_db[0] << std::endl;
    std::cout << "hidden_dW[0] : " << hidden_dW[0] << std::endl;
    std::cout << "hidden_dW[1] : " << hidden_dW[1] << std::endl;
    std::cout << "hidden_db[2] : " << hidden_dW[2] << std::endl;
    std::cout << "hidden_db[0] : " << hidden_db[0] << std::endl;

    std::cout << "in_err[0] : " << dx_buf[0] << std::endl;
    std::cout << "in_err[1] : " << dx_buf[1] << std::endl;
    std::cout << "in_err[2] : " << dx_buf[2] << std::endl;
    */

    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    std::cout << "totel : " << elapsed << " [msec]" << std::endl;

    return 0;
}

