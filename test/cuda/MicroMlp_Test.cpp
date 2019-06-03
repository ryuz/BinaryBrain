
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
            const float*    x_buf,
            float*          y_buf,
            const int*      input_index,
            const float*    hidden_W,
            const float*    hidden_b,
            const float*    output_W,
            const float*    output_b,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int const N = 6;
    int const M = 16;

    cudaDeviceSynchronize();
    auto time0 = std::chrono::system_clock::now();

    float* dev_x_buf;
    float* dev_y_buf;
    int*   dev_input_index;
    float* dev_hidden_W;
    float* dev_hidden_b;
    float* dev_output_W;
    float* dev_output_b;

    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_x_buf,       input_node_size * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_y_buf,       output_node_size * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_input_index, output_node_size * N * sizeof(int)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_hidden_W,    output_node_size * M * N * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_hidden_b,    output_node_size * M * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_output_W,    output_node_size * M * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_output_b,    output_node_size * sizeof(float)));
    
    cudaDeviceSynchronize();
    auto time1 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_input_index, input_index, output_node_size * N * sizeof(int), cudaMemcpyHostToDevice));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_hidden_W, hidden_W, output_node_size * M * N * sizeof(float), cudaMemcpyHostToDevice));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_hidden_b, hidden_b, output_node_size * M * sizeof(float), cudaMemcpyHostToDevice));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_output_W, output_W, output_node_size * M * sizeof(float), cudaMemcpyHostToDevice));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_output_b, output_b, output_node_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time2 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_x_buf, x_buf, input_node_size * frame_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time3 = std::chrono::system_clock::now();
    
    bbcu_fp32_MicroMlp6x16_Forward
        (
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_hidden_W,
            dev_hidden_b,
            dev_output_W,
            dev_output_b,
            input_node_size,
            output_node_size,
            frame_size,
            frame_stride
        );

    cudaDeviceSynchronize();
    auto time4 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(y_buf, dev_y_buf, output_node_size * frame_size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    auto time5 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaFree(dev_x_buf));
    BB_CUDA_SAFE_CALL(cudaFree(dev_y_buf));
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

//  double kernel_flops = (double)output_node_size *(double) frame_size * (M*N+M+M)*2.0 / elapsed_kernel / 1000000.0;

    std::cout << "malloc               : " << elapsed_malloc       << " [msec]" << std::endl;
    std::cout << "param copy(cpu->gpu) : " << elapsed_cpu_to_gpu_p << " [msec]" << std::endl;
    std::cout << "data copy(cpu->gpu)  : " << elapsed_cpu_to_gpu   << " [msec]" << std::endl;
//  std::cout << "kernel               : " << elapsed_kernel       << " [msec]  " << kernel_flops << " [GFLOPS]" << std::endl;
    std::cout << "kernel               : " << elapsed_kernel       << " [msec]" << std::endl;
    std::cout << "data copy(gpu->cpu)  : " << elapsed_gpu_to_cpu   << " [msec]" << std::endl;
    std::cout << "free                 : " << elapsed_free         << " [msec]" << std::endl;
    
    return 0;
}



int bbcu_eva_MicroMlp6x16_Backward
        (
            const float*    x_buf,
            float*          dx_buf,
            float*          dy_buf,
            const int*      input_index,
            const float*    hidden_W,
            const float*    hidden_b,
            float*          hidden_dW,
            float*          hidden_db,
            const float*    output_W,
            const float*    output_b,
            float*          output_dW,
            float*          output_db,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride
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

    float* dev_x_buf;
    float* dev_dx_buf;
    float* dev_dx_tmp;
    float* dev_dy_buf;

    int*   dev_input_index;
    float* dev_hidden_W;
    float* dev_hidden_b;
    float* dev_output_W;
    float* dev_output_b;
    float* dev_hidden_dW;
    float* dev_hidden_db;
    float* dev_output_dW;
    float* dev_output_db;

    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_x_buf,   input_node_size * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_dx_buf,  input_node_size * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_dx_tmp,  output_node_size * N * frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_dy_buf,  output_node_size * frame_size * sizeof(float)));
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

    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_dx_buf,  x_buf,  input_node_size * frame_size * sizeof(float), cudaMemcpyHostToDevice));
    BB_CUDA_SAFE_CALL(cudaMemcpy(dev_dy_buf, dy_buf, output_node_size * frame_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time3 = std::chrono::system_clock::now();

   bbcu_fp32_MicroMlp6x16_Backward(
            dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
            dev_dx_tmp,
            dev_input_index,
            dev_hidden_W,
            dev_hidden_b,
            dev_hidden_dW,
            dev_hidden_db,
            dev_output_W,
            dev_output_b,
            dev_output_dW,
            dev_output_db,
            input_node_size,
            output_node_size,
            frame_size,
            frame_stride
        );

    cudaDeviceSynchronize();
    auto time4 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(dx_buf,    dev_dx_buf, input_node_size * frame_size * sizeof(float), cudaMemcpyDeviceToHost));
    BB_CUDA_SAFE_CALL(cudaMemcpy(hidden_dW, dev_hidden_dW, output_node_size * M * N * sizeof(float), cudaMemcpyDeviceToHost));
    BB_CUDA_SAFE_CALL(cudaMemcpy(hidden_db, dev_hidden_db, output_node_size * M * sizeof(float), cudaMemcpyDeviceToHost));
    BB_CUDA_SAFE_CALL(cudaMemcpy(output_dW, dev_output_dW, output_node_size * M * sizeof(float), cudaMemcpyDeviceToHost));
    BB_CUDA_SAFE_CALL(cudaMemcpy(output_db, dev_output_db, output_node_size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    auto time5 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaFree(dev_x_buf));
    BB_CUDA_SAFE_CALL(cudaFree(dev_dx_buf));
    BB_CUDA_SAFE_CALL(cudaFree(dev_dx_tmp));
    BB_CUDA_SAFE_CALL(cudaFree(dev_dy_buf));
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
    std::cout << "malloc               : " << elapsed_malloc       << " [msec]" << std::endl;
    std::cout << "param copy(cpu->gpu) : " << elapsed_cpu_to_gpu_p << " [msec]" << std::endl;
    std::cout << "data copy(cpu->gpu)  : " << elapsed_cpu_to_gpu   << " [msec]" << std::endl;
    std::cout << "kernel               : " << elapsed_kernel       << " [msec]" << std::endl;
    std::cout << "data copy(gpu->cpu)  : " << elapsed_gpu_to_cpu   << " [msec]" << std::endl;
    std::cout << "free                 : " << elapsed_free         << " [msec]" << std::endl;
    
    return 0;
}




#define N                   6
#define M                   16

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
static float   hidden_W[OUTPUT_NODE_SIZE*M*N];
static float   hidden_b[OUTPUT_NODE_SIZE*M];
static float   output_W[OUTPUT_NODE_SIZE*M];
static float   output_b[OUTPUT_NODE_SIZE];
static float   hidden_dW[OUTPUT_NODE_SIZE*M*N];
static float   hidden_db[OUTPUT_NODE_SIZE*M];
static float   output_dW[OUTPUT_NODE_SIZE*M];
static float   output_db[OUTPUT_NODE_SIZE];


int Test_MicroMlp_Forward(void)
{
    cudaDeviceProp dev;
    BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));
    printf(" shared memory / block : %d (KB)\n", (int)dev.sharedMemPerBlock/1024);

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

    bbcu_eva_MicroMlp6x16_Forward
        (
            x_buf,
            y_buf,
            input_index,
            hidden_W,
            hidden_b,
            output_W,
            output_b,
            INPUT_NODE_SIZE,
            OUTPUT_NODE_SIZE,
            FRAME_SIZE,
            FRAME_SIZE
        );
    
    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    std::cout << "totel : " << elapsed << " [msec]" << std::endl;

    return 0;
}



int Test_MicroMlp_Backward(void)
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
    for (int i = 0; i < sizeof(hidden_W) / sizeof(float); ++i)  { hidden_W[i] = 0; }
    for (int i = 0; i < sizeof(hidden_b) / sizeof(float); ++i)  { hidden_b[i] = 0; }
    for (int i = 0; i < sizeof(output_W) / sizeof(float); ++i)  { output_W[i] = 0; }
    for (int i = 0; i < sizeof(output_b) / sizeof(float); ++i)  { output_b[i] = 0; }

    x_buf[0*N + 0] = 1;
    x_buf[0*N + 1] = 2;
    x_buf[0*N + 2] = 3;
    x_buf[0*N + 3] = 4;
    x_buf[0*N + 4] = 5;
    x_buf[0*N + 5] = 6;

    hidden_W[0*N + 0] = 1;
    hidden_W[0*N + 1] = 2;
    hidden_W[0*N + 2] = 3;
    hidden_W[0*N + 3] = 4;
    hidden_W[0*N + 4] = 5;
    hidden_W[0*N + 5] = 6;

    output_W[0*N + 0] = 1;

    dy_buf[0] = -1;

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間

    bbcu_eva_MicroMlp6x16_Backward
        (
            x_buf,
            dx_buf,
            dy_buf,
            input_index,
            hidden_W,
            hidden_b,
            hidden_dW,
            hidden_db,
            output_W,
            output_b,
            output_dW,
            output_db,
            INPUT_NODE_SIZE,
            OUTPUT_NODE_SIZE,
            FRAME_SIZE,
            FRAME_SIZE
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

