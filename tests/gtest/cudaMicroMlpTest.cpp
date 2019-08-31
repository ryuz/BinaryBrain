#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gtest/gtest.h"

#include "bb/NeuralNetStackedMicroAffine.h"
#include "bbcu/bbcu.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
    net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
    net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
    net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
    net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}



#define N                   6
#define M                   16

#if _DEBUG

#define FRAME_SIZE          (512)
#define INPUT_NODE_SIZE     (6)
#define OUTPUT_NODE_SIZE    (2)

#elif 0

#define FRAME_SIZE          (512)
#define INPUT_NODE_SIZE     (6)
#define OUTPUT_NODE_SIZE    (2)

//#define FRAME_SIZE            (64*28*28)
//#define INPUT_NODE_SIZE       (6)
//#define OUTPUT_NODE_SIZE  (2*3)

#else

#define FRAME_SIZE          (128*28*28)
#define INPUT_NODE_SIZE     (128*3*3)
#define OUTPUT_NODE_SIZE    (256)

#endif


/*
float   in_sig[INPUT_NODE_SIZE*FRAME_SIZE];
float   out_sig[OUTPUT_NODE_SIZE*FRAME_SIZE];
int     input_index[OUTPUT_NODE_SIZE*N];
float   hidden_W[OUTPUT_NODE_SIZE*M*N];
float   hidden_b[OUTPUT_NODE_SIZE*M];
float   output_W[OUTPUT_NODE_SIZE*M];
float   output_b[OUTPUT_NODE_SIZE];
*/

#if 1

TEST(cudaMicroMlpTest, test_cudaMicroMlp2)
{
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    
#if 0
    {
        float   test_src_buf[10 * 100];
        float   test_dst_buf[10];
        for ( int y = 0; y < 10; ++y ) {
            for ( int x = 0; x < 100; ++x ) {
                test_src_buf[y*100 + x] = y*1 + x + 1;
            }
        }
        horizontal_sum(test_src_buf, test_dst_buf, 100, 10);
        for ( int y = 0; y < 10; ++y ) {
            std::cout << test_dst_buf[y] << std::endl;
        }
    }
#endif

    std::vector<float>  in_sig(INPUT_NODE_SIZE*FRAME_SIZE);
    std::vector<float>  out_sig(OUTPUT_NODE_SIZE*FRAME_SIZE);
    std::vector<int>    input_index(OUTPUT_NODE_SIZE*N);
    std::vector<float>  hidden_W(OUTPUT_NODE_SIZE*M*N);
    std::vector<float>  hidden_b(OUTPUT_NODE_SIZE*M);
    std::vector<float>  output_W(OUTPUT_NODE_SIZE*M);
    std::vector<float>  output_b(OUTPUT_NODE_SIZE);
    

    bb::NeuralNetStackedMicroAffine<N, M> umlp_cpu(INPUT_NODE_SIZE, OUTPUT_NODE_SIZE);
    umlp_cpu.SetBatchSize(FRAME_SIZE);
    testSetupLayerBuffer(umlp_cpu);

    std::mt19937_64 mt(1);
    std::uniform_int_distribution<int>  index_rand(0, INPUT_NODE_SIZE-1);
    std::uniform_int_distribution<int>  norm_rand(-10, 10);
//  std::normal_distribution<float>     norm_rand(0, 1);
    
    
    for (size_t i = 0; i < in_sig.size(); ++i) { in_sig[i] = norm_rand(mt); }
//  for (size_t i = 0; i < input_index.size(); ++i) { input_index[i] = index_rand(mt); }
    for (size_t i = 0; i < input_index.size(); ++i) { input_index[i] = i % 6; }
    for (size_t i = 0; i < hidden_W.size(); ++i) { hidden_W[i] = norm_rand(mt); }
    for (size_t i = 0; i < hidden_b.size(); ++i) { hidden_b[i] = norm_rand(mt); }
    for (size_t i = 0; i < output_W.size(); ++i) { output_W[i] = norm_rand(mt); }
    for (size_t i = 0; i < output_b.size(); ++i) { output_b[i] = -norm_rand(mt); }
    

    std::cout << "\n\n";
    std::cout << "input size : " << INPUT_NODE_SIZE << "\n";
    std::cout <<  OUTPUT_NODE_SIZE << " node[6x16] x " << FRAME_SIZE << " frames\n\n";

    std::cout << "[GPU GT1030]" << std::endl;
    bbcu_eva_MicroMlp6x16_Forward
        (
            &in_sig[0],
            &out_sig[0],
            INPUT_NODE_SIZE,
            OUTPUT_NODE_SIZE,
            FRAME_SIZE,
            &input_index[0],
            &hidden_W[0],
            &hidden_b[0],
            &output_W[0],
            &output_b[0]
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

    auto fw_time0 = std::chrono::system_clock::now();

    umlp_cpu.Forward();

    auto fw_time1 = std::chrono::system_clock::now();

    std::cout << "\n\n[CPU]" << std::endl;
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(fw_time1-fw_time0).count();
    std::cout << "OpenMP + AVX2 : " << elapsed << " [msec]" << std::endl;
//  double flops = (double)OUTPUT_NODE_SIZE * (double)FRAME_SIZE * (16.0+6.0)*2.0 / elapsed / 1000000.0;
    double flops = (double)OUTPUT_NODE_SIZE * (double)FRAME_SIZE * (6.0*16.0+16.0+16.0)*2.0 / elapsed / 1000000.0;
    std::cout << "      " << flops << " [GFLOPS]  (" << flops / 435.2 * 100.0 << "% [peak 435.2 GFLOPS])" << std::endl;

    std::cout << "\n\n";
    
    for (int i = 0; i < OUTPUT_NODE_SIZE; i++) {
        for (int j = 0; j < FRAME_SIZE; j++) {
            EXPECT_EQ(out_sig_buf.GetReal(j, i), out_sig[FRAME_SIZE*i + j]);
//          std::cout << out_sig_buf.GetReal(j, i) << " " << out_sig[FRAME_SIZE*i + j] << std::endl;
        }
    }


    /// backward ////
#if 1

    // GPU
    std::vector<float>  in_err(INPUT_NODE_SIZE*FRAME_SIZE);
    std::vector<float>  out_err(OUTPUT_NODE_SIZE*FRAME_SIZE);
    std::vector<float>  hidden_dW(OUTPUT_NODE_SIZE*M*N);
    std::vector<float>  hidden_db(OUTPUT_NODE_SIZE*M);
    std::vector<float>  output_dW(OUTPUT_NODE_SIZE*M);
    std::vector<float>  output_db(OUTPUT_NODE_SIZE);

    for (size_t i = 0; i < out_err.size(); ++i) { out_err[i] = norm_rand(mt); }

    auto in_err_buf  = umlp_cpu.GetInputErrorBuffer();
    auto out_err_buf = umlp_cpu.GetOutputErrorBuffer();


    std::cout << "<<<Backward>>>\n" << std::endl;
    std::cout << "[GPU]" << std::endl;
    bbcu_eva_MicroMlp6x16_Backward
        (
            &in_sig[0],
            &in_err[0],
            &out_err[0],
            INPUT_NODE_SIZE,
            OUTPUT_NODE_SIZE,
            FRAME_SIZE,
            &input_index[0],
            &hidden_W[0],
            &hidden_b[0],
            &hidden_dW[0],
            &hidden_db[0],
            &output_W[0],
            &output_b[0],
            &output_dW[0],
            &output_db[0]
        );


    // CPU
    for (int i = 0; i < OUTPUT_NODE_SIZE; i++) {
        for (int j = 0; j < FRAME_SIZE; j++) {
            out_err_buf.SetReal(j, i, out_err[FRAME_SIZE*i + j]);
        }
    }
    
    auto bw_time0 = std::chrono::system_clock::now();
    umlp_cpu.Backward();
    auto bw_time1 = std::chrono::system_clock::now();

    std::cout << "[CPU]" << std::endl;
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(bw_time1-bw_time0).count();
    std::cout << "OpenMP + AVX2 : " << elapsed << " [msec]" << std::endl;
    std::cout << "\n\n";


    for (int i = 0; i < OUTPUT_NODE_SIZE; i++) {
        for (int j = 0; j < N; j++) {
            umlp_cpu.SetNodeInput(i, j, input_index[i*N+j]);
        }
    }

    for (int i = 0; i < OUTPUT_NODE_SIZE; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                EXPECT_EQ(umlp_cpu.dW0(i, j, k), hidden_dW[i*(M*N) + j*N + k]);
            }
            EXPECT_EQ(umlp_cpu.db0(i, j), hidden_db[i*M + j]);
        }

        for (int j = 0; j < M; j++) {
            EXPECT_EQ(umlp_cpu.dW1(i, j), output_dW[i*M + j]);
        }
        EXPECT_EQ(umlp_cpu.db1(i), output_db[i]);
    }

    for (int i = 0; i < INPUT_NODE_SIZE; i++) {
        for (int j = 0; j < FRAME_SIZE; j++) {
            EXPECT_EQ(in_err_buf.GetReal(j, i), in_err[FRAME_SIZE*i + j]);
        }
    }
#endif
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

//  in_sig[frame_size*0 + 0] = 2;
//  in_sig[frame_size*1 + 0] = 2;
//  in_sig[frame_size*2 + 0] = 3;
//  in_sig[frame_size*3 + 0] = 4;
//  in_sig[frame_size*4 + 0] = 5;
//  in_sig[frame_size*5 + 0] = 6;
//  in_sig[frame_size*0 + 1] = 11;
//  in_sig[frame_size*1 + 1] = 12;
//  in_sig[frame_size*2 + 1] = 13;
//  in_sig[frame_size*3 + 1] = 14;
//  in_sig[frame_size*4 + 1] = 15;
//  in_sig[frame_size*5 + 1] = 16;

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

