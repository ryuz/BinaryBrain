#include "mnist_sample.h"
#include "MnistDifferentiableLutHls.h"


void MnistDepthwiseAffine_layer(int y[10], const ap_uint<10*DWA_DEPTH> x)
{
    for ( int i = 0; i < 10; ++i ) {
        #pragma HLS unroll
        int sum = (int)b_tbl[i];
        for ( int j = 0; j < DWA_DEPTH; ++j ) {
            #pragma HLS unroll
            sum += (int)x[i*DWA_DEPTH + j] * (int)W_tbl[i][j];
        }
        y[i] = sum;
    }
}


// kernel
void mnist_sample(
            const ap_uint<1>    in[28*28],
            ap_uint<4>          out[1]
        )
{
    // input
    ap_uint<28*28>  x0;
    for ( int i = 0; i < 28*28; ++i ) {
        x0[i] = in[i];
    }
    auto x1 = MnistLut_layer1(x0);
    auto x2 = MnistLut_layer2(x1);
    auto x3 = MnistLut_layer3(x2);

    // Depthwise Affine
    int y[10];
    MnistDepthwiseAffine_layer(y, x3);

    // argmax
    int         max_val = -32768;
    ap_uint<4>  max_idx = 0;
    for ( int i = 0; i < 10; ++i ) {
        if ( y[i] > max_val ) {
            max_val = y[i];
            max_idx = i;
        }
    }

    // output
    out[0] = max_idx;
}

