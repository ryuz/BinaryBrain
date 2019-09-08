
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// generic
template<int N, typename T=float, int MAX_NODE_UNIT=32>
struct StochasticLut
{
    static __device__ __forceinline__ T NodeForward
        (
            int         node_id,
            T           xp[],
            T   const   W[][MAX_NODE_UNIT]
        )
    {
        T y = 0;
        for (int i = 0; i < (1 << N); ++i) {
            T w = W[i][node_id];
            for (int j = 0; j < N; ++j) {
                w *= ((i >> j) & 1) ? xp[j] : ((T)1.0 - xp[j]);
            }
            y += w;
        }

        return y;
    }

    static __device__ __forceinline__ void NodeBackward
        (
            int         node_id,
            T   const   xp[],
            T           dy,
            T           *dx_ptr,
            T   const   W[][MAX_NODE_UNIT],
            T           dW[],
            int         frame_stride
        )
    {
    }
};


// LUT6
template<typename T, int MAX_NODE_UNIT>
struct StochasticLut<6, T, MAX_NODE_UNIT>
{
    static __device__ __forceinline__ T NodeForward
            (
                int             node_id,
                T           xp[],
                T   const   W[][MAX_NODE_UNIT]
            )
    {
        T   xn[6];
        for ( int i = 0; i < 6; ++i) {
            xn[i] = 1.0 - xp[i];
        }

        T x0_00 = xn[1] * xn[0];
        T x0_01 = xn[1] * xp[0];
        T x0_10 = xp[1] * xn[0];
        T x0_11 = xp[1] * xp[0];
        T x1_00 = xn[3] * xn[2];
        T x1_01 = xn[3] * xp[2];
        T x1_10 = xp[3] * xn[2];
        T x1_11 = xp[3] * xp[2];
        T x2_00 = xn[5] * xn[4];
        T x2_01 = xn[5] * xp[4];
        T x2_10 = xp[5] * xn[4];
        T x2_11 = xp[5] * xp[4];

        T y = 0;
        T x2_00_x1_00 = x2_00 * x1_00;
        y += W[0 ][node_id] * x2_00_x1_00 * x0_00;
        y += W[1 ][node_id] * x2_00_x1_00 * x0_01;
        y += W[2 ][node_id] * x2_00_x1_00 * x0_10;
        y += W[3 ][node_id] * x2_00_x1_00 * x0_11;
        T x2_00_x1_01 = x2_00 * x1_01;
        y += W[4 ][node_id] * x2_00_x1_01 * x0_00;
        y += W[5 ][node_id] * x2_00_x1_01 * x0_01;
        y += W[6 ][node_id] * x2_00_x1_01 * x0_10;
        y += W[7 ][node_id] * x2_00_x1_01 * x0_11;
        T x2_00_x1_10 = x2_00 * x1_10;
        y += W[8 ][node_id] * x2_00_x1_10 * x0_00;
        y += W[9 ][node_id] * x2_00_x1_10 * x0_01;
        y += W[10][node_id] * x2_00_x1_10 * x0_10;
        y += W[11][node_id] * x2_00_x1_10 * x0_11;
        T x2_00_x1_11 = x2_00 * x1_11;
        y += W[12][node_id] * x2_00_x1_11 * x0_00;
        y += W[13][node_id] * x2_00_x1_11 * x0_01;
        y += W[14][node_id] * x2_00_x1_11 * x0_10;
        y += W[15][node_id] * x2_00_x1_11 * x0_11;
        T x2_01_x1_00 = x2_01 * x1_00;
        y += W[16][node_id] * x2_01_x1_00 * x0_00;
        y += W[17][node_id] * x2_01_x1_00 * x0_01;
        y += W[18][node_id] * x2_01_x1_00 * x0_10;
        y += W[19][node_id] * x2_01_x1_00 * x0_11;
        T x2_01_x1_01 = x2_01 * x1_01;
        y += W[20][node_id] * x2_01_x1_01 * x0_00;
        y += W[21][node_id] * x2_01_x1_01 * x0_01;
        y += W[22][node_id] * x2_01_x1_01 * x0_10;
        y += W[23][node_id] * x2_01_x1_01 * x0_11;
        T x2_01_x1_10 = x2_01 * x1_10;
        y += W[24][node_id] * x2_01_x1_10 * x0_00;
        y += W[25][node_id] * x2_01_x1_10 * x0_01;
        y += W[26][node_id] * x2_01_x1_10 * x0_10;
        y += W[27][node_id] * x2_01_x1_10 * x0_11;
        T x2_01_x1_11 = x2_01 * x1_11;
        y += W[28][node_id] * x2_01_x1_11 * x0_00;
        y += W[29][node_id] * x2_01_x1_11 * x0_01;
        y += W[30][node_id] * x2_01_x1_11 * x0_10;
        y += W[31][node_id] * x2_01_x1_11 * x0_11;
        T x2_10_x1_00 = x2_10 * x1_00;
        y += W[32][node_id] * x2_10_x1_00 * x0_00;
        y += W[33][node_id] * x2_10_x1_00 * x0_01;
        y += W[34][node_id] * x2_10_x1_00 * x0_10;
        y += W[35][node_id] * x2_10_x1_00 * x0_11;
        T x2_10_x1_01 = x2_10 * x1_01;
        y += W[36][node_id] * x2_10_x1_01 * x0_00;
        y += W[37][node_id] * x2_10_x1_01 * x0_01;
        y += W[38][node_id] * x2_10_x1_01 * x0_10;
        y += W[39][node_id] * x2_10_x1_01 * x0_11;
        T x2_10_x1_10 = x2_10 * x1_10;
        y += W[40][node_id] * x2_10_x1_10 * x0_00;
        y += W[41][node_id] * x2_10_x1_10 * x0_01;
        y += W[42][node_id] * x2_10_x1_10 * x0_10;
        y += W[43][node_id] * x2_10_x1_10 * x0_11;
        T x2_10_x1_11 = x2_10 * x1_11;
        y += W[44][node_id] * x2_10_x1_11 * x0_00;
        y += W[45][node_id] * x2_10_x1_11 * x0_01;
        y += W[46][node_id] * x2_10_x1_11 * x0_10;
        y += W[47][node_id] * x2_10_x1_11 * x0_11;
        T x2_11_x1_00 = x2_11 * x1_00;
        y += W[48][node_id] * x2_11_x1_00 * x0_00;
        y += W[49][node_id] * x2_11_x1_00 * x0_01;
        y += W[50][node_id] * x2_11_x1_00 * x0_10;
        y += W[51][node_id] * x2_11_x1_00 * x0_11;
        T x2_11_x1_01 = x2_11 * x1_01;
        y += W[52][node_id] * x2_11_x1_01 * x0_00;
        y += W[53][node_id] * x2_11_x1_01 * x0_01;
        y += W[54][node_id] * x2_11_x1_01 * x0_10;
        y += W[55][node_id] * x2_11_x1_01 * x0_11;
        T x2_11_x1_10 = x2_11 * x1_10;
        y += W[56][node_id] * x2_11_x1_10 * x0_00;
        y += W[57][node_id] * x2_11_x1_10 * x0_01;
        y += W[58][node_id] * x2_11_x1_10 * x0_10;
        y += W[59][node_id] * x2_11_x1_10 * x0_11;
        T x2_11_x1_11 = x2_11 * x1_11;
        y += W[60][node_id] * x2_11_x1_11 * x0_00;
        y += W[61][node_id] * x2_11_x1_11 * x0_01;
        y += W[62][node_id] * x2_11_x1_11 * x0_10;
        y += W[63][node_id] * x2_11_x1_11 * x0_11;

        // clamp
        y = max(0.0, y);
        y = min(1.0, y);

        return y;
    }


    static __device__ __forceinline__ void NodeBackward
        (
            int         node_id,
            T   const   xp[],
            T           dy,
            T           *dx_ptr,
            T   const   W[][MAX_NODE_UNIT],
            T           dW[],
            int             frame_stride
        )
    {
        T   xn[6];
        for (int i = 0; i < 6; ++i) {
            xn[i] = 1.0 - xp[i];
        }

        T x0_00 = xn[1] * xn[0];
        T x0_01 = xn[1] * xp[0];
        T x0_10 = xp[1] * xn[0];
        T x0_11 = xp[1] * xp[0];
        T x1_00 = xn[3] * xn[2];
        T x1_01 = xn[3] * xp[2];
        T x1_10 = xp[3] * xn[2];
        T x1_11 = xp[3] * xp[2];
        T x2_00 = xn[5] * xn[4];
        T x2_01 = xn[5] * xp[4];
        T x2_10 = xp[5] * xn[4];
        T x2_11 = xp[5] * xp[4];

        T  x2_00_x1_00 =  x2_00 * x1_00;
        T  x2_00_x1_01 =  x2_00 * x1_01;
        T  x2_00_x1_10 =  x2_00 * x1_10;
        T  x2_00_x1_11 =  x2_00 * x1_11;
        T  x2_01_x1_00 =  x2_01 * x1_00;
        T  x2_01_x1_01 =  x2_01 * x1_01;
        T  x2_01_x1_10 =  x2_01 * x1_10;
        T  x2_01_x1_11 =  x2_01 * x1_11;
        T  x2_10_x1_00 =  x2_10 * x1_00;
        T  x2_10_x1_01 =  x2_10 * x1_01;
        T  x2_10_x1_10 =  x2_10 * x1_10;
        T  x2_10_x1_11 =  x2_10 * x1_11;
        T  x2_11_x1_00 =  x2_11 * x1_00;
        T  x2_11_x1_01 =  x2_11 * x1_01;
        T  x2_11_x1_10 =  x2_11 * x1_10;
        T  x2_11_x1_11 =  x2_11 * x1_11;

        dW[ 0] += x2_00_x1_00 * x0_00 * dy;
        dW[ 1] += x2_00_x1_00 * x0_01 * dy;
        dW[ 2] += x2_00_x1_00 * x0_10 * dy;
        dW[ 3] += x2_00_x1_00 * x0_11 * dy;
        dW[ 4] += x2_00_x1_01 * x0_00 * dy;
        dW[ 5] += x2_00_x1_01 * x0_01 * dy;
        dW[ 6] += x2_00_x1_01 * x0_10 * dy;
        dW[ 7] += x2_00_x1_01 * x0_11 * dy;
        dW[ 8] += x2_00_x1_10 * x0_00 * dy;
        dW[ 9] += x2_00_x1_10 * x0_01 * dy;
        dW[10] += x2_00_x1_10 * x0_10 * dy;
        dW[11] += x2_00_x1_10 * x0_11 * dy;
        dW[12] += x2_00_x1_11 * x0_00 * dy;
        dW[13] += x2_00_x1_11 * x0_01 * dy;
        dW[14] += x2_00_x1_11 * x0_10 * dy;
        dW[15] += x2_00_x1_11 * x0_11 * dy;
        dW[16] += x2_01_x1_00 * x0_00 * dy;
        dW[17] += x2_01_x1_00 * x0_01 * dy;
        dW[18] += x2_01_x1_00 * x0_10 * dy;
        dW[19] += x2_01_x1_00 * x0_11 * dy;
        dW[20] += x2_01_x1_01 * x0_00 * dy;
        dW[21] += x2_01_x1_01 * x0_01 * dy;
        dW[22] += x2_01_x1_01 * x0_10 * dy;
        dW[23] += x2_01_x1_01 * x0_11 * dy;
        dW[24] += x2_01_x1_10 * x0_00 * dy;
        dW[25] += x2_01_x1_10 * x0_01 * dy;
        dW[26] += x2_01_x1_10 * x0_10 * dy;
        dW[27] += x2_01_x1_10 * x0_11 * dy;
        dW[28] += x2_01_x1_11 * x0_00 * dy;
        dW[29] += x2_01_x1_11 * x0_01 * dy;
        dW[30] += x2_01_x1_11 * x0_10 * dy;
        dW[31] += x2_01_x1_11 * x0_11 * dy;
        dW[32] += x2_10_x1_00 * x0_00 * dy;
        dW[33] += x2_10_x1_00 * x0_01 * dy;
        dW[34] += x2_10_x1_00 * x0_10 * dy;
        dW[35] += x2_10_x1_00 * x0_11 * dy;
        dW[36] += x2_10_x1_01 * x0_00 * dy;
        dW[37] += x2_10_x1_01 * x0_01 * dy;
        dW[38] += x2_10_x1_01 * x0_10 * dy;
        dW[39] += x2_10_x1_01 * x0_11 * dy;
        dW[40] += x2_10_x1_10 * x0_00 * dy;
        dW[41] += x2_10_x1_10 * x0_01 * dy;
        dW[42] += x2_10_x1_10 * x0_10 * dy;
        dW[43] += x2_10_x1_10 * x0_11 * dy;
        dW[44] += x2_10_x1_11 * x0_00 * dy;
        dW[45] += x2_10_x1_11 * x0_01 * dy;
        dW[46] += x2_10_x1_11 * x0_10 * dy;
        dW[47] += x2_10_x1_11 * x0_11 * dy;
        dW[48] += x2_11_x1_00 * x0_00 * dy;
        dW[49] += x2_11_x1_00 * x0_01 * dy;
        dW[50] += x2_11_x1_00 * x0_10 * dy;
        dW[51] += x2_11_x1_00 * x0_11 * dy;
        dW[52] += x2_11_x1_01 * x0_00 * dy;
        dW[53] += x2_11_x1_01 * x0_01 * dy;
        dW[54] += x2_11_x1_01 * x0_10 * dy;
        dW[55] += x2_11_x1_01 * x0_11 * dy;
        dW[56] += x2_11_x1_10 * x0_00 * dy;
        dW[57] += x2_11_x1_10 * x0_01 * dy;
        dW[58] += x2_11_x1_10 * x0_10 * dy;
        dW[59] += x2_11_x1_10 * x0_11 * dy;
        dW[60] += x2_11_x1_11 * x0_00 * dy;
        dW[61] += x2_11_x1_11 * x0_01 * dy;
        dW[62] += x2_11_x1_11 * x0_10 * dy;
        dW[63] += x2_11_x1_11 * x0_11 * dy;

        T  x2_00_x0_00 =  x2_00 * x0_00;
        T  x2_00_x0_01 =  x2_00 * x0_01;
        T  x2_00_x0_10 =  x2_00 * x0_10;
        T  x2_00_x0_11 =  x2_00 * x0_11;
        T  x2_01_x0_00 =  x2_01 * x0_00;
        T  x2_01_x0_01 =  x2_01 * x0_01;
        T  x2_01_x0_10 =  x2_01 * x0_10;
        T  x2_01_x0_11 =  x2_01 * x0_11;
        T  x2_10_x0_00 =  x2_10 * x0_00;
        T  x2_10_x0_01 =  x2_10 * x0_01;
        T  x2_10_x0_10 =  x2_10 * x0_10;
        T  x2_10_x0_11 =  x2_10 * x0_11;
        T  x2_11_x0_00 =  x2_11 * x0_00;
        T  x2_11_x0_01 =  x2_11 * x0_01;
        T  x2_11_x0_10 =  x2_11 * x0_10;
        T  x2_11_x0_11 =  x2_11 * x0_11;

        T  x1_00_x0_00 =  x1_00 * x0_00;
        T  x1_00_x0_01 =  x1_00 * x0_01;
        T  x1_00_x0_10 =  x1_00 * x0_10;
        T  x1_00_x0_11 =  x1_00 * x0_11;
        T  x1_01_x0_00 =  x1_01 * x0_00;
        T  x1_01_x0_01 =  x1_01 * x0_01;
        T  x1_01_x0_10 =  x1_01 * x0_10;
        T  x1_01_x0_11 =  x1_01 * x0_11;
        T  x1_10_x0_00 =  x1_10 * x0_00;
        T  x1_10_x0_01 =  x1_10 * x0_01;
        T  x1_10_x0_10 =  x1_10 * x0_10;
        T  x1_10_x0_11 =  x1_10 * x0_11;
        T  x1_11_x0_00 =  x1_11 * x0_00;
        T  x1_11_x0_01 =  x1_11 * x0_01;
        T  x1_11_x0_10 =  x1_11 * x0_10;
        T  x1_11_x0_11 =  x1_11 * x0_11;


        T dxi;
        T dx0_00 = 0;
        T dx0_01 = 0;
        T dx0_10 = 0;
        T dx0_11 = 0;
        T dx1_00 = 0;
        T dx1_01 = 0;
        T dx1_10 = 0;
        T dx1_11 = 0;
        T dx2_00 = 0;
        T dx2_01 = 0;
        T dx2_10 = 0;
        T dx2_11 = 0;
        dxi = W[ 0][node_id];  dx0_00 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_00_x0_00;
        dxi = W[ 1][node_id];  dx0_01 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_00_x0_01;
        dxi = W[ 2][node_id];  dx0_10 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_00_x0_10;
        dxi = W[ 3][node_id];  dx0_11 += dxi * x2_00_x1_00;  dx1_00 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_00_x0_11;
        dxi = W[ 4][node_id];  dx0_00 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_01_x0_00;
        dxi = W[ 5][node_id];  dx0_01 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_01_x0_01;
        dxi = W[ 6][node_id];  dx0_10 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_01_x0_10;
        dxi = W[ 7][node_id];  dx0_11 += dxi * x2_00_x1_01;  dx1_01 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_01_x0_11;
        dxi = W[ 8][node_id];  dx0_00 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_10_x0_00;
        dxi = W[ 9][node_id];  dx0_01 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_10_x0_01;
        dxi = W[10][node_id];  dx0_10 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_10_x0_10;
        dxi = W[11][node_id];  dx0_11 += dxi * x2_00_x1_10;  dx1_10 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_10_x0_11;
        dxi = W[12][node_id];  dx0_00 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_00;  dx2_00 += dxi * x1_11_x0_00;
        dxi = W[13][node_id];  dx0_01 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_01;  dx2_00 += dxi * x1_11_x0_01;
        dxi = W[14][node_id];  dx0_10 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_10;  dx2_00 += dxi * x1_11_x0_10;
        dxi = W[15][node_id];  dx0_11 += dxi * x2_00_x1_11;  dx1_11 += dxi * x2_00_x0_11;  dx2_00 += dxi * x1_11_x0_11;
        dxi = W[16][node_id];  dx0_00 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_00_x0_00;
        dxi = W[17][node_id];  dx0_01 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_00_x0_01;
        dxi = W[18][node_id];  dx0_10 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_00_x0_10;
        dxi = W[19][node_id];  dx0_11 += dxi * x2_01_x1_00;  dx1_00 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_00_x0_11;
        dxi = W[20][node_id];  dx0_00 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_01_x0_00;
        dxi = W[21][node_id];  dx0_01 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_01_x0_01;
        dxi = W[22][node_id];  dx0_10 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_01_x0_10;
        dxi = W[23][node_id];  dx0_11 += dxi * x2_01_x1_01;  dx1_01 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_01_x0_11;
        dxi = W[24][node_id];  dx0_00 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_10_x0_00;
        dxi = W[25][node_id];  dx0_01 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_10_x0_01;
        dxi = W[26][node_id];  dx0_10 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_10_x0_10;
        dxi = W[27][node_id];  dx0_11 += dxi * x2_01_x1_10;  dx1_10 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_10_x0_11;
        dxi = W[28][node_id];  dx0_00 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_00;  dx2_01 += dxi * x1_11_x0_00;
        dxi = W[29][node_id];  dx0_01 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_01;  dx2_01 += dxi * x1_11_x0_01;
        dxi = W[30][node_id];  dx0_10 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_10;  dx2_01 += dxi * x1_11_x0_10;
        dxi = W[31][node_id];  dx0_11 += dxi * x2_01_x1_11;  dx1_11 += dxi * x2_01_x0_11;  dx2_01 += dxi * x1_11_x0_11;
        dxi = W[32][node_id];  dx0_00 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_00_x0_00;
        dxi = W[33][node_id];  dx0_01 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_00_x0_01;
        dxi = W[34][node_id];  dx0_10 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_00_x0_10;
        dxi = W[35][node_id];  dx0_11 += dxi * x2_10_x1_00;  dx1_00 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_00_x0_11;
        dxi = W[36][node_id];  dx0_00 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_01_x0_00;
        dxi = W[37][node_id];  dx0_01 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_01_x0_01;
        dxi = W[38][node_id];  dx0_10 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_01_x0_10;
        dxi = W[39][node_id];  dx0_11 += dxi * x2_10_x1_01;  dx1_01 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_01_x0_11;
        dxi = W[40][node_id];  dx0_00 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_10_x0_00;
        dxi = W[41][node_id];  dx0_01 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_10_x0_01;
        dxi = W[42][node_id];  dx0_10 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_10_x0_10;
        dxi = W[43][node_id];  dx0_11 += dxi * x2_10_x1_10;  dx1_10 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_10_x0_11;
        dxi = W[44][node_id];  dx0_00 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_00;  dx2_10 += dxi * x1_11_x0_00;
        dxi = W[45][node_id];  dx0_01 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_01;  dx2_10 += dxi * x1_11_x0_01;
        dxi = W[46][node_id];  dx0_10 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_10;  dx2_10 += dxi * x1_11_x0_10;
        dxi = W[47][node_id];  dx0_11 += dxi * x2_10_x1_11;  dx1_11 += dxi * x2_10_x0_11;  dx2_10 += dxi * x1_11_x0_11;
        dxi = W[48][node_id];  dx0_00 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_00_x0_00;
        dxi = W[49][node_id];  dx0_01 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_00_x0_01;
        dxi = W[50][node_id];  dx0_10 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_00_x0_10;
        dxi = W[51][node_id];  dx0_11 += dxi * x2_11_x1_00;  dx1_00 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_00_x0_11;
        dxi = W[52][node_id];  dx0_00 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_01_x0_00;
        dxi = W[53][node_id];  dx0_01 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_01_x0_01;
        dxi = W[54][node_id];  dx0_10 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_01_x0_10;
        dxi = W[55][node_id];  dx0_11 += dxi * x2_11_x1_01;  dx1_01 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_01_x0_11;
        dxi = W[56][node_id];  dx0_00 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_10_x0_00;
        dxi = W[57][node_id];  dx0_01 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_10_x0_01;
        dxi = W[58][node_id];  dx0_10 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_10_x0_10;
        dxi = W[59][node_id];  dx0_11 += dxi * x2_11_x1_10;  dx1_10 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_10_x0_11;
        dxi = W[60][node_id];  dx0_00 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_00;  dx2_11 += dxi * x1_11_x0_00;
        dxi = W[61][node_id];  dx0_01 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_01;  dx2_11 += dxi * x1_11_x0_01;
        dxi = W[62][node_id];  dx0_10 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_10;  dx2_11 += dxi * x1_11_x0_10;
        dxi = W[63][node_id];  dx0_11 += dxi * x2_11_x1_11;  dx1_11 += dxi * x2_11_x0_11;  dx2_11 += dxi * x1_11_x0_11;
    
        T dxn;
        T dxp;
        T dx;
        dxn  = dx0_00 * xn[1];    dxn += dx0_10 * xp[1];
        dxp  = dx0_01 * xn[1];    dxp += dx0_11 * xp[1];
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[0 * frame_stride] = dx;

        dxn  = dx0_00 * xn[0];
        dxn += dx0_01 * xp[0];
        dxp  = dx0_10 * xn[0];
        dxp += dx0_11 * xp[0];
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[1 * frame_stride] = dx;

        dxn  = dx1_00 * xn[3];     
        dxp  = dx1_01 * xn[3];     
        dxn += dx1_10 * xp[3];     
        dxp += dx1_11 * xp[3];     
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[2 * frame_stride] = dx;

        dxn  = dx1_00 * xn[2];
        dxn += dx1_01 * xp[2];
        dxp  = dx1_10 * xn[2];
        dxp += dx1_11 * xp[2];
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[3 * frame_stride] = dx;

        dxn  = dx2_00 * xn[5];     
        dxp  = dx2_01 * xn[5];     
        dxn += dx2_10 * xp[5];     
        dxp += dx2_11 * xp[5];     
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[4 * frame_stride] = dx;

        dxn  = dx2_00 * xn[4];
        dxn += dx2_01 * xp[4];
        dxp  = dx2_10 * xn[4];
        dxp += dx2_11 * xp[4];
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[5 * frame_stride] = dx;
    }
};


// LUT4
template<typename T, int MAX_NODE_UNIT>
struct StochasticLut<4, T, MAX_NODE_UNIT>
{
    static __device__ __forceinline__ T NodeForward
            (
                int             node_id,
                T           xp[],
                T   const   W[][MAX_NODE_UNIT]
            )
    {
        T   xn[4];
        for ( int i = 0; i < 4; ++i) {
            xn[i] = 1.0 - xp[i];
        }

        T x0_00 = xn[1] * xn[0];
        T x0_01 = xn[1] * xp[0];
        T x0_10 = xp[1] * xn[0];
        T x0_11 = xp[1] * xp[0];
        T x1_00 = xn[3] * xn[2];
        T x1_01 = xn[3] * xp[2];
        T x1_10 = xp[3] * xn[2];
        T x1_11 = xp[3] * xp[2];

        T y = 0;
        y += W[0 ][node_id] * x1_00 * x0_00;
        y += W[1 ][node_id] * x1_00 * x0_01;
        y += W[2 ][node_id] * x1_00 * x0_10;
        y += W[3 ][node_id] * x1_00 * x0_11;
        y += W[4 ][node_id] * x1_01 * x0_00;
        y += W[5 ][node_id] * x1_01 * x0_01;
        y += W[6 ][node_id] * x1_01 * x0_10;
        y += W[7 ][node_id] * x1_01 * x0_11;
        y += W[8 ][node_id] * x1_10 * x0_00;
        y += W[9 ][node_id] * x1_10 * x0_01;
        y += W[10][node_id] * x1_10 * x0_10;
        y += W[11][node_id] * x1_10 * x0_11;
        y += W[12][node_id] * x1_11 * x0_00;
        y += W[13][node_id] * x1_11 * x0_01;
        y += W[14][node_id] * x1_11 * x0_10;
        y += W[15][node_id] * x1_11 * x0_11;

        // clamp
        y = max(0.0, y);
        y = min(1.0, y);

        return y;
    }


    static __device__ __forceinline__ void NodeBackward
        (
            int         node_id,
            T   const   xp[],
            T           dy,
            T           *dx_ptr,
            T   const   W[][MAX_NODE_UNIT],
            T           dW[],
            int             frame_stride
        )
    {
        T   xn[4];
        for (int i = 0; i < 4; ++i) {
            xn[i] = 1.0 - xp[i];
        }

        T x0_00 = xn[1] * xn[0];
        T x0_01 = xn[1] * xp[0];
        T x0_10 = xp[1] * xn[0];
        T x0_11 = xp[1] * xp[0];
        T x1_00 = xn[3] * xn[2];
        T x1_01 = xn[3] * xp[2];
        T x1_10 = xp[3] * xn[2];
        T x1_11 = xp[3] * xp[2];

        dW[ 0] += x1_00 * x0_00 * dy;
        dW[ 1] += x1_00 * x0_01 * dy;
        dW[ 2] += x1_00 * x0_10 * dy;
        dW[ 3] += x1_00 * x0_11 * dy;
        dW[ 4] += x1_01 * x0_00 * dy;
        dW[ 5] += x1_01 * x0_01 * dy;
        dW[ 6] += x1_01 * x0_10 * dy;
        dW[ 7] += x1_01 * x0_11 * dy;
        dW[ 8] += x1_10 * x0_00 * dy;
        dW[ 9] += x1_10 * x0_01 * dy;
        dW[10] += x1_10 * x0_10 * dy;
        dW[11] += x1_10 * x0_11 * dy;
        dW[12] += x1_11 * x0_00 * dy;
        dW[13] += x1_11 * x0_01 * dy;
        dW[14] += x1_11 * x0_10 * dy;
        dW[15] += x1_11 * x0_11 * dy;
        
        T dxi;
        T dx0_00 = 0;
        T dx0_01 = 0;
        T dx0_10 = 0;
        T dx0_11 = 0;
        T dx1_00 = 0;
        T dx1_01 = 0;
        T dx1_10 = 0;
        T dx1_11 = 0;
        dxi = W[ 0][node_id];  dx0_00 += dxi * x1_00;  dx1_00 += dxi * x0_00;
        dxi = W[ 1][node_id];  dx0_01 += dxi * x1_00;  dx1_00 += dxi * x0_01;
        dxi = W[ 2][node_id];  dx0_10 += dxi * x1_00;  dx1_00 += dxi * x0_10;
        dxi = W[ 3][node_id];  dx0_11 += dxi * x1_00;  dx1_00 += dxi * x0_11;
        dxi = W[ 4][node_id];  dx0_00 += dxi * x1_01;  dx1_01 += dxi * x0_00;
        dxi = W[ 5][node_id];  dx0_01 += dxi * x1_01;  dx1_01 += dxi * x0_01;
        dxi = W[ 6][node_id];  dx0_10 += dxi * x1_01;  dx1_01 += dxi * x0_10;
        dxi = W[ 7][node_id];  dx0_11 += dxi * x1_01;  dx1_01 += dxi * x0_11;
        dxi = W[ 8][node_id];  dx0_00 += dxi * x1_10;  dx1_10 += dxi * x0_00;
        dxi = W[ 9][node_id];  dx0_01 += dxi * x1_10;  dx1_10 += dxi * x0_01;
        dxi = W[10][node_id];  dx0_10 += dxi * x1_10;  dx1_10 += dxi * x0_10;
        dxi = W[11][node_id];  dx0_11 += dxi * x1_10;  dx1_10 += dxi * x0_11;
        dxi = W[12][node_id];  dx0_00 += dxi * x1_11;  dx1_11 += dxi * x0_00;
        dxi = W[13][node_id];  dx0_01 += dxi * x1_11;  dx1_11 += dxi * x0_01;
        dxi = W[14][node_id];  dx0_10 += dxi * x1_11;  dx1_11 += dxi * x0_10;
        dxi = W[15][node_id];  dx0_11 += dxi * x1_11;  dx1_11 += dxi * x0_11;
    
        T dxn;
        T dxp;
        T dx;
        dxn  = dx0_00 * xn[1];    dxn += dx0_10 * xp[1];
        dxp  = dx0_01 * xn[1];    dxp += dx0_11 * xp[1];
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[0 * frame_stride] = dx;

        dxn  = dx0_00 * xn[0];
        dxn += dx0_01 * xp[0];
        dxp  = dx0_10 * xn[0];
        dxp += dx0_11 * xp[0];
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[1 * frame_stride] = dx;

        dxn  = dx1_00 * xn[3];     
        dxp  = dx1_01 * xn[3];     
        dxn += dx1_10 * xp[3];     
        dxp += dx1_11 * xp[3];     
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[2 * frame_stride] = dx;

        dxn  = dx1_00 * xn[2];
        dxn += dx1_01 * xp[2];
        dxp  = dx1_10 * xn[2];
        dxp += dx1_11 * xp[2];
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[3 * frame_stride] = dx;
    }
};


// LUT2
template<typename T, int MAX_NODE_UNIT>
struct StochasticLut<2, T, MAX_NODE_UNIT>
{
    static __device__ __forceinline__ T NodeForward
            (
                int             node_id,
                T           xp[],
                T   const   W[][MAX_NODE_UNIT]
            )
    {
        T   xn[2];
        for ( int i = 0; i < 2; ++i) {
            xn[i] = 1.0 - xp[i];
        }

        T x00 = xn[1] * xn[0];
        T x01 = xn[1] * xp[0];
        T x10 = xp[1] * xn[0];
        T x11 = xp[1] * xp[0];

        T y = 0;
        y += W[0][node_id] * x10 * x00;
        y += W[1][node_id] * x10 * x01;
        y += W[2][node_id] * x11 * x00;
        y += W[3][node_id] * x11 * x01;

        // clamp
        y = max(0.0, y);
        y = min(1.0, y);

        return y;
    }


    static __device__ __forceinline__ void NodeBackward
        (
            int         node_id,
            T   const   xp[],
            T           dy,
            T           *dx_ptr,
            T   const   W[][MAX_NODE_UNIT],
            T           dW[],
            int             frame_stride
        )
    {
        T   xn[2];
        for (int i = 0; i < 2; ++i) {
            xn[i] = 1.0 - xp[i];
        }

        T x00 = xn[1] * xn[0];
        T x01 = xn[1] * xp[0];
        T x10 = xp[1] * xn[0];
        T x11 = xp[1] * xp[0];

        dW[ 0] += x10 * x00 * dy;
        dW[ 1] += x10 * x01 * dy;
        dW[ 2] += x11 * x00 * dy;
        dW[ 3] += x11 * x01 * dy;
        
        T dxi;
        T dx00 = 0;
        T dx01 = 0;
        T dx10 = 0;
        T dx11 = 0;
        dxi = W[ 0][node_id];  dx00 += dxi * x10;  dx10 += dxi * x00;
        dxi = W[ 1][node_id];  dx01 += dxi * x10;  dx10 += dxi * x01;
        dxi = W[ 2][node_id];  dx00 += dxi * x11;  dx11 += dxi * x00;
        dxi = W[ 3][node_id];  dx01 += dxi * x11;  dx11 += dxi * x01;
    
        T dxn;
        T dxp;
        T dx;
        dxn  = dx00 * xn[1];
        dxn += dx10 * xp[1];
        dxp  = dx01 * xn[1];
        dxp += dx11 * xp[1];
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[0 * frame_stride] = dx;

        dxn  = dx00 * xn[0];
        dxn += dx01 * xp[0];
        dxp  = dx10 * xn[0];
        dxp += dx11 * xp[0];
        dx = (dxp - dxn) * dy;
        if ( xp[0] == 0.0 || xp[0] == 1.0 ) { dx = 0; }
        dx_ptr[1 * frame_stride] = dx;
    }
};


// end of file
