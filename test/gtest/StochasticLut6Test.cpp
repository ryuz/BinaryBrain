#include <stdio.h>
#include <iostream>
#include <random>
#include "gtest/gtest.h"

#include "bb/BatchNormalization.h"
#include "bb/OptimizerAdam.h"
#include "bb/NormalDistributionGenerator.h"




template <typename T=float>
class SimpleBatchNorm
{
public:
	SimpleBatchNorm(int n) : n(n), x(n), y(n), dx(n), dy(n), xn(n), xc(n), dxn(n), dxc(n)
	{
	}
		
	int				n;
	
	std::vector<T>	x;
	std::vector<T>	y;

	T				mean;
	T				std;
	T				var;
	T				gamma = 1;
	T				beta = 0;

	std::vector<T>	dx;
	std::vector<T>	dy;
	
	T				dmean;
	T				dstd;
	T				dvar;
	T				dgamma;
	T				dbeta;

#if 1
	std::vector<T>	xn;
	std::vector<T>	xc;
	std::vector<T>	dxn;
	std::vector<T>	dxc;

	// オリジナル
	void Forward(void)
	{
		// 平均
		mean = 0;
		for (int i = 0; i < n; ++i) {
			mean += x[i];
		}
		mean /= (T)n;

		// 平均を引く
		for (int i = 0; i < n; ++i) {
			xc[i] = x[i] - mean;
		}

		// 分散
		var = 0;
		for (int i = 0; i < n; ++i) {
			var += xc[i] * xc[i];
		}
		var /= (T)n;

		// 偏差
		std = sqrt(var);

		// 正規化
		for (int i = 0; i < n; ++i) {
			xn[i] = xc[i] / (std + (T)10e-7);
		}

		// シフト
		for (int i = 0; i < n; ++i) {
			y[i] = xn[i] * gamma + beta;
		}
	}

	// mean/var一括 (step順)
	void Forward1(void)
	{
		// 平均/分散一括
		mean = 0;
		var = 0;
		for (int i = 0; i < n; ++i) {
			mean += x[i];
			var += x[i] * x[i];
		}
		mean /= (T)n;
		var = (var / (T)n) - (mean * mean);
		std = sqrt(var);

		// 平均を引く
		for (int i = 0; i < n; ++i) {
			xc[i] = x[i] - mean;
		}

		// 正規化
		for (int i = 0; i < n; ++i) {
			xn[i] = xc[i] / (std + (T)10e-7);
		}
			// シフト
		for (int i = 0; i < n; ++i) {
			y[i] = xn[i] * gamma + beta;
		}
	}

	// mean/var一括 (ループ最適化)
	void Forward2(void)
	{
		// 平均/分散一括
		mean = 0;
		var = 0;
		for (int i = 0; i < n; ++i) {
			mean += x[i];
			var  += x[i] * x[i];
		}
		mean /= (T)n;
		var = (var / (T)n) - (mean * mean);
		std = sqrt(var);

		for (int i = 0; i < n; ++i) {
			// 平均を引く
			xc[i] = x[i] - mean;

			// 正規化
			xn[i] = xc[i] / (std + (T)10e-7);

			// シフト
			y[i] = xn[i] * gamma + beta;
		}
	}
#endif

	// mean/var一括 (中間変数未使用)
	void Forward3(void)
	{
		// 平均/分散一括
		mean = 0;
		var = 0;
		for (int i = 0; i < n; ++i) {
			mean += x[i];
			var += x[i] * x[i];
		}
		mean /= (T)n;
		var = (var / (T)n) - (mean * mean);
		std = sqrt(var);

		for (int i = 0; i < n; ++i) {
			// 平均を引く
			T _xc = x[i] - mean;

			// 正規化
			T _xn = _xc / (std + (T)10e-7);

			// シフト
			y[i] = _xn * gamma + beta;
		}
	}

#if 1
	// オリジナル
	void Backward(void)
	{
		dbeta = 0;
		for (int i = 0; i < n; ++i) {
			dbeta += dy[i];
		}

		dgamma = 0;
		for (int i = 0; i < n; ++i) {
			dgamma += xn[i] * dy[i];
		}

		for (int i = 0; i < n; ++i) {
			dxn[i] = dy[i] * gamma;
		}

		for (int i = 0; i < n; ++i) {
			dxc[i] = dxn[i] / std;
		}

		dstd = 0;
		for (int i = 0; i < n; ++i) {
			dstd += -(dxn[i] * xc[i]) / (std * std);
		}

		dvar = (T)0.5 * dstd / std;

		for (int i = 0; i < n; ++i) {
			dxc[i] = dxc[i] + (((T)2.0 / (T)n) * dvar * xc[i]);
		}

		dmean = 0;
		for (int i = 0; i < n; ++i) {
			dmean += dxc[i];
		}

		for (int i = 0; i < n; ++i) {
			dx[i] = dxc[i] - (dmean / (T)n);
		}
	}


	// mean/var一括 (step順)
	void Backward1(void)
	{
		dbeta = 0;
		for (int i = 0; i < n; ++i) {
			dbeta += dy[i];
		}

		dgamma = 0;
		for (int i = 0; i < n; ++i) {
			dgamma += xn[i] * dy[i];
		}

		for (int i = 0; i < n; ++i) {
			dxn[i] = dy[i] * gamma;
		}

		for (int i = 0; i < n; ++i) {
			dxc[i] = dxn[i] / std;
		}

		dstd = 0;
		for (int i = 0; i < n; ++i) {
			dstd += -(dxn[i] * xc[i]) / (std * std);
		}

		dvar = (T)0.5 * dstd / std;

		T dmeanx = 0;
		for (int i = 0; i < n; ++i) {
			dmeanx += (-dxc[i]);
		}
		dmean = dmeanx + (2.0f * mean * -dvar);
		dmean /= (T)n;

		for (int i = 0; i < n; ++i) {
			dx[i] = dxc[i] + dmean + (((T)2 / (T)n) * x[i] * dvar);
		}
	}


	// mean/var一括 (ループ最適化)
	void Backward2(void)
	{
		dbeta = 0;
		dgamma = 0;
		dstd = 0;
		T dmeanx = 0;
		for (int i = 0; i < n; ++i) {
			dbeta  += dy[i];
			dgamma += xn[i] * dy[i];
			dxn[i] = dy[i] * gamma;
			dxc[i] = dxn[i] / std;
			dstd += -(dxn[i] * xc[i]) / (std * std);
			dmeanx += (-dxc[i]);
		}

		dvar = (T)0.5 * dstd / std;
		dmean = dmeanx + (2.0f * mean * -dvar);
		dmean /= (T)n;

		for (int i = 0; i < n; ++i) {
			dx[i] = dxc[i] + dmean + (((T)2 / (T)n) * x[i] * dvar);
		}
	}
#endif

	// 中間変数未使用 (step順)
	void Backward3(void)
	{
		dbeta = 0;
		dgamma = 0;
		dstd = 0;
		T dmeanx = 0;
		for (int i = 0; i < n; ++i) {
			T _xc = (x[i] - mean);
			T _xn = _xc / std;

			dbeta += dy[i];
			dgamma += _xn * dy[i];
			T _dxn = dy[i] * gamma;
			T _dxc = _dxn / std;
			dstd += -(_dxn * _xc) / var; // (std * std);
			dmeanx += (-_dxc);
		}

		dvar = dstd / std;
		dmean = dmeanx + (mean * -dvar);
		dmean /= (T)n;

		for (int i = 0; i < n; ++i) {
			T _dxn = dy[i] * gamma;
			T _dxc = _dxn / std;

			dx[i] = _dxc + dmean + (((T)1 / (T)n) * x[i] * dvar);
		}
	}



};




#if 1
TEST(BatchNormalizationTest, testBatchNormalization)
{
    bb::BatchNormalization<float>::create_t create;
    auto batch_norm = bb::BatchNormalization<float>::Create(create);

    bb::FrameBuffer x(BB_TYPE_FP32, 8, 2);
    
    batch_norm->SetInputShape({2});


    SimpleBatchNorm<double> exp_norm0(8);
	SimpleBatchNorm<double> exp_norm1(8);

	x.SetFP32(0, 0, 1);
	x.SetFP32(1, 0, 2);
	x.SetFP32(2, 0, 3);
	x.SetFP32(3, 0, 4);
	x.SetFP32(4, 0, 5);
	x.SetFP32(5, 0, 6);
	x.SetFP32(6, 0, 7);
	x.SetFP32(7, 0, 8);
	x.SetFP32(0, 1, 10);
	x.SetFP32(1, 1, 30);
	x.SetFP32(2, 1, 20);
	x.SetFP32(3, 1, 15);
	x.SetFP32(4, 1, 11);
	x.SetFP32(5, 1, 34);
	x.SetFP32(6, 1, 27);
	x.SetFP32(7, 1, 16);

	for (int i = 0; i < 8; i++) {
		exp_norm0.x[i] = x.GetFP32(i, 0);
		exp_norm1.x[i] = x.GetFP32(i, 1);
	}
    
	auto y = batch_norm->Forward(x, true);
//	auto y = batch_norm->Forward(x, false);

	exp_norm0.Forward();
	exp_norm1.Forward();


	/*
	[-1.52752510, -1.23359570],
	[-1.09108940, +1.14442010],
	[-0.65465360, -0.04458780],
	[-0.21821786, -0.63909180],
	[+0.21821786, -1.11469500],
	[+0.65465360, +1.62002340],
	[+1.09108940, +0.78771776],
	[+1.52752510, -0.52019095]
	*/

#if 0
	std::cout << out_sig.GetReal(0, 0) << std::endl;
	std::cout << out_sig.GetReal(1, 0) << std::endl;
	std::cout << out_sig.GetReal(2, 0) << std::endl;
	std::cout << out_sig.GetReal(3, 0) << std::endl;
	std::cout << out_sig.GetReal(4, 0) << std::endl;
	std::cout << out_sig.GetReal(5, 0) << std::endl;
	std::cout << out_sig.GetReal(6, 0) << std::endl;
	std::cout << out_sig.GetReal(7, 0) << std::endl;
#endif

	for (int i = 0; i < 8; i++) {
		EXPECT_NEAR(exp_norm0.y[i], y.GetFP32(i, 0), 0.000001);
		EXPECT_NEAR(exp_norm1.y[i], y.GetFP32(i, 1), 0.000001);
	}

	// _mm256_rsqrt_ps を使っているので精度は悪い
	EXPECT_NEAR(-1.52752510, y.GetFP32(0, 0), 0.000001);
	EXPECT_NEAR(-1.09108940, y.GetFP32(1, 0), 0.000001);
	EXPECT_NEAR(-0.65465360, y.GetFP32(2, 0), 0.000001);
	EXPECT_NEAR(-0.21821786, y.GetFP32(3, 0), 0.000001);
	EXPECT_NEAR(+0.21821786, y.GetFP32(4, 0), 0.000001);
	EXPECT_NEAR(+0.65465360, y.GetFP32(5, 0), 0.000001);
	EXPECT_NEAR(+1.09108940, y.GetFP32(6, 0), 0.000001);
	EXPECT_NEAR(+1.52752510, y.GetFP32(7, 0), 0.000001);

	EXPECT_NEAR(-1.23359570, y.GetFP32(0, 1), 0.000001);
	EXPECT_NEAR(+1.14442010, y.GetFP32(1, 1), 0.000001);
	EXPECT_NEAR(-0.04458780, y.GetFP32(2, 1), 0.000001);
	EXPECT_NEAR(-0.63909180, y.GetFP32(3, 1), 0.000001);
	EXPECT_NEAR(-1.11469500, y.GetFP32(4, 1), 0.000001);
	EXPECT_NEAR(+1.62002340, y.GetFP32(5, 1), 0.000001);
	EXPECT_NEAR(+0.78771776, y.GetFP32(6, 1), 0.000001);
	EXPECT_NEAR(-0.52019095, y.GetFP32(7, 1), 0.000001);

    bb::FrameBuffer dy(BB_TYPE_FP32, 8, 2);
	
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 8; j++) {
			dy.SetFP32(j, i, y.GetFP32(j, i));
		}
	}

	dy.SetFP32(0, 0, 8);
	dy.SetFP32(1, 0, 6);
	dy.SetFP32(2, 0, 3);
	dy.SetFP32(3, 0, 4);
	dy.SetFP32(4, 0, 5);
	dy.SetFP32(5, 0, 4);
	dy.SetFP32(6, 0, 6);
	dy.SetFP32(7, 0, 1);
	dy.SetFP32(0, 1, 20);
	dy.SetFP32(1, 1, 70);
	dy.SetFP32(2, 1, 40);
	dy.SetFP32(3, 1, 15);
	dy.SetFP32(4, 1, 31);
	dy.SetFP32(5, 1, 54);
	dy.SetFP32(6, 1, 37);
	dy.SetFP32(7, 1, 26);

	for (int i = 0; i < 8; i++) {
		exp_norm0.dy[i] = dy.GetFP32(i, 0);
		exp_norm1.dy[i] = dy.GetFP32(i, 1);
	}
        
    auto dx = batch_norm->Backward(dy);

	exp_norm0.Backward();
	exp_norm1.Backward();


#if 0
	std::cout << in_err.GetReal(0, 0) << std::endl;
	std::cout << in_err.GetReal(1, 0) << std::endl;
	std::cout << in_err.GetReal(2, 0) << std::endl;
	std::cout << in_err.GetReal(3, 0) << std::endl;
	std::cout << in_err.GetReal(4, 0) << std::endl;
	std::cout << in_err.GetReal(5, 0) << std::endl;
	std::cout << in_err.GetReal(6, 0) << std::endl;
	std::cout << in_err.GetReal(7, 0) << std::endl;
#endif

	/*
		[+0.65465380, +0.08798742],
		[+0.01558709, +2.05285700],
		[-1.05991530, +0.47591877],
		[-0.38967478, -1.50155930],
		[+0.28056574, +1.19688750],
		[+0.07793474, -0.64558935],
		[+1.18461110, -1.27384350],
		[-0.76376295, -0.39265870]]
	*/

	EXPECT_NEAR(+0.65465380, dx.GetFP32(0, 0), 0.00001);
	EXPECT_NEAR(+0.01558709, dx.GetFP32(1, 0), 0.00001);
	EXPECT_NEAR(-1.05991530, dx.GetFP32(2, 0), 0.00001);
	EXPECT_NEAR(-0.38967478, dx.GetFP32(3, 0), 0.00001);
	EXPECT_NEAR(+0.28056574, dx.GetFP32(4, 0), 0.00001);
	EXPECT_NEAR(+0.07793474, dx.GetFP32(5, 0), 0.00001);
	EXPECT_NEAR(+1.18461110, dx.GetFP32(6, 0), 0.00001);
	EXPECT_NEAR(-0.76376295, dx.GetFP32(7, 0), 0.00001);

	EXPECT_NEAR(+0.08798742, dx.GetFP32(0, 1), 0.00001);
	EXPECT_NEAR(+2.05285700, dx.GetFP32(1, 1), 0.00001);
	EXPECT_NEAR(+0.47591877, dx.GetFP32(2, 1), 0.00001);
	EXPECT_NEAR(-1.50155930, dx.GetFP32(3, 1), 0.00001);
	EXPECT_NEAR(+1.19688750, dx.GetFP32(4, 1), 0.00001);
	EXPECT_NEAR(-0.64558935, dx.GetFP32(5, 1), 0.00001);
	EXPECT_NEAR(-1.27384350, dx.GetFP32(6, 1), 0.00001);
	EXPECT_NEAR(-0.39265870, dx.GetFP32(7, 1), 0.00001);

	for (int i = 0; i < 8; i++) {
//		std::cout << exp_norm0.dx[i] << std::endl;
		EXPECT_NEAR(exp_norm0.dx[i], dx.GetFP32(i, 0), 0.001);
		EXPECT_NEAR(exp_norm1.dx[i], dx.GetFP32(i, 1), 0.001);
	}
}
#endif


#if 1
TEST(BatchNormalizationTest, testBatchNormalization_test02)
{
   int const node_size  = 3;
   int const frame_size = 8*123;

    bb::BatchNormalization<float>::create_t create;
    auto bn = bb::BatchNormalization<float>::Create(create);

    bb::FrameBuffer x_buf(BB_TYPE_FP32, frame_size, node_size);
    bn->SetInputShape(x_buf.GetShape());

    auto valgen = bb::NormalDistributionGenerator<float>::Create(1.2f, 3.3f, 1);
    for ( int frame = 0; frame < frame_size; ++frame) {
        for ( int node = 0; node < node_size; ++node ) {
            x_buf.SetFP32(frame, node, valgen->GetValue());
        }
    }


    SimpleBatchNorm<double> exp_norm0(frame_size);
	SimpleBatchNorm<double> exp_norm1(frame_size);
	SimpleBatchNorm<double> exp_norm2(frame_size);

#if 1
    {
        auto gamma_ptr = bn->lock_gamma();
        auto beta_ptr = bn->lock_beta();
        for ( int node = 0; node < node_size; ++node ) {
            gamma_ptr[node] = valgen->GetValue();
            beta_ptr[node] = valgen->GetValue();
        }

        exp_norm0.gamma = gamma_ptr[0];
        exp_norm1.gamma = gamma_ptr[1];
        exp_norm2.gamma = gamma_ptr[2];
        exp_norm0.beta  = beta_ptr[0];
        exp_norm1.beta  = beta_ptr[1];
        exp_norm2.beta  = beta_ptr[2];
    }
#endif

    for ( int frame = 0; frame < frame_size; ++frame) {
		exp_norm0.x[frame] = x_buf.GetFP32(frame, 0);
		exp_norm1.x[frame] = x_buf.GetFP32(frame, 1);
		exp_norm2.x[frame] = x_buf.GetFP32(frame, 2);
	}
    
	auto y_buf = bn->Forward(x_buf, true);

	exp_norm0.Forward();
	exp_norm1.Forward();
	exp_norm2.Forward();

    for ( int frame = 0; frame < frame_size; ++frame) {
		EXPECT_NEAR(exp_norm0.y[frame], y_buf.GetFP32(frame, 0), 0.0001f);
		EXPECT_NEAR(exp_norm1.y[frame], y_buf.GetFP32(frame, 1), 0.0001f);
		EXPECT_NEAR(exp_norm2.y[frame], y_buf.GetFP32(frame, 2), 0.0001f);
	}

    {
        auto tmp_mean = bn->lock_tmp_mean_const();
        EXPECT_NEAR(exp_norm0.mean, tmp_mean[0], 0.00001f);
        EXPECT_NEAR(exp_norm1.mean, tmp_mean[1], 0.00001f);
        EXPECT_NEAR(exp_norm2.mean, tmp_mean[2], 0.00001f);

        auto tmp_rstd = bn->lock_tmp_rstd_const();
        EXPECT_NEAR(1.0f/sqrt(exp_norm0.var), tmp_rstd[0], 0.00001f);
        EXPECT_NEAR(1.0f/sqrt(exp_norm1.var), tmp_rstd[1], 0.00001f);
        EXPECT_NEAR(1.0f/sqrt(exp_norm2.var), tmp_rstd[2], 0.00001f);
    }

    // backward
    bb::FrameBuffer dy_buf(BB_TYPE_FP32,  frame_size, node_size);
    for ( int frame = 0; frame < frame_size; ++frame) {
        for ( int node = 0; node < node_size; ++node ) {
            dy_buf.SetFP32(frame, node, valgen->GetValue());
        }
    }

    for ( int frame = 0; frame < frame_size; ++frame) {
		exp_norm0.dy[frame] = dy_buf.GetFP32(frame, 0);
		exp_norm1.dy[frame] = dy_buf.GetFP32(frame, 1);
		exp_norm2.dy[frame] = dy_buf.GetFP32(frame, 2);
	}
        
    auto dx_buf = bn->Backward(dy_buf);

	exp_norm0.Backward();
	exp_norm1.Backward();
	exp_norm2.Backward();

    for ( int frame = 0; frame < frame_size; ++frame) {
		EXPECT_NEAR(exp_norm0.dx[frame], dx_buf.GetFP32(frame, 0), 0.001);
		EXPECT_NEAR(exp_norm1.dx[frame], dx_buf.GetFP32(frame, 1), 0.001);
		EXPECT_NEAR(exp_norm2.dx[frame], dx_buf.GetFP32(frame, 2), 0.001);
	}

    auto dgamma_ptr = bn->lock_dgamma_const();
    auto dbeta_ptr  = bn->lock_dbeta_const();
	EXPECT_NEAR(exp_norm0.dgamma, dgamma_ptr[0], 0.001);
	EXPECT_NEAR(exp_norm1.dgamma, dgamma_ptr[1], 0.001);
	EXPECT_NEAR(exp_norm2.dgamma, dgamma_ptr[2], 0.001);

	EXPECT_NEAR(exp_norm0.dbeta, dbeta_ptr[0], 0.001);
	EXPECT_NEAR(exp_norm1.dbeta, dbeta_ptr[1], 0.001);
	EXPECT_NEAR(exp_norm2.dbeta, dbeta_ptr[2], 0.001);
}

#endif


#ifdef BB_WITH_CUDA

TEST(BatchNormalizationTest, testBatchNormalization_cmp)
{
    int const node_size  = 13;
    int const frame_size = 8*123;

    bb::BatchNormalization<float>::create_t create;
    auto bn_cpu = bb::BatchNormalization<float>::Create(create);
    auto bn_gpu = bb::BatchNormalization<float>::Create(create);

    auto opt_cpu = bb::OptimizerAdam<float>::Create();
    auto opt_gpu = bb::OptimizerAdam<float>::Create();

    bn_cpu->SendCommand("host_only true");

    bb::FrameBuffer x_cpu(BB_TYPE_FP32, frame_size, node_size, true);
    bb::FrameBuffer x_gpu(BB_TYPE_FP32, frame_size, node_size);
    
    bn_cpu->SetInputShape(x_cpu.GetShape());
    bn_gpu->SetInputShape(x_gpu.GetShape());

    opt_cpu->SetVariables(bn_cpu->GetParameters(), bn_cpu->GetGradients());
    opt_gpu->SetVariables(bn_gpu->GetParameters(), bn_gpu->GetGradients());


    auto valgen = bb::NormalDistributionGenerator<float>::Create(1.2f, 3.3f, 1);

    for ( int loop = 0; loop < 4; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                x_cpu.SetFP32(frame, node, valgen->GetValue());
                x_gpu.SetFP32(frame, node, x_cpu.GetFP32(frame, node));
            }
        }

        auto y_cpu = bn_cpu->Forward(x_cpu);
        auto y_gpu = bn_gpu->Forward(x_gpu);

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = x_cpu.GetFP32(frame, node);
                auto val_gpu = x_gpu.GetFP32(frame, node);
                EXPECT_FLOAT_EQ(val_cpu, val_gpu);
            }
        }

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = y_cpu.GetFP32(frame, node);
                auto val_gpu = y_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
            }
        }


        // backward
        bb::FrameBuffer dy_cpu(BB_TYPE_FP32, frame_size, node_size, true);
        bb::FrameBuffer dy_gpu(BB_TYPE_FP32, frame_size, node_size);
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                dy_cpu.SetFP32(frame, node, valgen->GetValue());
                dy_gpu.SetFP32(frame, node, dy_cpu.GetFP32(frame, node));
            }
        }

        auto dx_cpu = bn_cpu->Backward(dy_cpu);
        auto dx_gpu = bn_gpu->Backward(dy_gpu);

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = dx_cpu.GetFP32(frame, node);
                auto val_gpu = dx_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
            }
        }

        {
            auto dgamma_cpu = bn_cpu->lock_dgamma_const();
            auto dgamma_gpu = bn_gpu->lock_dgamma_const();
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = dgamma_cpu[node];
                auto val_gpu = dgamma_gpu[node];
                EXPECT_NEAR(val_cpu, val_gpu, 0.001f);
            }

            auto dbeta_cpu = bn_cpu->lock_dbeta_const();
            auto dbeta_gpu = bn_gpu->lock_dbeta_const();
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = dbeta_cpu[node];
                auto val_gpu = dbeta_gpu[node];
                EXPECT_NEAR(val_cpu, val_gpu, 0.001f);
            }

            auto mean_cpu = bn_cpu->lock_mean_const();
            auto mean_gpu = bn_gpu->lock_mean_const();
            auto var_cpu  = bn_cpu->lock_var_const();
            auto var_gpu  = bn_gpu->lock_var_const();
            auto tmp_mean_cpu = bn_cpu->lock_tmp_mean_const();
            auto tmp_mean_gpu = bn_gpu->lock_tmp_mean_const();
            auto tmp_rstd_cpu = bn_cpu->lock_tmp_rstd_const();
            auto tmp_rstd_gpu = bn_gpu->lock_tmp_rstd_const();
            for ( int node = 0; node < node_size; ++node ) {
                EXPECT_NEAR(mean_cpu[node], mean_gpu[node], 0.001f);
                EXPECT_NEAR(var_cpu[node], var_gpu[node], 0.001f);
                EXPECT_NEAR(tmp_mean_cpu[node], tmp_mean_gpu[node], 0.001f);
                EXPECT_NEAR(tmp_rstd_cpu[node], tmp_rstd_gpu[node], 0.001f);
            }
        }

        opt_cpu->Update();
        opt_gpu->Update();

        {
            auto gamma_cpu = bn_cpu->lock_gamma_const();
            auto gamma_gpu = bn_gpu->lock_gamma_const();
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = gamma_cpu[node];
                auto val_gpu = gamma_gpu[node];
                EXPECT_NEAR(val_cpu, val_gpu, 0.001f);
            }

            auto beta_cpu = bn_cpu->lock_beta_const();
            auto beta_gpu = bn_gpu->lock_beta_const();
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = beta_cpu[node];
                auto val_gpu = beta_gpu[node];
                EXPECT_NEAR(val_cpu, val_gpu, 0.001f);
            }
        }
    }


    for ( int loop = 0; loop < 4; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                x_cpu.SetFP32(frame, node, valgen->GetValue());
                x_gpu.SetFP32(frame, node, x_cpu.GetFP32(frame, node));
            }
        }

        auto y_cpu = bn_cpu->Forward(x_cpu, false);
        auto y_gpu = bn_gpu->Forward(x_gpu, false);

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = x_cpu.GetFP32(frame, node);
                auto val_gpu = x_gpu.GetFP32(frame, node);
                EXPECT_FLOAT_EQ(val_cpu, val_gpu);
            }
        }

        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < node_size; ++node ) {
                auto val_cpu = y_cpu.GetFP32(frame, node);
                auto val_gpu = y_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
            }
        }
    }
}

#endif


