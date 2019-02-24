#include <string>
#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "bb/MicroMlpAffine.h"
#include "bb/OptimizerAdam.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"


template <typename T = float>
class ModelAffine
{
public:
	std::vector<T>	W;
	T				b;
	std::vector<T>	dW;
	T				db;

	int				m_n;
	std::vector<T>	m_in_sig;

	explicit ModelAffine(int n) : m_n(n), W(n), dW(n, 0), db(0) {}

	T Forward(std::vector<T> in_sig)
	{
		m_in_sig = in_sig;

		T out_sig = b;
		for (int i = 0; i < m_n; ++i) {
			out_sig += W[i] * in_sig[i];
		}
		return out_sig;
	}

	std::vector<T> Backward(T out_err)
	{
		std::vector<T> in_err(m_n);
		db = out_err;
		for (int i = 0; i < m_n; ++i) {
			in_err[i] = out_err * W[i];
			dW[i]     = out_err * m_in_sig[i];
		}
		return in_err;
	}
};

template <typename T = float>
class ModelReLU
{
public:
	int				m_n;
	std::vector<T>	m_in_sig;

	explicit ModelReLU(int n) : m_n(n) {}

	std::vector<T> Foward(std::vector<T> in_sig)
	{
		m_in_sig = in_sig;

		std::vector<T> out_sig(m_n);
		for (int i = 0; i < m_n; ++i) {
			out_sig[i] = std::max(in_sig[i], (T)0);
		}
		return out_sig;
	}

	std::vector<T> Backward(std::vector<T> out_err)
	{
		std::vector<T> in_err(m_n);
		for (int i = 0; i < m_n; ++i) {
			in_err[i] = m_in_sig[i] > 0 ? out_err[i] : 0;
		}
		return in_err;
	}
};


#if 1

TEST(MicroMlpAffineTest, testMicroMlpAffine)
{
	auto mlp = bb::MicroMlpAffine<4, 2>::Create(1);

    bb::FrameBuffer x(BB_TYPE_FP32, 1, 6);

    auto x_ptr = x.GetPtr<float>();
    auto x_cptr = x.GetConstPtr<float>();

    float a = x_cptr.Get(0, 0);

    mlp->SetInputShape({6});
	
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 4; j++) {
			mlp->SetNodeInput(i, j, j);
		}
	}
	

    ModelAffine<float>	affine0_0(4);
	ModelAffine<float>	affine0_1(4);
	ModelReLU<float>	relu(2);
	ModelAffine<float>	affine1(2);

	std::vector<float>	in_sig_val(4);
	float				out_err_val;
	
	in_sig_val[0] = 1;
	in_sig_val[1] = 2;
	in_sig_val[2] = 3;
	in_sig_val[3] = 4;
	out_err_val = 2;


    x_ptr.Set(0, 0, in_sig_val[0]);
 	x_ptr.Set(0, 1, in_sig_val[1]);
	x_ptr.Set(0, 2, in_sig_val[2]);
	x_ptr.Set(0, 3, in_sig_val[3]);

    EXPECT_EQ(x.GetFP32(0, 0), in_sig_val[0]);
	EXPECT_EQ(x.GetFP32(0, 1), in_sig_val[1]);
	EXPECT_EQ(x.GetFP32(0, 2), in_sig_val[2]);
	EXPECT_EQ(x.GetFP32(0, 3), in_sig_val[3]);





	float W0[2][4];
	float b0[2];
	float W1[2];
	float b1;

	W0[0][0] = 2;
	W0[0][1] = 3;
	W0[0][2] = 4;
	W0[0][3] = 5;
	b0[0] = 1;

	W0[1][0] = 6;
	W0[1][1] = -7;
	W0[1][2] = -8;
	W0[1][3] = -9;
	b0[1] = 2;

	W1[0] = 5;
	W1[1] = 6;
	b1 = 1;

    {
        auto p_W0 = mlp->lock_W0();
        auto p_b0 = mlp->lock_b0();
        auto p_W1 = mlp->lock_W1();
        auto p_b1 = mlp->lock_b1();

	    for (int i = 0; i < 4; i++) {
		    p_W0(0, 0, i)  = W0[0][i];
		    affine0_0.W[i] = W0[0][i];
	    }
	    p_b0(0, 0)  = b0[0];
	    affine0_0.b = b0[0];

	    for (int i = 0; i < 4; i++) {
		    p_W0(0, 1, i)  = W0[1][i];
		    affine0_1.W[i] = W0[1][i];
	    }
	    p_b0(0, 1)  = b0[1];
	    affine0_1.b = b0[1];

	    for (int i = 0; i < 2; i++) {
		    p_W1(0, i) = W1[i];
		    affine1.W[i] = W1[i];
	    }
	    p_b1(0) = b1;
	    affine1.b = b1;
    }

    {
        auto p_W0 = mlp->lock_W0_const();
        auto p_b0 = mlp->lock_b0_const();
        auto p_W1 = mlp->lock_W1_const();
        auto p_b1 = mlp->lock_b1_const();

        for (int i = 0; i < 4; i++) {
		    EXPECT_EQ(p_W0(0, 0, i), W0[0][i]);
	    }
	    EXPECT_EQ(p_b0(0, 0), b0[0]);

	    for (int i = 0; i < 4; i++) {
		    EXPECT_EQ(p_W0(0, 1, i), W0[1][i]);
	    }
	    EXPECT_EQ(p_b0(0, 1), b0[1]);

	    for (int i = 0; i < 2; i++) {
		    EXPECT_EQ(p_W1(0, i), W1[i]);
	    }
	    EXPECT_EQ(p_b1(0), b1);
    }

	auto y = mlp->Forward(x);

	std::vector<float> hidden_sig0(2);
	hidden_sig0[0] = affine0_0.Forward(in_sig_val);
	hidden_sig0[1] = affine0_1.Forward(in_sig_val);
	auto hidden_sig1 = relu.Foward(hidden_sig0);
	float out_sig_val = affine1.Forward(hidden_sig1);


	EXPECT_EQ(out_sig_val, y.GetFP32(0, 0));

    bb::FrameBuffer dy(BB_TYPE_FP32, 1, 1);
	dy.SetFP32(0, 0, out_err_val);

	auto dx = mlp->Backward(dy);

	auto hidden_err1 = affine1.Backward(out_err_val);
	auto hidden_err0 = relu.Backward(hidden_err1);
	auto in_err_val0 = affine0_0.Backward(hidden_err0[0]);
	auto in_err_val1 = affine0_0.Backward(hidden_err0[1]);

	EXPECT_EQ(in_err_val0[0] + in_err_val1[0], dx.GetFP32(0, 0));
	EXPECT_EQ(in_err_val0[1] + in_err_val1[1], dx.GetFP32(0, 1));
	EXPECT_EQ(in_err_val0[2] + in_err_val1[2], dx.GetFP32(0, 2));
	EXPECT_EQ(in_err_val0[3] + in_err_val1[3], dx.GetFP32(0, 3));

    bb::OptimizerAdam<>::create_t create;
    bb::OptimizerAdam<> optimizer(create, mlp->GetParameters(), mlp->GetGradients());
    optimizer.Update();
}
#endif



#if 0

void DumpAffineLayer(std::ostream &os, std::string name, bb::MicroMlpAffine<6, 16, float> const &affine)
{
    static int num = 0;
    os << num << ":" << name << " W0 = " << *affine.m_W0 << std::endl;
    os << num << ":" << name << " b0 = " << *affine.m_b0 << std::endl;
    os << num << ":" << name << " W1 = " << *affine.m_W1 << std::endl;
    os << num << ":" << name << " b0 = " << *affine.m_b0 << std::endl;
    os << num << ":" << name << " dW0 = " << *affine.m_dW0 << std::endl;
    os << num << ":" << name << " db0 = " << *affine.m_db0 << std::endl;
    os << num << ":" << name << " dW1 = " << *affine.m_dW1 << std::endl;
    os << num << ":" << name << " db0 = " << *affine.m_db0 << std::endl;

    os << num << ":" << name << " x  = " << affine.m_x << std::endl;
    os << num << ":" << name << " y  = " << affine.m_y << std::endl;
    os << num << ":" << name << " dy = " << affine.m_dy << std::endl;
    os << num << ":" << name << " dx = " << affine.m_dx << std::endl;

//    num++;
}

TEST(MicroMlpAffineTest, testMicroMlpAffineFile)
{
    const int N = 6;
    const int M = 16;

    auto mlp1 = bb::MicroMlpAffine<N, M>::Create(1);
	auto mlp2 = bb::MicroMlpAffine<N, M>::Create(1);

    mlp1->Load("MicroMlpAffineTest/affine.bin");
    mlp2->Load("MicroMlpAffineTest/affine.bin");

    bb::FrameBuffer x_cpu(true);   x_cpu.Load("MicroMlpAffineTest/x.bin");
    bb::FrameBuffer x_gpu(false);  x_gpu.Load("MicroMlpAffineTest/x.bin");
//    bb::FrameBuffer y_cpu(true);   y_cpu.Load("MicroMlpAffineTest/y.bin");
//    bb::FrameBuffer y_gpu(false);  y_gpu.Load("MicroMlpAffineTest/y.bin");
    bb::FrameBuffer dy_cpu(true);
    dy_cpu.Load("MicroMlpAffineTest/dy.bin");
    bb::FrameBuffer dy_gpu(false); dy_gpu.Load("MicroMlpAffineTest/dy.bin");

    auto y_cpu = mlp1->Forward(x_cpu);
    auto y_gpu = mlp2->Forward(x_gpu);

    for (int i = 0; i < y_cpu.GetFrameSize(); i++) {
	    for (int j = 0; j < y_cpu.GetNodeSize(); j++) {
            EXPECT_NEAR(y_cpu.GetFP32(i, j), y_gpu.GetFP32(i, j), 0.0001);
        }
    }

    auto dx_cpu = mlp1->Backward(dy_cpu);
    auto dx_gpu = mlp2->Backward(dy_gpu);

    for (int i = 0; i < dx_cpu.GetFrameSize(); i++) {
	    for (int j = 0; j < dx_cpu.GetNodeSize(); j++) {
            EXPECT_NEAR(dx_cpu.GetFP32(i, j), dx_gpu.GetFP32(i, j), 0.0001);
        }
    }

    {
        std::ofstream ofs("log_cpu_t.txt");
        DumpAffineLayer(ofs, "affine0", *mlp1);
    }
    {
        std::ofstream ofs("log_gpu_t.txt");
        DumpAffineLayer(ofs, "affine0", *mlp2);
    }
}
#endif


TEST(MicroMlpAffineTest, testMicroMlpAffineCmp)
{
    const int N = 6;
    const int M = 16;
#if 0
    const int input_node_size = 360;
    const int output_node_size = 60;
    const int frame_size = 64;
#else
    const int input_node_size = 6;
    const int output_node_size = 1;
    const int frame_size = 8;
#endif

    float   W0[output_node_size][M][N];
    float   b0[output_node_size][M];
    float   W1[output_node_size][M];
    float   b1[output_node_size];

    std::mt19937_64                 mt(1);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

	auto mlp1 = bb::MicroMlpAffine<N, M>::Create(output_node_size);
	auto mlp2 = bb::MicroMlpAffine<N, M>::Create(output_node_size);
    
    bb::FrameBuffer x_cpu(BB_TYPE_FP32, frame_size, input_node_size, true);
    bb::FrameBuffer x_gpu(BB_TYPE_FP32, frame_size, input_node_size, false);

    mlp1->SetInputShape({input_node_size});
    mlp2->SetInputShape({input_node_size});
	
    
    bb::ShuffleSet<bb::index_t> ss(input_node_size, 1);
	for (int i = 0; i < output_node_size; i++) {
        auto s = ss.GetRandomSet(N);
		for (int j = 0; j < N; j++) {
            int idx = (int)s[j]; // mt() % input_node_size;
//			mlp1->SetNodeInput(i, j, idx);
//			mlp2->SetNodeInput(i, j, idx);
			mlp2->SetNodeInput(i, j, mlp1->GetNodeInput(i, j));
		}
	}

    {
        auto p1_W0 = mlp1->lock_W0();
        auto p1_b0 = mlp1->lock_b0();
        auto p1_W1 = mlp1->lock_W1();
        auto p1_b1 = mlp1->lock_b1();
        auto p2_W0 = mlp2->lock_W0();
        auto p2_b0 = mlp2->lock_b0();
        auto p2_W1 = mlp2->lock_W1();
        auto p2_b1 = mlp2->lock_b1();

        for (int i = 0; i < output_node_size; i++) {
            for (int j = 0; j < M; j++) {
                for (int k = 0; k < N; k++) {
                    int val = mt() % 100;
                    W0[i][j][k] = (float)val;
                    p1_W0(i, j, k) = (float)val;
                    p2_W0(i, j, k) = (float)val;
                }

                int val = mt() % 100;
                b0[i][j] = (float)val;
                p1_b0(i, j) = (float)val;
                p2_b0(i, j) = (float)val;
            }

            for (int j = 0; j < M; j++) {
                int val = mt() % 100;
                W1[i][j] = (float)val;
                p1_W1(i, j) = (float)val;
                p2_W1(i, j) = (float)val;
            }

            int val = mt() % 100;
            b1[i] = (float)val;
            p1_b1(i) = (float)val;
            p2_b1(i) = (float)val;
        }
    }

    for ( int loop = 0; loop < 2; ++loop ) {

	    for (int i = 0; i < frame_size; i++) {
		    for (int j = 0; j < input_node_size; j++) {
                int val = mt() % 1000;
                x_cpu.SetFP32(i, j, (float)val);
                x_gpu.SetFP32(i, j, (float)val);
            }
        }

        {
            auto p1_W0 = mlp1->lock_W0_const();
            auto p1_b0 = mlp1->lock_b0_const();
            auto p1_W1 = mlp1->lock_W1_const();
            auto p1_b1 = mlp1->lock_b1_const();
            auto p2_W0 = mlp2->lock_W0_const();
            auto p2_b0 = mlp2->lock_b0_const();
            auto p2_W1 = mlp2->lock_W1_const();
            auto p2_b1 = mlp2->lock_b1_const();

	        for (int i = 0; i < output_node_size; i++) {
    	        for (int j = 0; j < M; j++) {
        	        for (int k = 0; k < N; k++) {
	        	        EXPECT_FLOAT_EQ(W0[i][j][k], p1_W0(i, j, k));
	        	        EXPECT_FLOAT_EQ(W0[i][j][k], p2_W0(i, j, k));
                    }

        	        EXPECT_FLOAT_EQ(b0[i][j], p1_b0(i, j));
        	        EXPECT_FLOAT_EQ(b0[i][j], p2_b0(i, j));
                }

	            for (int j = 0; j < M; j++) {
        	        EXPECT_FLOAT_EQ(W1[i][j], p1_W1(i, j));
        	        EXPECT_FLOAT_EQ(W1[i][j], p2_W1(i, j));
	            }

      	        EXPECT_FLOAT_EQ(b1[i], p1_b1(i));
      	        EXPECT_FLOAT_EQ(b1[i], p2_b1(i));
            }
        }


	    auto y_cpu = mlp1->Forward(x_cpu);
	    auto y_gpu = mlp2->Forward(x_gpu);

	    for (int i = 0; i < frame_size; i++) {
            for ( int j = 0; j < output_node_size; ++j ) {
        	    EXPECT_FLOAT_EQ(y_cpu.GetFP32(i, j), y_gpu.GetFP32(i, j));
            }
        }
    

        // backward
	    for (int i = 0; i < output_node_size; i++) {
		    for (int j = 0; j < N; j++) {
			    EXPECT_EQ(mlp1->GetNodeInput(i, j), mlp2->GetNodeInput(i, j));
		    }
	    }

        bb::FrameBuffer dy_cpu(BB_TYPE_FP32, frame_size, output_node_size, true);
        bb::FrameBuffer dy_gpu(BB_TYPE_FP32, frame_size, output_node_size, false);
	
	    for (int i = 0; i < frame_size; i++) {
		    for (int j = 0; j < output_node_size; j++) {
                int val = mt() % 1000;
                dy_cpu.SetFP32(i, j, (float)val);
                dy_gpu.SetFP32(i, j, (float)val);
            }
        }

	    auto dx_cpu = mlp1->Backward(dy_cpu);
	    auto dx_gpu = mlp2->Backward(dy_gpu);

  	    for (int i = 0; i < frame_size; i++) {
		    for (int j = 0; j < input_node_size; j++) {
                EXPECT_FLOAT_EQ(dx_cpu.GetFP32(i, j), dx_gpu.GetFP32(i, j));
            }
        }

        {
            auto p1_W0 = mlp1->lock_W0_const();
            auto p1_b0 = mlp1->lock_b0_const();
            auto p1_W1 = mlp1->lock_W1_const();
            auto p1_b1 = mlp1->lock_b1_const();
            auto p2_W0 = mlp2->lock_W0_const();
            auto p2_b0 = mlp2->lock_b0_const();
            auto p2_W1 = mlp2->lock_W1_const();
            auto p2_b1 = mlp2->lock_b1_const();

	        for (int i = 0; i < output_node_size; i++) {
    	        for (int j = 0; j < M; j++) {
        	        for (int k = 0; k < N; k++) {
	        	        EXPECT_FLOAT_EQ(W0[i][j][k], p1_W0(i, j, k));
	        	        EXPECT_FLOAT_EQ(W0[i][j][k], p2_W0(i, j, k));
                    }

        	        EXPECT_FLOAT_EQ(b0[i][j], p1_b0(i, j));
        	        EXPECT_FLOAT_EQ(b0[i][j], p2_b0(i, j));
                }

	            for (int j = 0; j < M; j++) {
        	        EXPECT_FLOAT_EQ(W1[i][j], p1_W1(i, j));
        	        EXPECT_FLOAT_EQ(W1[i][j], p2_W1(i, j));
	            }

      	        EXPECT_FLOAT_EQ(b1[i], p1_b1(i));
      	        EXPECT_FLOAT_EQ(b1[i], p2_b1(i));
            }
        }

        {
            auto p1_dW0 = mlp1->lock_dW0_const();
            auto p1_db0 = mlp1->lock_db0_const();
            auto p1_dW1 = mlp1->lock_dW1_const();
            auto p1_db1 = mlp1->lock_db1_const();
            auto p2_dW0 = mlp2->lock_dW0_const();
            auto p2_db0 = mlp2->lock_db0_const();
            auto p2_dW1 = mlp2->lock_dW1_const();
            auto p2_db1 = mlp2->lock_db1_const();

	        for (int i = 0; i < output_node_size; i++) {
    	        for (int j = 0; j < M; j++) {
        	        for (int k = 0; j < N; j++) {
	        	        EXPECT_FLOAT_EQ(p1_dW0(i, j, k), p2_dW0(i, j, k));
//                      std::cout << "dW0 : " << p1_dW0(i, j, k) << ", " << p2_dW0(i, j, k) << "\n";
                    }
        	        EXPECT_FLOAT_EQ(p1_db0(i, j), p2_db0(i, j));
//                  std::cout << "db0 : " << p1_db0(i, j) << ", " << p2_db0(i, j) << "\n";
                }

	            for (int j = 0; j < M; j++) {
                    EXPECT_FLOAT_EQ(p1_dW1(i, j), p2_dW1(i, j));
//                  std::cout << "dW1 : " << p1_dW1(i, j) << ", " << p2_dW1(i, j) << "\n";
	            }

                EXPECT_FLOAT_EQ(p1_db1(i), p2_db1(i));
//              std::cout << "db1 : " << p1_db1(i) << ", " << p2_db1(i) << "\n";
            }
        }
    }
}



#if 0

TEST(NeuralNetStackedMicroAffineTest, testNeuralNetStackedMicroAffine)
{
	bb::NeuralNetStackedMicroAffine<4, 2> lut(6, 1);
	lut.SetBatchSize(1);
	testSetupLayerBuffer(lut);
	
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 4; j++) {
			lut.SetNodeInput(i, j, j);
		}
	}

	auto in_sig  = lut.GetInputSignalBuffer();
	auto out_sig = lut.GetOutputSignalBuffer();
	auto in_err  = lut.GetInputErrorBuffer();
	auto out_err = lut.GetOutputErrorBuffer();
	
	ModelAffine<float>	affine0_0(4);
	ModelAffine<float>	affine0_1(4);
	ModelReLU<float>	relu(2);
	ModelAffine<float>	affine1(2);

	std::vector<float>	in_sig_val(4);
	float				out_err_val;
	
	in_sig_val[0] = 1;
	in_sig_val[1] = 2;
	in_sig_val[2] = 3;
	in_sig_val[3] = 4;
	out_err_val = 2;
	
	in_sig.SetReal(0, 0, in_sig_val[0]);
	in_sig.SetReal(0, 1, in_sig_val[1]);
	in_sig.SetReal(0, 2, in_sig_val[2]);
	in_sig.SetReal(0, 3, in_sig_val[3]);

	float W0[2][4];
	float b0[2];
	float W1[2];
	float b1;

	W0[0][0] = 2;
	W0[0][1] = 3;
	W0[0][2] = 4;
	W0[0][3] = 5;
	b0[0] = 1;

	W0[1][0] = 6;
	W0[1][1] = -7;
	W0[1][2] = -8;
	W0[1][3] = -9;
	b0[1] = 2;

	W1[0] = 5;
	W1[1] = 6;
	b1 = 1;

	for (int i = 0; i < 4; i++) {
		lut.W0(0, 0, i)   = W0[0][i];
		affine0_0.W[i] = W0[0][i];
	}
	lut.b0(0, 0) = b0[0];
	affine0_0.b = b0[0];

	for (int i = 0; i < 4; i++) {
		lut.W0(0, 1, i)   = W0[1][i];
		affine0_1.W[i] = W0[1][i];
	}
	lut.b0(0, 1) = b0[1];
	affine0_1.b = b0[1];

	for (int i = 0; i < 2; i++) {
		lut.W1(0, i) = W1[i];
		affine1.W[i] = W1[i];
	}
	lut.b1(0) = b1;
	affine1.b = b1;


	lut.Forward();

	std::vector<float> hidden_sig0(2);
	hidden_sig0[0] = affine0_0.Forward(in_sig_val);
	hidden_sig0[1] = affine0_1.Forward(in_sig_val);
	auto hidden_sig1 = relu.Foward(hidden_sig0);
	float out_sig_val = affine1.Forward(hidden_sig1);

	EXPECT_EQ(out_sig_val, out_sig.GetReal(0, 0));
	
	out_err.SetReal(0, 0, out_err_val);
	lut.Backward();

	auto hidden_err1 = affine1.Backward(out_err_val);
	auto hidden_err0 = relu.Backward(hidden_err1);
	auto in_err_val0 = affine0_0.Backward(hidden_err0[0]);
	auto in_err_val1 = affine0_0.Backward(hidden_err0[1]);

	EXPECT_EQ(in_err_val0[0] + in_err_val1[0], in_err.GetReal(0, 0));
	EXPECT_EQ(in_err_val0[1] + in_err_val1[1], in_err.GetReal(0, 1));
	EXPECT_EQ(in_err_val0[2] + in_err_val1[2], in_err.GetReal(0, 2));
	EXPECT_EQ(in_err_val0[3] + in_err_val1[3], in_err.GetReal(0, 3));


	lut.Update();
}

#endif
