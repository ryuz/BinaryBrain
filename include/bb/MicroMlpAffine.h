// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>
#include <random>

#include "bb/Layer.h"
#include "bb/ShuffleSet.h"

namespace bb {


// Mini-MLP (SparseAffine - ReLU - SparseAffine)
template <int N = 6, int M = 16, typename T = float>
class MicroMlpAffine : public Layer
{
protected:
public:

	bool			        m_binary_mode = false;
	bool			        m_host_only   = false;
    bool                    m_host_simd   = true;

    std::mt19937_64         m_mt;

    index_t                 m_input_node_size = 0;
    index_t                 m_output_node_size = 0;
    indices_t               m_input_shape;
    indices_t               m_output_shape;

    Tensor_<std::int32_t>   m_input_index;

    std::shared_ptr<Tensor> m_W0;
    std::shared_ptr<Tensor> m_b0;
    std::shared_ptr<Tensor> m_dW0;
    std::shared_ptr<Tensor> m_db0;

    std::shared_ptr<Tensor> m_W1;
    std::shared_ptr<Tensor> m_b1;
    std::shared_ptr<Tensor> m_dW1;
    std::shared_ptr<Tensor> m_db1;

public:
    FrameBuffer             m_x;
    FrameBuffer             m_y;
    FrameBuffer             m_dx;

    FrameBuffer             m_dy;   // debug
    
#ifdef BB_WITH_CUDA
    FrameBuffer             m_dx_tmp;
#endif

protected:
	MicroMlpAffine() {
        m_W0  = std::make_shared<Tensor>();
        m_b0  = std::make_shared<Tensor>();
        m_dW0 = std::make_shared<Tensor>();
        m_db0 = std::make_shared<Tensor>();
        m_W1  = std::make_shared<Tensor>();
        m_b1  = std::make_shared<Tensor>();
        m_dW1 = std::make_shared<Tensor>();
        m_db1 = std::make_shared<Tensor>();
    }

 	void CommandProc(std::vector<std::string> args)
	{
        // バイナリモード設定
        if ( args.size() == 2 && args[0] == "bainary" )
        {
            m_binary_mode = EvalBool(args[1]);
        }

        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }

        // Host SIMDモード設定
        if (args.size() == 2 && args[0] == "host_simd")
        {
            m_host_simd = EvalBool(args[1]);
        }
	}

public:
	~MicroMlpAffine() {}


    struct create_t
    {
        indices_t       output_shape;
        std::uint64_t   seed = 1;
    };

    static std::shared_ptr<MicroMlpAffine> Create(create_t const &create)
    {
        auto self = std::shared_ptr<MicroMlpAffine>(new MicroMlpAffine);
        BB_ASSERT(!create.output_shape.empty());

        self->m_mt.seed(create.seed);

        self->m_output_shape = create.output_shape;
        self->m_output_node_size = GetShapeSize(self->m_output_shape);

        return self;
    }

    static std::shared_ptr<MicroMlpAffine> Create(indices_t const &output_shape)
    {
        create_t create;
        create.output_shape = output_shape;
        return Create(create);
    }

    static std::shared_ptr<MicroMlpAffine> Create(index_t output_node_size)
    {
        create_t create;
        create.output_shape.resize(1);
        create.output_shape[0] = output_node_size;
        return Create(create);
    }
	

	std::string GetClassName(void) const { return "MicroMlpAffine"; }


  	Tensor       &W0(void)       { return *m_W0; }
	Tensor const &W0(void) const { return *m_W0; }
  	Tensor       &b0(void)       { return *m_b0; }
	Tensor const &b0(void) const { return *m_b0; }
  	Tensor       &W1(void)       { return *m_W1; }
	Tensor const &W1(void) const { return *m_W1; }
  	Tensor       &b1(void)       { return *m_b1; }
	Tensor const &b1(void) const { return *m_b1; }
    
   	Tensor       &dW0(void)       { return *m_dW0; }
	Tensor const &dW0(void) const { return *m_dW0; }
  	Tensor       &db0(void)       { return *m_db0; }
	Tensor const &db0(void) const { return *m_db0; }
  	Tensor       &dW1(void)       { return *m_dW1; }
	Tensor const &dW1(void) const { return *m_dW1; }
  	Tensor       &db1(void)       { return *m_db1; }
	Tensor const &db1(void) const { return *m_db1; }


   	auto lock_InputIndex(void)             { return m_input_index.GetPtr(); }
	auto lock_InputIndex_const(void) const { return m_input_index.GetConstPtr(); }

	auto lock_W0(void)             { return m_W0->GetPtr<T>(); }
	auto lock_W0_const(void) const { return m_W0->GetConstPtr<T>(); }
	auto lock_b0(void)             { return m_b0->GetPtr<T>(); }
	auto lock_b0_const(void) const { return m_b0->GetConstPtr<T>(); }
	auto lock_W1(void)             { return m_W1->GetPtr<T>(); }
	auto lock_W1_const(void) const { return m_W1->GetConstPtr<T>(); }
	auto lock_b1(void)             { return m_b1->GetPtr<T>(); }
	auto lock_b1_const(void) const { return m_b1->GetConstPtr<T>(); }

	auto lock_dW0(void)             { return m_dW0->GetPtr<T>(); }
	auto lock_dW0_const(void) const { return m_dW0->GetConstPtr<T>(); }
	auto lock_db0(void)             { return m_db0->GetPtr<T>(); }
	auto lock_db0_const(void) const { return m_db0->GetConstPtr<T>(); }
	auto lock_dW1(void)             { return m_dW1->GetPtr<T>(); }
	auto lock_dW1_const(void) const { return m_dW1->GetConstPtr<T>(); }
	auto lock_db1(void)             { return m_db1->GetPtr<T>(); }
	auto lock_db1_const(void) const { return m_db1->GetConstPtr<T>(); }

    void  SetNodeInput(index_t node, index_t input_index, index_t input_node)
    {
        auto ptr = lock_InputIndex();
        ptr(node, input_index) = (std::int32_t)input_node;
    }

    index_t GetNodeInput(index_t node, index_t input_index) const
    {
        auto ptr = lock_InputIndex_const();
        return (index_t)ptr(node, input_index);
    }


   /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 形状設定
        m_input_shape = shape;
        m_input_node_size = GetShapeSize(shape);
        
        // 接続初期化
        m_input_index.Resize(m_output_node_size, N);
        {
            auto ptr = m_input_index.GetPtr();
            ShuffleSet<std::int32_t> shuffle((std::int32_t)m_input_node_size);
            for ( index_t i = 0; i < m_output_node_size; ++i ) {
                auto idx = shuffle.GetRandomSet(N);
                for ( index_t j = 0; j < N; ++j ) {
                    ptr(i, j) = idx[j];
                }
            }
        }

        // パラメータ初期化
        m_W0->Resize(DataType<T>::type, m_output_node_size, M, N);  m_W0->InitNormalDistribution(0.0, 1.0, m_mt());
        m_b0->Resize(DataType<T>::type, m_output_node_size, M);     m_b0->InitNormalDistribution(0.0, 1.0, m_mt());
        m_W1->Resize(DataType<T>::type, m_output_node_size, M);     m_W1->InitNormalDistribution(0.0, 1.0, m_mt());
        m_b1->Resize(DataType<T>::type, m_output_node_size);        m_b1->InitNormalDistribution(0.0, 1.0, m_mt());

        m_dW0->Resize(DataType<T>::type, m_output_node_size, M, N); m_dW0->FillZero();
        m_db0->Resize(DataType<T>::type, m_output_node_size, M);    m_db0->FillZero();
        m_dW1->Resize(DataType<T>::type, m_output_node_size, M);    m_dW1->FillZero();
        m_db1->Resize(DataType<T>::type, m_output_node_size);       m_db1->FillZero();

        return m_output_shape;
    }
    

   /**
     * @brief  出力のshape設定
     * @detail 出力のshape設定
     *         出力ノード数が変わらない限りshpeは自由
     * @param shape 新しいshape
     * @return なし
     */
    void SetOutputShape(indices_t const &shape)
    {
        BB_ASSERT(GetShapeSize(shape) == m_output_node_size);
        m_output_shape = shape;
    }

    
    Variables GetParameters(void)
    {
        Variables parameters;
        parameters.PushBack(m_W0);
        parameters.PushBack(m_b0);
        parameters.PushBack(m_W1);
        parameters.PushBack(m_b1);
        return parameters;
    }

    Variables GetGradients(void)
    {
        Variables gradients;
        gradients.PushBack(m_dW0);
        gradients.PushBack(m_db0);
        gradients.PushBack(m_dW1);
        gradients.PushBack(m_db1);
        return gradients;
    }
    

    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        BB_ASSERT(x.GetType() == DataType<T>::type);

        // backwardの為に保存
        m_x = x;

        // SetInputShpaeされていなければ初回に設定
        if (m_x.GetNodeSize() != m_input_node_size) {
            SetInputShape(m_x.GetShape());
        }

        // 出力を設定
        m_y.Resize(DataType<T>::type, m_x.GetFrameSize(), m_output_shape);

        // CUDA版
#ifdef BB_WITH_CUDA
        if ( N == 6 && M == 16 && DataType<T>::type == BB_TYPE_FP32
            && !m_host_only && x.IsDeviceAvailable() && m_y.IsDeviceAvailable() ) {
            ForwardCudaFP32();
            return m_y;
        }
#endif

        // AVX版
        if ( DataType<T>::type == BB_TYPE_FP32 && m_host_simd ) {
            ForwardHostSimdFP32();
            return m_y;
        }
        
        ForwardHost(x, train);
        return m_y;
    }


    FrameBuffer Backward(FrameBuffer dy)
    {
        BB_ASSERT(dy.GetType() == DataType<T>::type);

        m_dy = dy;

        m_dx.Resize(DataType<T>::type, dy.GetFrameSize(), m_input_node_size);

        m_dW0->FillZero();
        m_db0->FillZero();
        m_dW1->FillZero();
        m_db1->FillZero();

        // CUDA版
#ifdef BB_WITH_CUDA
        if ( N == 6 && M == 16 && DataType<T>::type == BB_TYPE_FP32
                && !m_host_only && m_x.IsDeviceAvailable() && m_dx.IsDeviceAvailable() && dy.IsDeviceAvailable() ) {
            BackwardCudaFP32(dy);
            return m_dx;
        }
#endif
        
        if ( DataType<T>::type == BB_TYPE_FP32 ) {
            BackwardHostSimdFP32(dy);
            return m_dx;
        }
    }
    


protected:
    // Forward
    FrameBuffer ForwardHost(FrameBuffer const &x, bool train = true)
	{
        BB_ASSERT(x.GetType() == DataType<T>::type);

        auto frame_size = m_x.GetFrameSize();
        auto x_ptr = x.GetConstPtr<T>();
        auto y_ptr = m_y.GetPtr<T>();
        auto input_index_ptr = m_input_index.GetConstPtr();
        auto W0_ptr = lock_W0_const();
        auto b0_ptr = lock_b0_const();
        auto W1_ptr = lock_W1_const();
        auto b1_ptr = lock_b1_const();

#pragma omp parallel for
		for ( index_t node = 0; node < m_output_node_size; ++node ) {
            index_t in_idx[N];
			for ( int i = 0; i < N; ++i) {
                in_idx[i] = input_index_ptr(node, i);
            }
			for (index_t frame = 0; frame < frame_size; ++frame ) {
				T   in_sig[N];
				for ( int i = 0; i < N; ++i) {
					in_sig[i] = x_ptr.Get(frame, in_idx[i]);
				}

				T	sum1 = b1_ptr(node);
				for (int i = 0; i < M; ++i) {
					// sub-layer0
					T	sum0 = b0_ptr(node, i);
					for (int j = 0; j < N; ++j) {
						sum0 += in_sig[j] * W0_ptr(node, i, j);
					}

					// ReLU
					sum0 = sum0 > (T)0 ? sum0 : (T)0;

					// sub-layer1
					sum1 += sum0 * W1_ptr(node, i);
				}

                y_ptr.Set(frame, node, sum1);
			}
        }

        return m_y;
	}
    
    void ForwardHostSimdFP32(void)
	{
		const index_t   frame_size = m_x.GetFrameStride() / sizeof(float);
		const __m256	zero = _mm256_set1_ps(0);

        auto x_ptr = m_x.GetMemoryConstPtr();
        auto y_ptr = m_y.GetMemoryPtr();
        auto input_index_ptr = m_input_index.GetConstPtr();
        auto W0_ptr = lock_W0_const();
        auto b0_ptr = lock_b0_const();
        auto W1_ptr = lock_W1_const();
        auto b1_ptr = lock_b1_const();
        
		auto in_sig_buf  = (float const *)x_ptr.GetAddr();
		auto out_sig_buf = (float       *)y_ptr.GetAddr();

#pragma omp parallel for
		for (index_t node = 0; node < m_output_node_size; ++node) {
			__m256	W0[M][N];
			__m256	b0[M];
			__m256	W1[M];
			__m256	b1;
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					W0[i][j] = _mm256_set1_ps(W0_ptr(node, i, j));
				}
				b0[i] = _mm256_set1_ps(b0_ptr(node, i));
				W1[i] = _mm256_set1_ps(W1_ptr(node, i));
			}
			b1 = _mm256_set1_ps(b1_ptr(node));

			float const *in_sig_ptr[N];
			float       *out_sig_ptr;
			for (int i = 0; i < N; ++i) {
				in_sig_ptr[i] = &in_sig_buf[input_index_ptr(node, i) * frame_size];
			}
			out_sig_ptr = &out_sig_buf[node * frame_size];

			for (index_t frame = 0; frame < frame_size; frame += 8) {
				__m256	in_sig[N];
				for (int i = 0; i < N; ++i) {
					in_sig[i] = _mm256_load_ps(&in_sig_ptr[i][frame]);
				}

				__m256	sum1 = b1;
				for (int i = 0; i < M; ++i) {
					// sub-layer0
					__m256	sum0 = b0[i];
					for (int j = 0; j < N; ++j) {
						sum0 = _mm256_fmadd_ps(in_sig[j], W0[i][j], sum0);
					}

					// ReLU
					sum0 = _mm256_max_ps(sum0, zero);

					// sub-layer1
					sum1 = _mm256_fmadd_ps(sum0, W1[i], sum1);
				}

				_mm256_store_ps(&out_sig_ptr[frame], sum1);
			}
        }
	}
    
#ifdef BB_WITH_CUDA
    void ForwardCudaFP32(void)
    {
        // 出力を設定
        auto frame_size = m_x.GetFrameSize();
        m_y.Resize(DataType<T>::type, frame_size, m_output_shape);

        // CUDA版
        auto input_index_ptr = m_input_index.GetMemoryDevConstPtr();
        auto x_ptr  = m_x.GetMemoryDevConstPtr();
        auto y_ptr  = m_y.GetMemoryDevPtr();
        auto W0_ptr = m_W0->GetMemoryDevConstPtr();
        auto b0_ptr = m_b0->GetMemoryDevConstPtr();
        auto W1_ptr = m_W1->GetMemoryDevConstPtr();
        auto b1_ptr = m_b1->GetMemoryDevConstPtr();
        bbcu_MicroMlp6x16_Forward
    		(
                (const float *)x_ptr.GetAddr(),
                (float *)y_ptr.GetAddr(),
                (int)m_input_node_size,
                (int)m_output_node_size,
                (int)m_x.GetFrameStride() / sizeof(float),
                (int *)input_index_ptr.GetAddr(),
                (float *)W0_ptr.GetAddr(),
                (float *)b0_ptr.GetAddr(),
                (float *)W1_ptr.GetAddr(),
                (float *)b1_ptr.GetAddr()
            );
    }
#endif
    

    // Backward
    void BackwardHostSimdFP32(FrameBuffer const &dy)
	{
		index_t frame_size = dy.GetFrameStride() / sizeof(float);
		index_t node_size  = m_output_node_size;

        if (dy.IsZero<float>()) {
            std::cout << "zero\n";
        }

   		m_dx.FillZero();

        auto dy_ptr = dy.GetMemoryConstPtr();
        auto dx_ptr = m_dx.GetMemoryPtr();
        auto x_ptr  = m_x.GetMemoryConstPtr();

        auto input_index_ptr = m_input_index.GetConstPtr();
        auto W0_ptr = lock_W0_const();
        auto b0_ptr = lock_b0_const();
        auto W1_ptr = lock_W1_const();
        auto b1_ptr = lock_b1_const();
        auto dW0_ptr = lock_dW0();
        auto db0_ptr = lock_db0();
        auto dW1_ptr = lock_dW1();
        auto db1_ptr = lock_db1();
        
		auto dy_buf = (float const *)dy_ptr.GetAddr();
		auto dx_buf = (float       *)dx_ptr.GetAddr();
		auto x_buf  = (float const *)x_ptr.GetAddr();

		const __m256	zero = _mm256_set1_ps(0);

		float* tmp_err_buf = (float *)aligned_memory_alloc(node_size*N*frame_size*sizeof(float), 32);
		
#pragma omp parallel for
		for (int node = 0; node < (int)node_size; ++node) {
			__m256	W0[M][N];
			__m256	b0[M];
			__m256	dW0[M][N];
			__m256	db0[M];
			__m256	W1[M];
			__m256	dW1[M];
			__m256	db1;
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					W0[i][j]  = _mm256_set1_ps(W0_ptr (node, i, j));
					dW0[i][j] = _mm256_set1_ps(dW0_ptr(node, i, j));
				}
				b0[i]  = _mm256_set1_ps(b0_ptr(node, i));
				db0[i] = _mm256_set1_ps(db0_ptr(node, i));
				W1[i]  = _mm256_set1_ps(W1_ptr(node, i));
				dW1[i] = _mm256_set1_ps(dW1_ptr(node, i));
			}
			db1 = _mm256_set1_ps(db1_ptr(node));

			float const *out_err_ptr;
			float const *in_sig_ptr[N];

			float*	tmp_err_ptr = &tmp_err_buf[node * N*frame_size];


			out_err_ptr = &dy_buf[frame_size * node];
			for (int i = 0; i < N; ++i) {
				in_sig_ptr[i] = &x_buf[frame_size * input_index_ptr(node, i)];
			}

			for (int frame = 0; frame < frame_size; frame += 8) {
				__m256	in_sig[N];
				for (int i = 0; i < N; ++i) {
					in_sig[i] = _mm256_load_ps(&in_sig_ptr[i][frame]);
				}

				// 一層目の信号を再構成
				__m256	sig0[M];
				for (int i = 0; i < M; ++i) {
					// sub-layer0
					__m256	sum0 = b0[i];
					for (int j = 0; j < N; ++j) {
						sum0 = _mm256_fmadd_ps(in_sig[j], W0[i][j], sum0);
					}

					// ReLU
					sum0 = _mm256_max_ps(sum0, zero);

					sig0[i] = sum0;
				}

				// 逆伝播
				__m256	in_err[N];
				for (int i = 0; i < N; ++i) {
					in_err[i] = zero;
				}

				__m256 out_err = _mm256_load_ps(&out_err_ptr[frame]);
				db1 = _mm256_add_ps(db1, out_err);
				for (int i = 0; i < M; ++i) {
					__m256 err0 = _mm256_mul_ps(W1[i], out_err);
					__m256 mask = _mm256_cmp_ps(sig0[i], zero, _CMP_GT_OS);
					dW1[i] = _mm256_fmadd_ps(sig0[i], out_err, dW1[i]);

					err0 = _mm256_and_ps(err0, mask);		// ReLU

					db0[i] = _mm256_add_ps(db0[i], err0);
					for (int j = 0; j < N; ++j) {
						in_err[j] = _mm256_fmadd_ps(err0, W0[i][j], in_err[j]);
						dW0[i][j] = _mm256_fmadd_ps(err0, in_sig[j], dW0[i][j]);
					}
				}

				for (int i = 0; i < N; ++i) {
					_mm256_store_ps(&tmp_err_ptr[i*frame_size + frame], in_err[i]);
				}
			}

			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					dW0_ptr(node, i, j) += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW0[i][j]));
				}
				db0_ptr(node, i) += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db0[i]));
				dW1_ptr(node, i) += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW1[i]));
			}
			db1_ptr(node) += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db1));
		}

		// 足しこみ
		for (int node = 0; node < (int)node_size; ++node) {
			float*	in_err_ptr[N];
			for (int i = 0; i < N; ++i) {
				in_err_ptr[i] = &dx_buf[frame_size * input_index_ptr(node, i)];
			}
			float*	tmp_err_ptr = &tmp_err_buf[node * N*frame_size];

#pragma omp parallel for
			for (int frame = 0; frame < frame_size; frame += 8) {
				for (int i = 0; i < N; ++i) {
					__m256 in_err = _mm256_load_ps(&in_err_ptr[i][frame]);
					__m256 tmp_err = _mm256_load_ps(&tmp_err_ptr[i*frame_size + frame]);
					in_err = _mm256_add_ps(in_err, tmp_err);
					_mm256_store_ps(&in_err_ptr[i][frame], in_err);
				}
			}

        }
		aligned_memory_free(tmp_err_buf);
	}
    

#ifdef BB_WITH_CUDA
    void BackwardCudaFP32(FrameBuffer const &dy)
    {
        // CUDA版
        auto input_index_ptr = m_input_index.GetMemoryDevConstPtr();
        auto x_ptr  = m_x.GetMemoryDevConstPtr();
        auto y_ptr  = m_y.GetMemoryDevConstPtr();
        auto dx_ptr = m_dx.GetMemoryDevPtr();
        auto dy_ptr = dy.GetMemoryDevConstPtr();
        auto W0_ptr = m_W0->GetMemoryDevConstPtr();
        auto b0_ptr = m_b0->GetMemoryDevConstPtr();
        auto W1_ptr = m_W1->GetMemoryDevConstPtr();
        auto b1_ptr = m_b1->GetMemoryDevConstPtr();
        auto dW0_ptr = m_dW0->GetMemoryDevPtr();
        auto db0_ptr = m_db0->GetMemoryDevPtr();
        auto dW1_ptr = m_dW1->GetMemoryDevPtr();
        auto db1_ptr = m_db1->GetMemoryDevPtr();

        m_dx_tmp.Resize(BB_TYPE_FP32, dy.GetFrameSize(), m_output_node_size * N);
        m_dx_tmp.FillZero();
        auto dx_tmp_ptr = m_dx_tmp.GetMemoryDevPtr();

        bbcu_MicroMlp6x16_Backward
            (
			    (float const *)x_ptr.GetAddr(),
			    (float *)dx_ptr.GetAddr(),
			    (float *)dx_tmp_ptr.GetAddr(),
			    (float *)dy_ptr.GetAddr(),
			    (int)m_input_node_size,
			    (int)m_output_node_size,
			    (int)dy.GetFrameStride() / sizeof(float),
			    (int const *)input_index_ptr.GetAddr(),
			    (float const *)W0_ptr.GetAddr(),
			    (float const *)b0_ptr.GetAddr(),
			    (float *)dW0_ptr.GetAddr(),
			    (float *)db0_ptr.GetAddr(),
			    (float const *)W1_ptr.GetAddr(),
			    (float const *)b1_ptr.GetAddr(),
			    (float *)dW1_ptr.GetAddr(),
			    (float *)db1_ptr.GetAddr()
		    );
    }
#endif
       




public:
#if 0

	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
	}

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
	}

	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
	}
#endif


    void Save(std::ostream &os) const 
    {
        SaveIndex(os, m_input_node_size);
        SaveIndex(os, m_output_node_size);
        SaveIndices(os, m_input_shape);
        SaveIndices(os, m_output_shape);
        m_input_index.Save(os);
        m_W0->Save(os);
        m_b0->Save(os);
        m_W1->Save(os);
        m_b1->Save(os);
    }

    void Load(std::istream &is)
    {
        m_input_node_size  = LoadIndex(is); 
        m_output_node_size = LoadIndex(is);
        m_input_shape      = LoadIndices(is);
        m_output_shape     = LoadIndices(is);
        m_input_index.Load(is);
        m_W0->Load(is);
        m_b0->Load(is);
        m_W1->Load(is);
        m_b1->Load(is);
        m_dW0->Resize(m_W0->GetType(), m_W0->GetShape());
        m_db0->Resize(m_b0->GetType(), m_b0->GetShape());
        m_dW1->Resize(m_W1->GetType(), m_W1->GetShape());
        m_db1->Resize(m_b1->GetType(), m_b1->GetShape());
    }

    void Save(std::string filename) const { Save(std::ofstream(filename, std::ios::binary)); }
    void Load(std::string filename)       { Load(std::ifstream(filename, std::ios::binary)); }
};


}
