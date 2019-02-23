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
        // backwardの為に保存
        m_x = x;

        if (m_x.GetNodeSize() != m_input_node_size) {
            SetInputShape(m_x.GetShape());
        }

        // 出力を設定
        auto frame_size = m_x.GetFrameSize();
        m_y.Resize(DataType<T>::type, frame_size, m_output_shape);

#ifdef BB_WITH_CUDA
        if ( N == 6 && M == 16 && DataType<T>::type == BB_TYPE_FP32
            && x.IsDeviceAvailable() && m_y.IsDeviceAvailable() ) {
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
            return m_y;
        }
#endif

        if ( DataType<T>::type == BB_TYPE_FP32 ) {
            return Forward_AVX_FP32(x, train);
        }
        
        return m_y;
    }


    FrameBuffer Backward(FrameBuffer dy)
    {
        m_dy = dy;

        m_dx.Resize(DataType<T>::type, dy.GetFrameSize(), m_input_node_size);

        auto frame_size = dy.GetFrameSize();

        m_dW0->FillZero();
        m_db0->FillZero();
        m_dW1->FillZero();
        m_db1->FillZero();

#ifdef BB_WITH_CUDA
        if ( N == 6 && M == 16 && DataType<T>::type == BB_TYPE_FP32
            && m_x.IsDeviceAvailable() && m_dx.IsDeviceAvailable() && dy.IsDeviceAvailable() ) {
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

            float* dev_in_err_tmp;
            BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_in_err_tmp,  m_output_node_size * N * frame_size * sizeof(float)));
            BB_CUDA_SAFE_CALL(cudaMemset(dev_in_err_tmp, 0, m_output_node_size * N * frame_size * sizeof(float)));

            bbcu_MicroMlp6x16_Backward
                (
			        (float const *)x_ptr.GetAddr(),
			        (float *)dx_ptr.GetAddr(),
			        dev_in_err_tmp,
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

#if 0
            float tmp[8][1][6];
            BB_CUDA_SAFE_CALL(cudaMemcpy(tmp, dev_in_err_tmp, sizeof(tmp), cudaMemcpyDeviceToHost));
            float dst[8][1][6];
            BB_CUDA_SAFE_CALL(cudaMemcpy(dst, dx_ptr.GetAddr(), sizeof(tmp), cudaMemcpyDeviceToHost));
#endif

            BB_CUDA_SAFE_CALL(cudaFree(dev_in_err_tmp));

            return m_dx;
        }
#endif

        if ( DataType<T>::type == BB_TYPE_FP32 ) {
            return Backward_AVX_FP32(dy);
        }
    }
    
           
    FrameBuffer Forward_AVX_FP32(FrameBuffer const &x, bool train = true)
	{
		const index_t   frame_size = x.GetFrameStride() / sizeof(float);
		const __m256	zero = _mm256_set1_ps(0);

        auto x_ptr = x.GetMemoryConstPtr();
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

        return m_y;
	}


    FrameBuffer Backward_AVX_FP32(FrameBuffer const &dy)
	{
		index_t frame_size = dy.GetFrameStride() / sizeof(float);
		index_t node_size  = m_output_node_size;

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
		m_dx.FillZero();
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

        return m_dx;
	}
    

#if 0
	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		auto& nd = m_node[node];

		// affine0
		std::vector<T> value0(M);
		for (INDEX i = 0; i < M; ++i) {
			value0[i] = nd.b0[i];
			for (INDEX j = 0; j < N; ++j) {
				value0[i] += input_value[j] * nd.W0[i*N + j];
			}
		}

		// ReLU
		for (INDEX i = 0; i < M; ++i) {
			value0[i] = std::max(value0[i], (T)0);;
		}

		// affine1
		std::vector<T> value1(1);
		value1[0] = nd.b1;
		for (INDEX i = 0; i < M; ++i) {
			value1[0] += value0[i] * nd.W1[i];
		}
		
		return value1;
	}


	void Resize(INDEX input_node_size, INDEX output_node_size)
	{
		super::Resize(input_node_size, output_node_size);

		m_node.resize(this->m_output_node_size);
	}

	void InitializeCoeff(std::uint64_t seed)
	{
		super::InitializeCoeff(seed);

		std::mt19937_64 mt(seed);
		std::normal_distribution<T> distribution((T)0.0, (T)1.0);
		
		for (auto& node : m_node) {
			for (auto& W0 : node.W0) { W0 = distribution(mt); }
			for (auto& b0 : node.b0) { b0 = distribution(mt); }
			for (auto& W1 : node.W1) { W1 = distribution(mt); }
			node.b1 = distribution(mt);

			std::fill(node.dW0.begin(), node.dW0.end(), (T)0);
			std::fill(node.db0.begin(), node.db0.end(), (T)0);
			std::fill(node.dW1.begin(), node.dW1.end(), (T)0);
			node.db1 = 0;
		}
	}

	void SetOptimizer(const NeuralNetOptimizer<T>* optimizer)
	{
		for (auto& node : m_node) {
			node.optimizer_W0.reset(optimizer->Create(M*N));
			node.optimizer_b0.reset(optimizer->Create(M));
			node.optimizer_W1.reset(optimizer->Create(M));
			node.optimizer_b1.reset(optimizer->Create(1));
		}
	}

	void  SetBinaryMode(bool enable)
	{
		m_binary_mode = enable;
	}

	int   GetNodeInputSize(INDEX node) const { return N; }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) {
		BB_ASSERT(node >= 0 && node < GetOutputNodeSize());
		BB_ASSERT(input_index >= 0 && input_index < GetNodeInputSize(node));
		BB_ASSERT(input_node >= 0 && input_node < GetInputNodeSize());
		m_node[node].input[input_index] = input_node;
	}
	INDEX GetNodeInput(INDEX node, int input_index) const { return m_node[node].input[input_index]; }

	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }

	int   GetInputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }



public:
	void Forward(bool train=true)
	{
		const int		node_size = (int)this->GetOutputNodeSize();
		const int		frame_size = (int)((m_frame_size + 7) / 8 * 8);
		const __m256	zero = _mm256_set1_ps(0);

		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();

		if (typeid(T) == typeid(float)) {

#pragma omp parallel for
			for (int node = 0; node < (int)node_size; ++node) {
				auto& nd = m_node[node];

				__m256	W0[M][N];
				__m256	b0[M];
				__m256	W1[M];
				__m256	b1;
				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						W0[i][j] = _mm256_set1_ps(nd.W0[i*N + j]);
					}
					b0[i] = _mm256_set1_ps(nd.b0[i]);
					W1[i] = _mm256_set1_ps(nd.W1[i]);
				}
				b1 = _mm256_set1_ps(nd.b1);

				float*	in_sig_ptr[N];
				float*	out_sig_ptr;
				for (int i = 0; i < N; ++i) {
					in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(m_node[node].input[i]);
				}
				out_sig_ptr = (float*)out_sig_buf.GetPtr(node);

				for (int frame = 0; frame < frame_size; frame += 8) {
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
	}

#if 0
	void Backward(void)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		auto in_err_buf = this->GetInputErrorBuffer();
		auto out_err_buf = this->GetOutputErrorBuffer();

		auto node_size = this->GetOutputNodeSize();
		const __m256	zero = _mm256_set1_ps(0);

		in_err_buf.Clear();

		if (typeid(T) == typeid(float)) {
			INDEX frame_size = (m_frame_size + 7) / 8 * 8;

			for (int node = 0; node < (int)node_size; ++node) {
				auto& nd = m_node[node];

				__m256	W0[M][N];
				__m256	b0[M];
				__m256	dW0[M][N];
				__m256	db0[M];
				__m256	W1[M];
				__m256	dW1[M];
				__m256	db1;
				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						W0[i][j] = _mm256_set1_ps(nd.W0[i*N + j]);
						dW0[i][j] = _mm256_set1_ps(nd.dW0[i*N + j]);
					}
					b0[i] = _mm256_set1_ps(nd.b0[i]);
					db0[i] = _mm256_set1_ps(nd.db0[i]);
					W1[i] = _mm256_set1_ps(nd.W1[i]);
					dW1[i] = _mm256_set1_ps(nd.dW1[i]);
				}
				db1 = _mm256_set1_ps(nd.db1);

				float*	out_err_ptr;
				float*	in_err_ptr[N];
				float*	in_sig_ptr[N];

				out_err_ptr = (float*)out_err_buf.GetPtr(node);
				for (int i = 0; i < N; ++i) {
					in_err_ptr[i] = (float*)in_err_buf.GetPtr(nd.input[i]);
					in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(nd.input[i]);
				}

#pragma omp parallel for
				for (int frame = 0; frame < frame_size; frame += 8) {
					__m256	in_sig[N];
					__m256	in_err[N];
					for (int i = 0; i < N; ++i) {
						in_sig[i] = _mm256_load_ps(&in_sig_ptr[i][frame]);
						in_err[i] = _mm256_load_ps(&in_err_ptr[i][frame]);
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
						_mm256_store_ps(&in_err_ptr[i][frame], in_err[i]);
					}
				}

				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						nd.dW0[i*N + j] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW0[i][j]));
					}
					nd.db0[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db0[i]));
					nd.dW1[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW1[i]));
				}
				nd.db1 += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db1));
			}
		}
	}
#else
	void Backward(void)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		auto out_err_buf = this->GetOutputErrorBuffer();

		auto node_size = this->GetOutputNodeSize();
		const __m256	zero = _mm256_set1_ps(0);
		
		if (typeid(T) == typeid(float)) {
			INDEX frame_size = (m_frame_size + 7) / 8 * 8;

			float* tmp_err_buf = (float *)aligned_memory_alloc(node_size*N*frame_size*sizeof(float), 32);
			
#pragma omp parallel for
			for (int node = 0; node < (int)node_size; ++node) {
				auto& nd = m_node[node];

				__m256	W0[M][N];
				__m256	b0[M];
				__m256	dW0[M][N];
				__m256	db0[M];
				__m256	W1[M];
				__m256	dW1[M];
				__m256	db1;
				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						W0[i][j] = _mm256_set1_ps(nd.W0[i*N + j]);
						dW0[i][j] = _mm256_set1_ps(nd.dW0[i*N + j]);
					}
					b0[i] = _mm256_set1_ps(nd.b0[i]);
					db0[i] = _mm256_set1_ps(nd.db0[i]);
					W1[i] = _mm256_set1_ps(nd.W1[i]);
					dW1[i] = _mm256_set1_ps(nd.dW1[i]);
				}
				db1 = _mm256_set1_ps(nd.db1);

				float*	out_err_ptr;
				float*	in_sig_ptr[N];

				float*	tmp_err_ptr = &tmp_err_buf[node * N*frame_size];


				out_err_ptr = (float*)out_err_buf.GetPtr(node);
				for (int i = 0; i < N; ++i) {
					in_sig_ptr[i] = (float*)in_sig_buf.GetPtr(nd.input[i]);
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
						nd.dW0[i*N + j] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW0[i][j]));
					}
					nd.db0[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db0[i]));
					nd.dW1[i] += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW1[i]));
				}
				nd.db1 += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(db1));
			}

			// 足しこみ
			auto in_err_buf = this->GetInputErrorBuffer();
			in_err_buf.Clear();
			for (int node = 0; node < (int)node_size; ++node) {
				auto& nd = m_node[node];

				float*	in_err_ptr[N];
				for (int i = 0; i < N; ++i) {
					in_err_ptr[i] = (float*)in_err_buf.GetPtr(nd.input[i]);
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
	}
#endif

	void Update(void)
	{
		auto node_size = this->GetOutputNodeSize();

		// update
		for (auto& nd : m_node) {
			nd.optimizer_W0->Update(nd.W0, nd.dW0);
			nd.optimizer_b0->Update(nd.b0, nd.db0);
			nd.optimizer_W1->Update(nd.W1, nd.dW1);
			nd.optimizer_b1->Update(nd.b1, nd.db1);
		}

		// clear
		for (auto& nd : m_node) {
			std::fill(nd.dW0.begin(), nd.dW0.end(), (T)0);
			std::fill(nd.db0.begin(), nd.db0.end(), (T)0);
			std::fill(nd.dW1.begin(), nd.dW1.end(), (T)0);
			nd.db1 = 0;
		}

		// clip
		if (m_binary_mode) {
			for (auto& nd : m_node) {
				for (auto& W0 : nd.W0) { W0 = std::min((T)+1, std::max((T)-1, W0)); }
				for (auto& W1 : nd.W1) { W1 = std::min((T)+1, std::max((T)-1, W1)); }
			}
		}
	}

public:

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
};


}
