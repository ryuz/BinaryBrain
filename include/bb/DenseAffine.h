// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>

#include <Eigen/Core>

#include "bb/Layer.h"


namespace bb {


// Affineレイヤー
template <typename T = float>
class DenseAffine : public Layer
{
protected:
	using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using Stride = Eigen::Stride<Eigen::Dynamic, 1>;
	using MatMap = Eigen::Map<Matrix, 0, Stride>;
	using VecMap = Eigen::Map<Vector>;

    index_t                     m_input_node_size = 0;
    indices_t                   m_input_shape;
    index_t                     m_output_node_size = 0;
	indices_t	                m_output_shape;

    FrameBuffer                 m_x;
    FrameBuffer                 m_y;
    FrameBuffer                 m_dx;

//	Matrix		                m_W;
//	Vector		                m_b;
//	Matrix		                m_dW;
//	Vector		                m_db;
	std::shared_ptr<Tensor>		m_W;
	std::shared_ptr<Tensor>		m_b;
	std::shared_ptr<Tensor>		m_dW;
	std::shared_ptr<Tensor>		m_db;
    
    std::mt19937_64             m_mt;

	bool	                	m_binary_mode = false;

protected:
	DenseAffine() {
        m_W  = std::make_shared<Tensor>();
        m_b  = std::make_shared<Tensor>();
        m_dW = std::make_shared<Tensor>();
        m_db = std::make_shared<Tensor>();
    }

   	void CommandProc(std::vector<std::string> args)
	{
        // バイナリモード設定
        if ( args.size() == 2 && args[0] == "binary" )
        {
            m_binary_mode = EvalBool(args[1]);
        }

        /*
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
        */
	}


public:
	~DenseAffine() {}		// デストラクタ

    struct create_t
    {
        indices_t       output_shape;
        std::uint64_t   seed = 1;
    };

    static std::shared_ptr<DenseAffine> Create(create_t const &create)
    {
        auto self = std::shared_ptr<DenseAffine>(new DenseAffine);
        BB_ASSERT(!create.output_shape.empty());

        self->m_mt.seed(create.seed);

        self->m_output_shape = create.output_shape;
        self->m_output_node_size = GetShapeSize(self->m_output_shape);

        return self;
    }

    static std::shared_ptr<DenseAffine> Create(indices_t const &output_shape)
    {
        create_t create;
        create.output_shape = output_shape;
        return Create(create);
    }

    static std::shared_ptr<DenseAffine> Create(index_t output_node_size)
    {
        create_t create;
        create.output_shape.resize(1);
        create.output_shape[0] = output_node_size;
        return Create(create);
    }

    std::string GetClassName(void) const { return "DenseAffine"; }
	
  	Tensor       &W(void)       { return *m_W; }
	Tensor const &W(void) const { return *m_W; }
  	Tensor       &b(void)       { return *m_b; }
	Tensor const &b(void) const { return *m_b; }
   
   	Tensor       &dW(void)       { return *m_dW; }
	Tensor const &dW(void) const { return *m_dW; }
  	Tensor       &db(void)       { return *m_db; }
	Tensor const &db(void) const { return *m_db; }

	auto lock_W(void)             { return m_W->GetPtr<T>(); }
	auto lock_W_const(void) const { return m_W->GetConstPtr<T>(); }
	auto lock_b(void)             { return m_b->GetPtr<T>(); }
	auto lock_b_const(void) const { return m_b->GetConstPtr<T>(); }

	auto lock_dW(void)             { return m_dW->GetPtr<T>(); }
	auto lock_dW_const(void) const { return m_dW->GetConstPtr<T>(); }
	auto lock_db(void)             { return m_db->GetPtr<T>(); }
	auto lock_db_const(void) const { return m_db->GetConstPtr<T>(); }


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

        // パラメータ初期化
        m_W->Resize(DataType<T>::type, m_output_node_size, m_input_node_size);      m_W->InitNormalDistribution(0.0, 1.0, m_mt());
        m_b->Resize(DataType<T>::type, m_output_node_size);                         m_b->InitNormalDistribution(0.0, 1.0, m_mt());
        m_dW->Resize(DataType<T>::type, m_output_node_size, m_input_node_size);     m_dW->FillZero();
        m_db->Resize(DataType<T>::type, m_output_node_size);                        m_db->FillZero();

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
        BB_ASSERT(GetShapeSize(shape) == m_input_node_size);
        m_output_shape = shape;
    }
    

    Variables GetParameters(void)
    {
        Variables parameters;
        parameters.PushBack(m_W);
        parameters.PushBack(m_b);
        return parameters;
    }

    Variables GetGradients(void)
    {
        Variables gradients;
        gradients.PushBack(m_dW);
        gradients.PushBack(m_db);
        return gradients;
    }


    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        BB_ASSERT(x.GetType() == DataType<T>::type);
        BB_ASSERT(x.GetNodeSize() == m_input_node_size);

        // backwardの為に保存
        m_x = x;

        // SetInputShpaeされていなければ初回に設定
        if (m_x.GetNodeSize() != m_input_node_size) {
            SetInputShape(m_x.GetShape());
        }

        // フレーム数
        auto frame_size   = x.GetFrameSize();

        // 出力を設定
        m_y.Resize(DataType<T>::type, m_x.GetFrameSize(), m_output_shape);

        {
            auto x_ptr = m_x.GetConstPtr<T>();
            auto y_ptr = m_y.GetPtr<T>();
            auto W_ptr = lock_W_const();
            auto b_ptr = lock_b_const();

            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t output_node = 0; output_node < m_output_node_size; ++output_node) {
                    y_ptr.Set(frame, output_node, b_ptr(output_node));
                    for (index_t input_node = 0; input_node < m_input_node_size; ++input_node) {
                        y_ptr.Add(frame, output_node, x_ptr.Get(frame, input_node) * W_ptr(output_node, input_node));
                    }
                }
            }

            return m_y;
        }


        if ( 0 ) {
            auto x_ptr = m_x.GetMemoryConstPtr();
            auto y_ptr = m_y.GetMemoryPtr();
            auto W_ptr = lock_W_const();
            auto b_ptr = lock_b_const();

            auto frame_size   = x.GetFrameSize();

    //		Eigen::Map<Matrix> x((T*)m_input_signal_buffer.GetBuffer(), m_input_signal_buffer.GetFrameStride() / sizeof(T), m_input_size);
    //		Eigen::Map<Matrix> y((T*)m_output_signal_buffer.GetBuffer(), m_output_signal_buffer.GetFrameStride() / sizeof(T), m_output_size);

    //     MatMap W(W_ptr.GetAddr(), );

	//	    MatMap x((T*)x_ptr.GetAddr(), frame_size, m_input_size, x.GetFrameStride() / sizeof(T), 1));
	//	    MatMap y((T*)y_ptr.GetAddr(), frame_size, m_output_size, y.GetFrameStride() / sizeof(T), 1));

	//	    y = x * m_W;
	//	    y.rowwise() += m_b;
        }
	}


    FrameBuffer Backward(FrameBuffer dy)
    {
        BB_ASSERT(dy.GetType() == DataType<T>::type);

        // フレーム数
        auto frame_size = dy.GetFrameSize();

        m_dx.Resize(DataType<T>::type, dy.GetFrameSize(), m_input_node_size);

        m_dx.FillZero();
        m_dW->FillZero();
        m_db->FillZero();

        {
            auto x_ptr  = m_x.GetConstPtr<T>();
            auto dy_ptr = dy.GetConstPtr<T>();
            auto dx_ptr = m_dx.GetPtr<T>();
            auto W_ptr  = lock_W_const();
            auto b_ptr  = lock_b_const();
            auto dW_ptr = lock_dW();
            auto db_ptr = lock_db();

            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t output_node = 0; output_node < m_output_node_size; ++output_node) {
                    auto grad = dy_ptr.Get(frame, output_node);
                    db_ptr(output_node) += grad;
                    for (index_t input_node = 0; input_node < m_input_node_size; ++input_node) {
                        dx_ptr.Add(frame, input_node, grad * W_ptr(output_node, input_node));
                        dW_ptr(output_node, input_node) += grad * x_ptr.Get(frame, input_node);
                    }
                }
            }

            return m_dx;
        }
    }
    
	
#if 0
	void Backward(void)
	{
//		Eigen::Map<Matrix> dy((T*)m_output_error_buffer.GetBuffer(), m_output_error_buffer.GetFrameStride() / sizeof(T), m_output_size);
//		Eigen::Map<Matrix> dx((T*)m_input_error_buffer.GetBuffer(), m_input_error_buffer.GetFrameStride() / sizeof(T), m_input_size);
//		Eigen::Map<Matrix> x((T*)m_input_signal_buffer.GetBuffer(), m_input_signal_buffer.GetFrameStride() / sizeof(T), m_input_size);
		MatMap dy((T*)this->m_output_error_buffer.GetBuffer(), m_frame_size, m_output_size, Stride(this->m_output_error_buffer.GetFrameStride() / sizeof(T), 1));
		MatMap dx((T*)this->m_input_error_buffer.GetBuffer(), m_frame_size, m_input_size, Stride(this->m_input_error_buffer.GetFrameStride() / sizeof(T), 1));
		MatMap x((T*)this->m_input_signal_buffer.GetBuffer(), m_frame_size, m_input_size, Stride(this->m_input_signal_buffer.GetFrameStride() / sizeof(T), 1));

		dx = dy * m_W.transpose();
		m_dW = x.transpose() * dy;
		m_db = dy.colwise().sum();
	}
#endif

public:
	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
//		archive(cereal::make_nvp("input_size", m_input_size));
//		archive(cereal::make_nvp("output_size", m_output_size));
//		archive(cereal::make_nvp("W", W));
//		archive(cereal::make_nvp("b", b));
	}

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
//		archive(cereal::make_nvp("input_size", m_input_size));
//		archive(cereal::make_nvp("output_size", m_output_size));
		
//		std::vector< std::vector<T> >	W(m_output_size);
//		std::vector<T>					b(m_output_size);
//		archive(cereal::make_nvp("W", W));
//		archive(cereal::make_nvp("b", b));
	}

	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
//		archive(cereal::make_nvp("NeuralNetDenseAffine", *this));
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
//		archive(cereal::make_nvp("NeuralNetDenseAffine", *this));
	}

};

}