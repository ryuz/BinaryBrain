// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <array>
#include <vector>
#include "bb/LutLayer.h"


namespace bb {


// テーブルサイズ固定LUT
template <int N = 6, typename FT = Bit, typename BT = float>
class BinaryLutN : public LutLayer<FT, BT>
{
protected:
    indices_t       m_input_shape;
    indices_t       m_output_shape;

    FrameBuffer     m_x;
    FrameBuffer     m_y;

    struct LutNode {
		std::array< std::int8_t, (1 << N) > table;
		std::array< size_t, N >             input;
	};

	std::vector<LutNode>		          	m_lut;

    std::mt19937_64                         m_mt;

protected:
    BinaryLutN() {}

    /*
 	void CommandProc(std::vector<std::string> args)
	{
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
    */

public:
	~BinaryLutN() {}

    struct create_t
    {
        indices_t       output_shape;
        std::uint64_t   seed = 1;
    };

    static std::shared_ptr<BinaryLutN> Create(create_t const &create)
    {
        auto self = std::shared_ptr<BinaryLutN>(new BinaryLutN);
        BB_ASSERT(!create.output_shape.empty());

        self->m_mt.seed(create.seed);

        self->m_output_shape = create.output_shape;
        self->m_lut.resize(GetShapeSize(self->m_output_shape));

        return self;
    }

    static std::shared_ptr<BinaryLutN> Create(indices_t const &output_shape, std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.seed         = seed;
        return Create(create);
    }

    static std::shared_ptr<BinaryLutN> Create(index_t output_node_size, std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape.resize(1);
        create.output_shape[0] = output_node_size;
        create.seed            = seed;
        return Create(create);
    }

	std::string GetClassName(void) const { return "BinaryLutN"; }


  	// 疎結合の管理
    index_t GetNodeInputSize(index_t node) const
    {
        return N;
    }

    void SetNodeInput(index_t node, index_t input_index, index_t input_node)
    {
        BB_ASSERT(node >= 0 && (size_t)node < m_lut.size());
        BB_ASSERT(input_index >= 0 && input_index < N);
        BB_DEBUG_ASSERT(input_node >= 0 && input_node < GetInputNodeSize());
        m_lut[node].input[input_index] = input_node;
    }

    index_t GetNodeInput(index_t node, index_t input_index) const
    {
        BB_ASSERT(node >= 0 && (size_t)node < m_lut.size());
        BB_ASSERT(input_index >= 0 && input_index < N);
        return m_lut[node].input[input_index];
    }
    

    // LUT操作の定義
    int GetLutTableSize(index_t node) const
    {
        return (1 << N);
    }

    void SetLutTable(index_t node, int bitpos, bool value)
    {
        BB_ASSERT(node >= 0 && (size_t)node < m_lut.size());
        BB_ASSERT(bitpos >= 0 && bitpos < (1 << N));
        m_lut[node].table[bitpos] = value;
    }

    bool GetLutTable(index_t node, int bitpos) const
    {
        BB_ASSERT(node >= 0 && (size_t)node < m_lut.size());
        BB_ASSERT(bitpos >= 0 && bitpos < (1 << N));
        return (m_lut[node].table[bitpos] != 0);
    }

    bool GetLutInput(index_t frame, index_t node, int bitpos) const
    {
        auto input_node = GetNodeInput(node, (index_t)bitpos);
        auto ptr = m_x.GetConstPtr<FT>();
        return ptr.Get(frame, input_node);
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
        
        // 接続初期化
//      m_input_index.Resize(m_output_node_size, N);
        InitializeNodeInput(m_mt());

        // テーブル初期化
        InitializeLutTable(m_mt());

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


    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_input_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_output_shape;
    }
    
    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        BB_ASSERT(x.GetType() == DataType<FT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x.GetShape() != m_input_shape) {
            SetInputShape(m_x.GetShape());
        }
        
        // 出力を設定
        m_y.Resize(DataType<FT>::type, x.GetFrameSize(), m_output_shape);


    	{
            auto x_ptr = x.GetConstPtr<FT>();
            auto y_ptr = m_y.GetPtr<FT>();

            index_t frame_size = x.GetFrameSize();
            index_t node_size  = GetOutputNodeSize();

       		for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
			        int index = 0;
			        int mask = 1;
			        for (index_t i = 0; i < N; i++) {
    				    index_t input_node = GetNodeInput(node, i);
	    			    bool input_signal = x_ptr.Get(frame, input_node);
    		    		index |= input_signal ? mask : 0;
	    		    	mask <<= 1;
		    	    }
    			    bool output_signal = GetLutTable(node, index);
    			    y_ptr.Set(frame, node, output_signal);
                }
            }

            return m_y;
		}
    }


    FrameBuffer Backward(FrameBuffer dy)
    {
        FrameBuffer dx(DataType<BT>::type, dy.GetFrameSize(), m_input_shape);
        return dx;
    }


    /*
    virtual void Forward(bool train = true)
	{
		INDEX node_size = this->GetOutputNodeSize();
		int   lut_input_size = this->GetLutInputSize();

		#pragma omp parallel for
		for ( int node = 0; node < (int)node_size; ++node) {
			ForwardNode(node);
		}
	}
    */
	

    // なんか昔こころみたっぽい
    /*
    void Backward(void)
	{
		auto& out_err = this->GetOutputErrorBuffer();
		auto& in_err = this->GetInputErrorBuffer();

		INDEX frame_size = this->GetOutputFrameSize();
		INDEX node_size = this->GetOutputNodeSize();
		int lut_input_size = this->GetLutInputSize();
		int lut_table_size = this->GetLutTableSize();

		// ゼロ初期化
		INDEX input_node_size = this->GetInputNodeSize();
		for (INDEX node = 0; node < input_node_size; ++node) {
			for (INDEX frame = 0; frame < frame_size; ++frame) {
				in_err.template Set<T>(frame, node, 0);
			}
		}

		std::mt19937_64 mt(1);

		// 計算
		std::vector<T> table_err(lut_table_size);
		for (INDEX node = 0; node < node_size; ++node) {
			std::fill(table_err.begin(), table_err.end(), (T)0);
			for (INDEX frame = 0; frame < frame_size; ++frame) {
				// 入力値取得
				int input_index = this->GetLutInputIndex(frame, node);
				T err = out_err.template Get<T>(frame, node);

				// テーブルに対する誤差計算
				table_err[input_index] += err;	// 積算していく
			}

			for (int bitpos = 0; bitpos < lut_input_size; ++bitpos) {
				if ( std::abs(table_err[bitpos]) > (mt() % 16)+5 ) {
					this->SetLutTable(node, bitpos, table_err[bitpos] > 0);
				}
			}
			
			for (INDEX frame = 0; frame < frame_size; ++frame) {
				int input_index = this->GetLutInputIndex(frame, node);
				T err = out_err.template Get<T>(frame, node);

				bool val = GetLutTable(node, input_index);
				if ((val && err < 0) || (val && err > 0)) {

					// 入力に対する伝播誤差計算
					int mask = 1;
			//		for (int bitpos = 0; bitpos < lut_input_size; ++bitpos) {
					{
						int bitpos = (int)(mt() % lut_input_size);

						INDEX input_node = GetLutInput(node, bitpos);
						// 各入力項に対するテーブルの偏微分を計算
						int index0 = (input_index & ~mask);
						int index1 = (input_index | mask);
						bool val0 = this->GetLutTable(node, index0);
						bool val1 = this->GetLutTable(node, index1);

						if (!val0 && val1) {
							in_err.template Set<T>(frame, input_node, in_err.template Get<T>(frame, input_node) + err);
						}
						else if (val0 && !val1) {
							in_err.template Set<T>(frame, input_node, in_err.template Get<T>(frame, input_node) - err);
						}
						mask <<= 1;
					}
				}
			}

		}
	}
	*/

};


}