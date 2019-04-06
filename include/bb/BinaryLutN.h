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
    bool            m_host_only = false;
    bool            m_host_simd = false;

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

 	void CommandProc(std::vector<std::string> args)
	{
        // バイナリモード設定
//      if ( args.size() == 2 && args[0] == "binary" )
//      {
//          m_binary_mode = EvalBool(args[1]);
//      }

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
        m_lut[node].table[bitpos] = value ? -1 : 0;
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
        auto ptr = m_x.LockConst<FT>();
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
        this->InitializeNodeInput(m_mt());

        // テーブル初期化
        this->InitializeLutTable(m_mt());

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
        BB_ASSERT(GetShapeSize(shape) == this->m_output_node_size);
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
    

private:
    template<int LUT, int VAL>
    inline __m256i lut_mask_unit(__m256i& val, __m256i& lut)
    {
        if ((LUT & (1 << VAL)) == 0) {
            return _mm256_andnot_si256(val, lut);
        }
	    else {
	        return _mm256_and_si256(val, lut);
	    }
    }
    
    template<int LUT>
    inline void lut_mask(__m256i& msk, __m256i lut, __m256i val[6])
    {
	    lut = lut_mask_unit<LUT, 0>(val[0], lut);
	    lut = lut_mask_unit<LUT, 1>(val[1], lut);
	    lut = lut_mask_unit<LUT, 2>(val[2], lut);
	    lut = lut_mask_unit<LUT, 3>(val[3], lut);
	    lut = lut_mask_unit<LUT, 4>(val[4], lut);
	    lut = lut_mask_unit<LUT, 5>(val[5], lut);
	    msk = _mm256_or_si256(msk, lut);
    }

public:
    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        BB_ASSERT(x.GetType() == DataType<FT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x.GetShape() != m_input_shape) {
            SetInputShape(m_x.GetShape());
        }
        
        // 出力を設定
        m_y.Resize(DataType<FT>::type, x.GetFrameSize(), m_output_shape);

        if ( N == 6 && DataType<FT>::type == BB_TYPE_BIT && m_host_simd ) {
            auto x_ptr = x.LockConst<Bit>();
            auto y_ptr = m_y.Lock<Bit>(true);

            index_t node_size  = m_y.GetNodeSize();
            index_t frame_size = m_y.GetFrameStride() / sizeof(__m256i);

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {

        		auto& lut = m_lut[node];

		        __m256i*	in_sig_ptr[6];
		        __m256i*	out_sig_ptr;
		        __m256i		in_sig[6];

		        in_sig_ptr[0] = (__m256i*)x_ptr.GetAddr(lut.input[0]);
		        in_sig_ptr[1] = (__m256i*)x_ptr.GetAddr(lut.input[1]);
		        in_sig_ptr[2] = (__m256i*)x_ptr.GetAddr(lut.input[2]);
		        in_sig_ptr[3] = (__m256i*)x_ptr.GetAddr(lut.input[3]);
		        in_sig_ptr[4] = (__m256i*)x_ptr.GetAddr(lut.input[4]);
		        in_sig_ptr[5] = (__m256i*)x_ptr.GetAddr(lut.input[5]);
		        out_sig_ptr   = (__m256i*)y_ptr.GetAddr(node);

		        for (index_t i = 0; i < frame_size; i++) {
			        // input
			        in_sig[0] = _mm256_loadu_si256(&in_sig_ptr[0][i]);
			        in_sig[1] = _mm256_loadu_si256(&in_sig_ptr[1][i]);
			        in_sig[2] = _mm256_loadu_si256(&in_sig_ptr[2][i]);
			        in_sig[3] = _mm256_loadu_si256(&in_sig_ptr[3][i]);
			        in_sig[4] = _mm256_loadu_si256(&in_sig_ptr[4][i]);
			        in_sig[5] = _mm256_loadu_si256(&in_sig_ptr[5][i]);

			        // LUT
			        __m256i msk = _mm256_set1_epi8(0);
			        lut_mask<0>(msk, _mm256_set1_epi8(lut.table[0]), in_sig);
			        lut_mask<1>(msk, _mm256_set1_epi8(lut.table[1]), in_sig);
			        lut_mask<2>(msk, _mm256_set1_epi8(lut.table[2]), in_sig);
			        lut_mask<3>(msk, _mm256_set1_epi8(lut.table[3]), in_sig);
			        lut_mask<4>(msk, _mm256_set1_epi8(lut.table[4]), in_sig);
			        lut_mask<5>(msk, _mm256_set1_epi8(lut.table[5]), in_sig);
			        lut_mask<6>(msk, _mm256_set1_epi8(lut.table[6]), in_sig);
			        lut_mask<7>(msk, _mm256_set1_epi8(lut.table[7]), in_sig);
			        lut_mask<8>(msk, _mm256_set1_epi8(lut.table[8]), in_sig);
			        lut_mask<9>(msk, _mm256_set1_epi8(lut.table[9]), in_sig);
			        lut_mask<10>(msk, _mm256_set1_epi8(lut.table[10]), in_sig);
			        lut_mask<11>(msk, _mm256_set1_epi8(lut.table[11]), in_sig);
			        lut_mask<12>(msk, _mm256_set1_epi8(lut.table[12]), in_sig);
			        lut_mask<13>(msk, _mm256_set1_epi8(lut.table[13]), in_sig);
			        lut_mask<14>(msk, _mm256_set1_epi8(lut.table[14]), in_sig);
			        lut_mask<15>(msk, _mm256_set1_epi8(lut.table[15]), in_sig);
			        lut_mask<16>(msk, _mm256_set1_epi8(lut.table[16]), in_sig);
			        lut_mask<17>(msk, _mm256_set1_epi8(lut.table[17]), in_sig);
			        lut_mask<18>(msk, _mm256_set1_epi8(lut.table[18]), in_sig);
			        lut_mask<19>(msk, _mm256_set1_epi8(lut.table[19]), in_sig);
			        lut_mask<20>(msk, _mm256_set1_epi8(lut.table[20]), in_sig);
			        lut_mask<21>(msk, _mm256_set1_epi8(lut.table[21]), in_sig);
			        lut_mask<22>(msk, _mm256_set1_epi8(lut.table[22]), in_sig);
			        lut_mask<23>(msk, _mm256_set1_epi8(lut.table[23]), in_sig);
			        lut_mask<24>(msk, _mm256_set1_epi8(lut.table[24]), in_sig);
			        lut_mask<25>(msk, _mm256_set1_epi8(lut.table[25]), in_sig);
			        lut_mask<26>(msk, _mm256_set1_epi8(lut.table[26]), in_sig);
			        lut_mask<27>(msk, _mm256_set1_epi8(lut.table[27]), in_sig);
			        lut_mask<28>(msk, _mm256_set1_epi8(lut.table[28]), in_sig);
			        lut_mask<29>(msk, _mm256_set1_epi8(lut.table[29]), in_sig);
			        lut_mask<30>(msk, _mm256_set1_epi8(lut.table[30]), in_sig);
			        lut_mask<31>(msk, _mm256_set1_epi8(lut.table[31]), in_sig);
			        lut_mask<32>(msk, _mm256_set1_epi8(lut.table[32]), in_sig);
			        lut_mask<33>(msk, _mm256_set1_epi8(lut.table[33]), in_sig);
			        lut_mask<34>(msk, _mm256_set1_epi8(lut.table[34]), in_sig);
			        lut_mask<35>(msk, _mm256_set1_epi8(lut.table[35]), in_sig);
			        lut_mask<36>(msk, _mm256_set1_epi8(lut.table[36]), in_sig);
			        lut_mask<37>(msk, _mm256_set1_epi8(lut.table[37]), in_sig);
			        lut_mask<38>(msk, _mm256_set1_epi8(lut.table[38]), in_sig);
			        lut_mask<39>(msk, _mm256_set1_epi8(lut.table[39]), in_sig);
			        lut_mask<40>(msk, _mm256_set1_epi8(lut.table[40]), in_sig);
			        lut_mask<41>(msk, _mm256_set1_epi8(lut.table[41]), in_sig);
			        lut_mask<42>(msk, _mm256_set1_epi8(lut.table[42]), in_sig);
			        lut_mask<43>(msk, _mm256_set1_epi8(lut.table[43]), in_sig);
			        lut_mask<44>(msk, _mm256_set1_epi8(lut.table[44]), in_sig);
			        lut_mask<45>(msk, _mm256_set1_epi8(lut.table[45]), in_sig);
			        lut_mask<46>(msk, _mm256_set1_epi8(lut.table[46]), in_sig);
			        lut_mask<47>(msk, _mm256_set1_epi8(lut.table[47]), in_sig);
			        lut_mask<48>(msk, _mm256_set1_epi8(lut.table[48]), in_sig);
			        lut_mask<49>(msk, _mm256_set1_epi8(lut.table[49]), in_sig);
			        lut_mask<50>(msk, _mm256_set1_epi8(lut.table[50]), in_sig);
			        lut_mask<51>(msk, _mm256_set1_epi8(lut.table[51]), in_sig);
			        lut_mask<52>(msk, _mm256_set1_epi8(lut.table[52]), in_sig);
			        lut_mask<53>(msk, _mm256_set1_epi8(lut.table[53]), in_sig);
			        lut_mask<54>(msk, _mm256_set1_epi8(lut.table[54]), in_sig);
			        lut_mask<55>(msk, _mm256_set1_epi8(lut.table[55]), in_sig);
			        lut_mask<56>(msk, _mm256_set1_epi8(lut.table[56]), in_sig);
			        lut_mask<57>(msk, _mm256_set1_epi8(lut.table[57]), in_sig);
			        lut_mask<58>(msk, _mm256_set1_epi8(lut.table[58]), in_sig);
			        lut_mask<59>(msk, _mm256_set1_epi8(lut.table[59]), in_sig);
			        lut_mask<60>(msk, _mm256_set1_epi8(lut.table[60]), in_sig);
			        lut_mask<61>(msk, _mm256_set1_epi8(lut.table[61]), in_sig);
			        lut_mask<62>(msk, _mm256_set1_epi8(lut.table[62]), in_sig);
			        lut_mask<63>(msk, _mm256_set1_epi8(lut.table[63]), in_sig);

			        _mm256_storeu_si256(&out_sig_ptr[i], msk);
		        }
	        }

            return m_y;
        }


    	{
            // 汎用版
            auto x_ptr = x.LockConst<FT>();
            auto y_ptr = m_y.Lock<FT>();

            index_t frame_size = x.GetFrameSize();
            index_t node_size  = this->GetOutputNodeSize();

            #pragma omp parallel for
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

    // Backwardは存在しない
    FrameBuffer Backward(FrameBuffer dy)
    {
        FrameBuffer dx(DataType<BT>::type, dy.GetFrameSize(), m_input_shape);
        return dx;
    }


};


}