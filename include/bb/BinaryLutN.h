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
    bool                    m_host_only = false;
    bool                    m_host_simd = true;

    indices_t               m_input_shape;
    indices_t               m_output_shape;

//    FrameBuffer             m_x_buf;
//    FrameBuffer             m_y_buf;

    static int const        m_table_size = (1 << N);
    static int const        m_table_bits = sizeof(std::int32_t) * 8;
    static int const        m_table_unit = (m_table_size + (m_table_bits - 1)) / m_table_bits;
    Tensor_<std::int32_t>   m_table;

    Tensor_<std::int32_t>   m_input_index;

    std::mt19937_64         m_mt;

public:
    struct create_t
    {
        indices_t       output_shape;
        std::uint64_t   seed = 1;
    };

protected:
    BinaryLutN(create_t const &create)
    {
        BB_ASSERT(!create.output_shape.empty());
        m_mt.seed(create.seed);
        m_output_shape = create.output_shape;
        m_input_index.Resize(GetShapeSize(m_output_shape), (index_t)N);
        m_table.Resize(GetShapeSize(m_output_shape), (index_t)m_table_unit);
    }

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

public:
    ~BinaryLutN() {}

    static std::shared_ptr<BinaryLutN> Create(create_t const &create)
    {
        return std::shared_ptr<BinaryLutN>(new BinaryLutN(create));
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

    // python用
    static std::shared_ptr<BinaryLutN> CreateEx(
                indices_t       output_shape,
                std::uint64_t   seed = 1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.seed         = seed;
        return Create(create);
    }

    std::string GetClassName(void) const { return "BinaryLutN"; }

    auto lock_InputIndex(void)             { return m_input_index.Lock(); }
    auto lock_InputIndex_const(void) const { return m_input_index.LockConst(); }

    // 疎結合の管理
    index_t GetNodeConnectionSize(index_t node) const
    {
        return N;
    }

    void SetNodeConnectionIndex(index_t node, index_t input_index, index_t input_node)
    {
        BB_ASSERT(node >= 0 && node < GetShapeSize(m_output_shape));
        BB_ASSERT(input_index >= 0 && input_index < N);
        BB_DEBUG_ASSERT(input_node >= 0 && input_node < GetInputNodeSize());

        auto ptr = lock_InputIndex();
        ptr(node, input_index) = (std::int32_t)input_node;
    }

    index_t GetNodeConnectionIndex(index_t node, index_t input_index) const
    {
        BB_ASSERT(node >= 0 && node < GetShapeSize(m_output_shape));
        BB_ASSERT(input_index >= 0 && input_index < N);
        
        auto ptr = lock_InputIndex_const();
        return (index_t)ptr(node, input_index);
    }
    
    // LUT操作の定義
    int GetLutTableSize(index_t node) const
    {
        return m_table_size;
    }

    void SetLutTable(index_t node, int bitpos, bool value)
    {
        BB_ASSERT(node >= 0 && node < GetShapeSize(m_output_shape));
        BB_ASSERT(bitpos >= 0 && bitpos < m_table_size);

        int idx = bitpos / m_table_bits;
        int bit = bitpos % m_table_bits;

        auto ptr = m_table.Lock();
        if ( value ) {
            ptr(node, idx) |= (1 << bit);
        }
        else {
            ptr(node, idx) &= ~(1 << bit);
        }
    }

    bool GetLutTable(index_t node, int bitpos) const
    {
        BB_ASSERT(node >= 0 && node < GetShapeSize(m_output_shape));
        BB_ASSERT(bitpos >= 0 && bitpos < (1 << N));

        int idx = bitpos / m_table_bits;
        int bit = bitpos % m_table_bits;

        auto ptr = m_table.LockConst();
        return ((ptr(node, idx) & (1 << bit)) != 0);
    }

    /*
    bool GetLutInput(index_t frame, index_t node, int bitpos) const
    {
        auto input_node = GetNodeInput(node, (index_t)bitpos);
        auto ptr = m_x_buf.LockConst<FT>();
        return ptr.Get(frame, input_node);
    }
    */


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
    inline void lut6_mask(__m256i& msk, __m256i lut, __m256i val[6])
    {
        lut = lut_mask_unit<LUT, 0>(val[0], lut);
        lut = lut_mask_unit<LUT, 1>(val[1], lut);
        lut = lut_mask_unit<LUT, 2>(val[2], lut);
        lut = lut_mask_unit<LUT, 3>(val[3], lut);
        lut = lut_mask_unit<LUT, 4>(val[4], lut);
        lut = lut_mask_unit<LUT, 5>(val[5], lut);
        msk = _mm256_or_si256(msk, lut);
    }

    inline bool GetLutTableFromPtr(Tensor_<std::int32_t>::ConstPtr ptr, index_t node, int index)
    {
        auto idx = index / m_table_bits;
        auto bit = index % m_table_bits;
        return (((ptr(node, idx) >> bit) & 1) != 0);
    }

public:
    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_input_shape) {
            SetInputShape(x_buf.GetShape());
        }
        
        // 出力を設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), m_output_shape, DataType<FT>::type);

#ifdef BB_WITH_CUDA
        if ( N == 6 && DataType<FT>::type == BB_TYPE_BIT && !m_host_only
                && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            auto x_ptr           = x_buf.LockDeviceMemoryConst();
            auto y_ptr           = y_buf.LockDeviceMemory(true);
            auto input_index_ptr = m_input_index.LockDeviceMemoryConst();
            auto table_ptr       = m_table.LockDeviceMemoryConst();

            bbcu_bit_BinatyLut6_Forward
                (
                    (int const *)x_ptr.GetAddr(),
                    (int       *)y_ptr.GetAddr(),
                    (int const *)input_index_ptr.GetAddr(),
                    (int const *)table_ptr.GetAddr(),
                    (int        )y_buf.GetNodeSize(),
                    (int        )y_buf.GetFrameSize(),
                    (int        )(y_buf.GetFrameStride() / sizeof(int))
                );

            return y_buf;
        }
#endif

        if ( N == 6 && DataType<FT>::type == BB_TYPE_BIT && m_host_simd ) {
            auto x_ptr = x_buf.LockConst<Bit>();
            auto y_ptr = y_buf.Lock<Bit>(true);

            auto input_index_ptr = m_input_index.LockConst();
            auto table_ptr       = m_table.LockConst();

            index_t node_size  = y_buf.GetNodeSize();
            index_t frame_size = y_buf.GetFrameStride() / sizeof(__m256i);

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                __m256i*    x_addr[6];
                __m256i*    y_addr;
                __m256i     x[6];

                x_addr[0] = (__m256i*)x_ptr.GetAddr(input_index_ptr(node, 0));
                x_addr[1] = (__m256i*)x_ptr.GetAddr(input_index_ptr(node, 1));
                x_addr[2] = (__m256i*)x_ptr.GetAddr(input_index_ptr(node, 2));
                x_addr[3] = (__m256i*)x_ptr.GetAddr(input_index_ptr(node, 3));
                x_addr[4] = (__m256i*)x_ptr.GetAddr(input_index_ptr(node, 4));
                x_addr[5] = (__m256i*)x_ptr.GetAddr(input_index_ptr(node, 5));
                y_addr    = (__m256i*)y_ptr.GetAddr(node);

                char    table[64];
                std::int32_t t0 = table_ptr(node, 0);
                std::int32_t t1 = table_ptr(node, 1);
                for (int i = 0; i < 32; ++i) { table[i]    = (t0 & (1 << i)) ? -1 : 0; }
                for (int i = 0; i < 32; ++i) { table[i+32] = (t1 & (1 << i)) ? -1 : 0; }

                for (index_t frame = 0; frame < frame_size; ++frame) {
                    // input
                    x[0] = _mm256_loadu_si256(&x_addr[0][frame]);
                    x[1] = _mm256_loadu_si256(&x_addr[1][frame]);
                    x[2] = _mm256_loadu_si256(&x_addr[2][frame]);
                    x[3] = _mm256_loadu_si256(&x_addr[3][frame]);
                    x[4] = _mm256_loadu_si256(&x_addr[4][frame]);
                    x[5] = _mm256_loadu_si256(&x_addr[5][frame]);

                    // LUT
                    __m256i y = _mm256_set1_epi8(0);
                    lut6_mask< 0>(y, _mm256_set1_epi8(table[0]), x);
                    lut6_mask< 1>(y, _mm256_set1_epi8(table[1]), x);
                    lut6_mask< 2>(y, _mm256_set1_epi8(table[2]), x);
                    lut6_mask< 3>(y, _mm256_set1_epi8(table[3]), x);
                    lut6_mask< 4>(y, _mm256_set1_epi8(table[4]), x);
                    lut6_mask< 5>(y, _mm256_set1_epi8(table[5]), x);
                    lut6_mask< 6>(y, _mm256_set1_epi8(table[6]), x);
                    lut6_mask< 7>(y, _mm256_set1_epi8(table[7]), x);
                    lut6_mask< 8>(y, _mm256_set1_epi8(table[8]), x);
                    lut6_mask< 9>(y, _mm256_set1_epi8(table[9]), x);
                    lut6_mask<10>(y, _mm256_set1_epi8(table[10]), x);
                    lut6_mask<11>(y, _mm256_set1_epi8(table[11]), x);
                    lut6_mask<12>(y, _mm256_set1_epi8(table[12]), x);
                    lut6_mask<13>(y, _mm256_set1_epi8(table[13]), x);
                    lut6_mask<14>(y, _mm256_set1_epi8(table[14]), x);
                    lut6_mask<15>(y, _mm256_set1_epi8(table[15]), x);
                    lut6_mask<16>(y, _mm256_set1_epi8(table[16]), x);
                    lut6_mask<17>(y, _mm256_set1_epi8(table[17]), x);
                    lut6_mask<18>(y, _mm256_set1_epi8(table[18]), x);
                    lut6_mask<19>(y, _mm256_set1_epi8(table[19]), x);
                    lut6_mask<20>(y, _mm256_set1_epi8(table[20]), x);
                    lut6_mask<21>(y, _mm256_set1_epi8(table[21]), x);
                    lut6_mask<22>(y, _mm256_set1_epi8(table[22]), x);
                    lut6_mask<23>(y, _mm256_set1_epi8(table[23]), x);
                    lut6_mask<24>(y, _mm256_set1_epi8(table[24]), x);
                    lut6_mask<25>(y, _mm256_set1_epi8(table[25]), x);
                    lut6_mask<26>(y, _mm256_set1_epi8(table[26]), x);
                    lut6_mask<27>(y, _mm256_set1_epi8(table[27]), x);
                    lut6_mask<28>(y, _mm256_set1_epi8(table[28]), x);
                    lut6_mask<29>(y, _mm256_set1_epi8(table[29]), x);
                    lut6_mask<30>(y, _mm256_set1_epi8(table[30]), x);
                    lut6_mask<31>(y, _mm256_set1_epi8(table[31]), x);
                    lut6_mask<32>(y, _mm256_set1_epi8(table[32]), x);
                    lut6_mask<33>(y, _mm256_set1_epi8(table[33]), x);
                    lut6_mask<34>(y, _mm256_set1_epi8(table[34]), x);
                    lut6_mask<35>(y, _mm256_set1_epi8(table[35]), x);
                    lut6_mask<36>(y, _mm256_set1_epi8(table[36]), x);
                    lut6_mask<37>(y, _mm256_set1_epi8(table[37]), x);
                    lut6_mask<38>(y, _mm256_set1_epi8(table[38]), x);
                    lut6_mask<39>(y, _mm256_set1_epi8(table[39]), x);
                    lut6_mask<40>(y, _mm256_set1_epi8(table[40]), x);
                    lut6_mask<41>(y, _mm256_set1_epi8(table[41]), x);
                    lut6_mask<42>(y, _mm256_set1_epi8(table[42]), x);
                    lut6_mask<43>(y, _mm256_set1_epi8(table[43]), x);
                    lut6_mask<44>(y, _mm256_set1_epi8(table[44]), x);
                    lut6_mask<45>(y, _mm256_set1_epi8(table[45]), x);
                    lut6_mask<46>(y, _mm256_set1_epi8(table[46]), x);
                    lut6_mask<47>(y, _mm256_set1_epi8(table[47]), x);
                    lut6_mask<48>(y, _mm256_set1_epi8(table[48]), x);
                    lut6_mask<49>(y, _mm256_set1_epi8(table[49]), x);
                    lut6_mask<50>(y, _mm256_set1_epi8(table[50]), x);
                    lut6_mask<51>(y, _mm256_set1_epi8(table[51]), x);
                    lut6_mask<52>(y, _mm256_set1_epi8(table[52]), x);
                    lut6_mask<53>(y, _mm256_set1_epi8(table[53]), x);
                    lut6_mask<54>(y, _mm256_set1_epi8(table[54]), x);
                    lut6_mask<55>(y, _mm256_set1_epi8(table[55]), x);
                    lut6_mask<56>(y, _mm256_set1_epi8(table[56]), x);
                    lut6_mask<57>(y, _mm256_set1_epi8(table[57]), x);
                    lut6_mask<58>(y, _mm256_set1_epi8(table[58]), x);
                    lut6_mask<59>(y, _mm256_set1_epi8(table[59]), x);
                    lut6_mask<60>(y, _mm256_set1_epi8(table[60]), x);
                    lut6_mask<61>(y, _mm256_set1_epi8(table[61]), x);
                    lut6_mask<62>(y, _mm256_set1_epi8(table[62]), x);
                    lut6_mask<63>(y, _mm256_set1_epi8(table[63]), x);

                    _mm256_storeu_si256(&y_addr[frame], y);
                }
            }

            return y_buf;
        }


        {
            // 汎用版
            auto x_ptr           = x_buf.LockConst<FT>();
            auto y_ptr           = y_buf.Lock<FT>();
            auto input_index_ptr = m_input_index.LockConst();
            auto table_ptr       = m_table.LockConst();

            index_t frame_size = x_buf.GetFrameSize();
            index_t node_size  = this->GetOutputNodeSize();

            #pragma omp parallel for
            for (index_t node = 0; node < node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    int index = 0;
                    int mask  = 1;
                    for (index_t i = 0; i < N; i++) {
                        index_t input_node = input_index_ptr(node, i);
                        bool x = (x_ptr.Get(frame, input_node) != 0);
                        index |= x ? mask : 0;
                        mask <<= 1;
                    }
                    auto y = GetLutTableFromPtr(table_ptr, node, index);
                    y_ptr.Set(frame, node, y);
                }
            }

            return y_buf;
        }
    }

    // Backwardは存在しない
    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), m_input_shape, DataType<BT>::type);
        return dx_buf;
    }
};


}