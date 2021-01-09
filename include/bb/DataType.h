// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2020 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <memory>
#include <cstdint>
#include <iostream>

#include "bb/Assert.h"
#include "bb/SimdSupport.h"


namespace bb {


// 現在、実数とバイナリを実装中。将来は整数も実装するかも
// 実数は float と double が切り替えできる程度にはテンプレートで
// 書いたつもりだが、double の需要は無いと思われる
// 将来整数をやる場合はいろいろ必要と思う


//#define BB_TYPE_BOOL          (0x0000 + 1)
#define BB_TYPE_BIT             (0x0000 + 1)
#define BB_TYPE_BINARY          (0x0000 + 2)

#define BB_TYPE_FP16            (0x0100 + 16)
#define BB_TYPE_FP32            (0x0100 + 32)
#define BB_TYPE_FP64            (0x0100 + 64)

#define BB_TYPE_INT8            (0x0200 + 8)
#define BB_TYPE_INT16           (0x0200 + 16)
#define BB_TYPE_INT32           (0x0200 + 32)
#define BB_TYPE_INT64           (0x0200 + 64)

#define BB_TYPE_UINT8           (0x0300 + 8)
#define BB_TYPE_UINT16          (0x0300 + 16)
#define BB_TYPE_UINT32          (0x0300 + 32)
#define BB_TYPE_UINT64          (0x0300 + 64)




// Index型 (配列等の添え字用、符号付き size_t としての扱い)
using index_t   = std::int64_t;                // 以前は std::intptr_t だったがデータ互換性上ややこしくなるので64bitに固定(ver4.1.0)
using indices_t = std::vector<index_t>;        // Tensorなどの多次元配列の添え字、shape用


inline void SaveIndex(std::ostream &os, index_t index)
{
    std::int64_t index64 = index;
    os.write((const char*)&index64, sizeof(index64));
}


inline index_t LoadIndex(std::istream &is)
{
    std::int64_t index64;
    is.read((char*)&index64, sizeof(index64));

    return (index_t)index64;
}


inline index_t CalcShapeSize(indices_t const & shape)
{
    index_t size = 1;
    for (auto s : shape) {
        size *= (index_t)s;
    }
    return size;
}


inline bool CalcNextIndices(indices_t& indices, indices_t const & shape)
{
    index_t size = indices.size();
    BB_DEBUG_ASSERT(shape.size() == size);
    for ( index_t i = size-1; i >= 0; --i) {
        if ( ++indices[i] < shape[i] ) {
            return true;
        }
        indices[i] = 0;
    }
    return false;
}


template <typename T=double>
inline indices_t RegurerlizeIndices(std::vector<T> indices, indices_t const &shape)
{
    BB_ASSERT(indices.size() == shape.size());

    // 負数の折り返し
    for ( index_t i = (index_t)indices.size()-1; i >= 0;  --i ) {
        while ( indices[i] < 0 ) {
            indices[i] += (T)shape[i];
        }
    }

    indices_t   reg_indices(indices.size());
    for ( index_t i = (index_t)indices.size()-1; i >= 0;  --i ) {
        reg_indices[i] = (index_t)indices[i] % shape[i];
    }

    return reg_indices;
}


inline indices_t ConvertIndexToIndices(index_t index, indices_t const & shape)
{
    indices_t indices(shape.size());
    BB_DEBUG_ASSERT(index >= 0 && index < CalcShapeSize(shape));

    for ( index_t i = (index_t)shape.size()-1; i >= 0;  --i ) {
        indices[i] = index % shape[i];
        index /= shape[i];
    }
    return indices;
}


inline index_t ConvertIndicesToIndex(indices_t const & indices, indices_t const & shape)
{
    BB_DEBUG_ASSERT(indices.size() == shape.size());

    index_t stride = 1;
    index_t index = 0;
    for ( index_t i = (index_t)shape.size()-1; i >= 0;  --i ) {
        BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < shape[i]);
        index += stride * indices[i];
        stride *= shape[i];
    }
    return index;
}


inline index_t ConvertIndicesToIndex(index_t i0, indices_t const & shape)
{
    BB_DEBUG_ASSERT(shape.size() == 1);
    BB_DEBUG_ASSERT(i0 >= 0 && i0 <= shape[0]);

    return i0;
}

inline index_t ConvertIndicesToIndex(index_t i0, index_t i1, indices_t const & shape)
{
    BB_DEBUG_ASSERT(shape.size() == 2);
    BB_DEBUG_ASSERT(i0 >= 0 && i0 <= shape[0]);
    BB_DEBUG_ASSERT(i1 >= 0 && i1 <= shape[1]);

    return (i0 * shape[1]) + i1;
}

inline index_t ConvertIndicesToIndex(index_t i0, index_t i1, index_t i2, indices_t const & shape)
{
    BB_DEBUG_ASSERT(shape.size() == 3);
    BB_DEBUG_ASSERT(i0 >= 0 && i0 <= shape[0]);
    BB_DEBUG_ASSERT(i1 >= 0 && i1 <= shape[1]);
    BB_DEBUG_ASSERT(i2 >= 0 && i2 <= shape[2]);

    return ((i0 * shape[1] + i1) * shape[2]) + i2;
}


inline index_t CalcIndexWithStride(indices_t const & indices, indices_t const & stride)
{
    BB_DEBUG_ASSERT(indices.size() == stride.size());

    index_t index = 0;
    for ( index_t i = (index_t)stride.size()-1; i >= 0;  --i ) {
        BB_DEBUG_ASSERT(indices[i] >= 0);
        index += stride[i] * indices[i];
    }
    return index;
}

inline index_t CalcIndexWithStride(indices_t const & indices, indices_t const & stride, indices_t const & shape)
{
    BB_DEBUG_ASSERT(indices.size() == shape.size());

    index_t index = 0;
    for ( index_t i = 0; i < (index_t)shape.size(); ++i ) {
        BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < shape[i]);
        index += stride[i] * indices[i];
    }
    return index;
}

inline std::ostream& operator<<(std::ostream& os, indices_t const &indices)
{
    bool first = true;
    os << "{";
    for (auto i : indices) {
        if ( !first ) {
            os << ", ";
        }
        os << i;
        first = false;
    }
    os << "}";

    return os;
}

inline void SaveIndices(std::ostream &os, indices_t const &indices)
{
    auto size = (index_t)indices.size();
    SaveIndex(os, (index_t)indices.size());
    for (index_t i = 0; i < size; ++i) {
        SaveIndex(os, indices[i]);
    }
}

inline indices_t LoadIndices(std::istream &is)
{
    indices_t indices;

    auto size = LoadIndex(is);
    indices.resize((size_t)size);
    for (index_t i = 0; i < size; ++i) {
        indices[(size_t)i] = LoadIndex(is);
    }

    return indices;
}



template <typename T = float>
struct TrainData
{
    indices_t                       x_shape;
    indices_t                       t_shape;
    std::vector< std::vector<T> >   x_train;
    std::vector< std::vector<T> >   t_train;
    std::vector< std::vector<T> >   x_test;
    std::vector< std::vector<T> >   t_test;

    void clear(void) {
        x_shape.clear();
        t_shape.clear();
        x_train.clear();
        t_train.clear();
        x_test.clear();
        t_test.clear();
    }

    bool empty(void) {
        return x_train.empty() || t_train.empty() || x_test.empty() || t_test.empty();
    }
};



class Bit;
class Binary;
class Sign;


// メモリ効率とSIMDでの並列演算を意図してメモリ上で1bitづつパッキングして配置する事を意図した型
class Bit
{
protected:
    bool    m_value;

public:
    Bit() {}
    template<typename Tp>
    Bit(Tp v) { m_value = (v > 0); }
    Bit(const Bit& bit) { m_value = bit.m_value; }
    Bit(bool v) { m_value = v; }
    inline Bit(const Binary& sign);
    inline Bit(const Sign& sign);

    template<typename Tp>
    Bit& operator=(const Tp& v)    { m_value = (v > 0); return *this; }
    Bit& operator=(const Bit& bit) { m_value = bit.m_value; return *this; }
    Bit& operator=(const bool& v)  { m_value = v; return *this; }
    inline Bit& operator=(const Binary& bin);
    inline Bit& operator=(const Sign& sign);

    bool operator==(Bit const &bit) { return (m_value == bit.m_value); }
    bool operator>(Bit const &bit) { return ((int)m_value > (int)bit.m_value); }
    bool operator>=(Bit const &bit) { return ((int)m_value >= (int)bit.m_value); }
    bool operator<(Bit const &bit) { return ((int)m_value < (int)bit.m_value); }
    bool operator<=(Bit const &bit) { return ((int)m_value <= (int)bit.m_value); }

    template<typename Tp>
    operator Tp() { return m_value ? (Tp)1.0 : (Tp)0.0; }
    operator bool() const { return m_value; }
};


// SSE命令のマスクを意図して、メモリ上で8bitで、0x00 or 0xff の型を定義
class Binary
{
protected:
    std::uint8_t    m_value;

public:
    Binary() {}
    template<typename Tp>
    Binary(Tp v) { m_value = (v > 0) ? 0xff : 0x00; }
    Binary(const Binary& bin) { m_value = bin.m_value; }
    Binary(bool v) { m_value = v ? 0xff : 0x00; }
    inline Binary(const Bit& bit);
    inline Binary(const Sign& sign);

    template<typename Tp>
    Binary& operator=(const Tp& v) { m_value = (v > 0) ? 0xff : 0x00; return *this; }
    Binary& operator=(const Binary& bin) { m_value = bin.m_value; return *this; }
    Binary& operator=(const bool& v) { m_value = v; return *this; }
    inline Binary& operator=(const Bit& bit);
    inline Binary& operator=(const Sign& sign);

    template<typename Tp>
    operator Tp() { return m_value ? (Tp)1.0 : (Tp)0.0; }
    operator bool() const { return (m_value != 0); }
};


// Sign 数値にキャストしたときに -1 or +1 となる bool 型的なものを定義
class Sign
{
protected:
    bool    m_value;

public:
    Sign() {}
    template<typename Tp>
    Sign(Tp v) { m_value = (v > 0); }
    Sign(const Sign& sign) { m_value = sign.m_value; }
    Sign(bool v) { m_value = v; }
    inline Sign(const Bit& bit);
    inline Sign(const Binary& bin);

    template<typename Tp>
    Sign& operator=(const Tp& v) { m_value = (v > 0); return *this; }
    Sign& operator=(const Sign& sign) { m_value = sign.m_value; return *this; }
    Sign& operator=(const bool& v) { m_value = v; return *this; }
    inline Sign& operator=(const Bit& bit);
    inline Sign& operator=(const Binary& bin);

    template<typename Tp>
    operator Tp() { return m_value ? (Tp)+1.0 : (Tp)-1.0; }
    operator bool() const { return m_value; }
};

inline Bit::Bit(const Binary& bin) { m_value = (bool)bin; }
inline Bit& Bit::operator=(const Binary& bin) { m_value = (bool)bin; return *this; }
inline Bit::Bit(const Sign& sign) { m_value = (bool)sign; }
inline Bit& Bit::operator=(const Sign& sign) { m_value = (bool)sign; return *this; }

inline Binary::Binary(const Bit& bit) { m_value = (bool)bit; }
inline Binary& Binary::operator=(const Bit& bit) { m_value = (bool)bit; return *this; }
inline Binary::Binary(const Sign& sign) { m_value = (bool)sign; }
inline Binary& Binary::operator=(const Sign& sign) { m_value = (bool)sign; return *this; }

inline Sign::Sign(const Bit& bit) { m_value = (bool)bit; }
inline Sign& Sign::operator=(const Bit& bit) { m_value = (bool)bit; return *this; }
inline Sign::Sign(const Binary& bin) { m_value = (bool)bin; }
inline Sign& Sign::operator=(const Binary& bin) { m_value = (bool)bin; return *this; }



// データタイプ定義
template<typename _Tp> class DataType
{
public:
    typedef _Tp value_type;
    enum {
        type = 0,
        bit_size = 0
    };
    static inline std::string Name(void) { return "Unknown"; }
};

/*
template<> class DataType<bool>
{
public:
    typedef bool value_type;
    enum {
        type = BB_TYPE_BOOL,
        bit_size = 1,
    };
};
*/

template<> class DataType<Bit>
{
public:
    typedef bool value_type;
    enum {
        type = BB_TYPE_BIT,
        size = 1,
        bit_size = 1,
    };
    static inline std::string Name(void) { return "bit"; }
};

template<> class DataType<Binary>
{
public:
    typedef bool value_type;
    enum {
        type = BB_TYPE_BINARY,
        size = 1,
        bit_size = 8,
    };
    static inline std::string Name(void) { return "bin"; }
};

template<> class DataType<std::int8_t>
{
public:
    typedef std::int8_t value_type;
    enum {
        type = BB_TYPE_INT8,
        size = 1,
        bit_size = 8,
    };
    static inline std::string Name(void) { return "int8"; }
};

template<> class DataType<std::int16_t>
{
public:
    typedef std::int16_t value_type;
    enum {
        type = BB_TYPE_INT16,
        size = 2,
        bit_size = 16,
    };
    static inline std::string Name(void) { return "int16"; }
};

template<> class DataType<std::int32_t>
{
public:
    typedef std::int32_t value_type;
    enum {
        type = BB_TYPE_INT32,
        size = 4,
        bit_size = 32,
    };
    static inline std::string Name(void) { return "int32"; }
};

template<> class DataType<std::int64_t>
{
public:
    typedef std::int64_t value_type;
    enum {
        type = BB_TYPE_INT64,
        size = 8,
        bit_size = 64,
    };
    static inline std::string Name(void) { return "int64"; }
};

template<> class DataType<std::uint8_t>
{
public:
    typedef std::uint8_t value_type;
    enum {
        type = BB_TYPE_UINT8,
        size = 1,
        bit_size = 8,
    };
    static inline std::string Name(void) { return "uint8"; }
};

template<> class DataType<std::uint16_t>
{
public:
    typedef std::uint16_t value_type;
    enum {
        type = BB_TYPE_UINT16,
        size = 2,
        bit_size = 16,
    };
    static inline std::string Name(void) { return "uint16"; }
};

template<> class DataType<std::uint32_t>
{
public:
    typedef std::uint32_t value_type;
    enum {
        type = BB_TYPE_UINT32,
        size = 4,
        bit_size = 32,
    };
    static inline std::string Name(void) { return "uint32"; }
};

template<> class DataType<std::uint64_t>
{
public:
    typedef std::uint64_t value_type;
    enum {
        type = BB_TYPE_UINT64,
        size = 8,
        bit_size = 64,
    };
    static inline std::string Name(void) { return "uint64"; }
};

template<> class DataType<float>
{
public:
    typedef float value_type;
    enum {
        type = BB_TYPE_FP32,
        size = 4,
        bit_size = 32,
    };
    static inline std::string Name(void) { return "fp32"; }
};

template<> class DataType<double>
{
public:
    typedef double value_type;
    enum {
        type = BB_TYPE_FP64,
        size = 8,
        bit_size = 64,
    };
    static inline std::string Name(void) { return "fp64"; }
};


inline int DataType_GetBitSize(int type)
{
    switch (type) {
    case BB_TYPE_BIT:    return 1;
    case BB_TYPE_BINARY: return 8;
    case BB_TYPE_FP16:   return 16;
    case BB_TYPE_FP32:   return 32;
    case BB_TYPE_FP64:   return 64;
    case BB_TYPE_INT8:   return 8;
    case BB_TYPE_INT16:  return 16;
    case BB_TYPE_INT32:  return 32;
    case BB_TYPE_INT64:  return 64;
    case BB_TYPE_UINT8:  return 8;
    case BB_TYPE_UINT16: return 16;
    case BB_TYPE_UINT32: return 32;
    case BB_TYPE_UINT64: return 64;
    }

    return 0;
}


inline int DataType_GetByteSize(int type)
{
    switch (type) {
    case BB_TYPE_BIT:    return 1;
    case BB_TYPE_BINARY: return 1;
    case BB_TYPE_FP16:   return 2;
    case BB_TYPE_FP32:   return 4;
    case BB_TYPE_FP64:   return 8;
    case BB_TYPE_INT8:   return 1;
    case BB_TYPE_INT16:  return 2;
    case BB_TYPE_INT32:  return 4;
    case BB_TYPE_INT64:  return 8;
    case BB_TYPE_UINT8:  return 1;
    case BB_TYPE_UINT16: return 2;
    case BB_TYPE_UINT32: return 4;
    case BB_TYPE_UINT64: return 8;
    }

    return 0;
}



// アクセサ
template<typename Tp>
inline Tp DataType_Read(const void *base, index_t index)
{
    const Tp* ptr = (Tp*)base;
    return ptr[index];
}

template<>
inline Bit DataType_Read<Bit>(const void* base, index_t index)
{
    const std::uint8_t* ptr = (std::uint8_t*)base;
    std::uint8_t mask = (std::uint8_t)(1 << (index % 8));
    return ((ptr[index / 8] & mask) != 0);
}

template<>
inline Bit const DataType_Read<Bit const>(const void* base, index_t index)
{
    const std::uint8_t* ptr = (std::uint8_t*)base;
    std::uint8_t mask = (std::uint8_t)(1 << (index % 8));
    return ((ptr[index / 8] & mask) != 0);
}


template<typename Tp>
inline void DataType_Write(void* base, index_t index, Tp value)
{
    Tp* ptr = (Tp*)base;
    ptr[index] = value;
}

template<>
inline void DataType_Write<Bit>(void* base, index_t index, Bit value)
{
    std::uint8_t* ptr = (std::uint8_t*)base;
    std::uint8_t mask = (std::uint8_t)(1 << (index % 8));
    if (value) {
        ptr[index / 8] |= mask;
    }
    else {
        ptr[index / 8] &= ~mask;
    }
}


template<typename Tp>
inline void DataType_Add(void* base, index_t index, Tp value)
{
    Tp* ptr = (Tp*)base;
    ptr[index] += value;
}

template<>
inline void DataType_Add<Bit>(void* base, index_t index, Bit value)
{
    if (value) {
        std::uint8_t* ptr = (std::uint8_t*)base;
        std::uint8_t mask = (std::uint8_t)(1 << (index % 8));
        
        ptr[index / 8] |= mask;
    }
}



// シリアライズ用
template<typename T>
inline void SaveValue(std::ostream &os, T const &val)
{
    os.write((char const *)&val, sizeof(T));
}

template<typename T>
inline void LoadValue(std::istream &is, T &val)
{
    is.read((char *)&val, sizeof(T));
}


template<typename T>
inline void SaveValue(std::ostream &os, std::vector<T> const &vec)
{
    std::uint64_t size = (std::uint64_t)vec.size();
    os.write((char const *)&size, sizeof(size));
    if ( size > 0 ) {
        os.write((char const *)&vec[0], size*sizeof(T));
    }
}

template<typename T>
inline void LoadValue(std::istream &is, std::vector<T>  &vec)
{
    std::uint64_t size;
    is.read((char *)&size, sizeof(size));
    vec.resize(size);
    if ( size > 0 ) {
        is.read((char *)&vec[0], size*sizeof(T));
    }
}


template<>
inline void SaveValue(std::ostream &os, std::string const &str)
{
    std::uint64_t size = (std::uint64_t)str.size();
    os.write((char const *)&size, sizeof(size));
    if ( size > 0 ) {
        os.write((char const *)&str[0], size*sizeof(str[0]));
    }
}

template<>
inline void LoadValue(std::istream &is, std::string &str)
{
    std::uint64_t size;
    is.read((char *)&size, sizeof(size));
    str.resize(size);
    if ( size > 0 ) {
        is.read((char *)&str[0], size*sizeof(str[0]));
    }
}


template<>
inline void SaveValue(std::ostream &os, bool const &val)
{
    std::int8_t val8 = (std::int8_t)val;
    os.write((const char*)&val8, sizeof(val8));
}

template<>
inline void LoadValue(std::istream &is, bool &val)
{
    std::int8_t val8;
    is.read((char*)&val8, sizeof(val8));
    val = val8;
}


}

// end of file
