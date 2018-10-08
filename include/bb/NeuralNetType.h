// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <assert.h>

namespace bb {


// 現在、実数とバイナリを実装中。将来は整数も実装するかも
// 実数は float と double が切り替えできる程度にはテンプレートで
// 書いたつもりだが、double の需要は無いと思われる
// 将来整数をやる場合はいろいろ必要と思う


#define BB_TYPE_BOOL		(0x0000 + 1)
#define BB_TYPE_BINARY		(0x0000 + 1)

//#define BB_TYPE_REAL16	(0x0100 + 16)
#define BB_TYPE_REAL32		(0x0100 + 32)
#define BB_TYPE_REAL64		(0x0100 + 64)

#define BB_TYPE_INT8		(0x0200 + 8)
#define BB_TYPE_INT16		(0x0200 + 16)
#define BB_TYPE_INT32		(0x0200 + 32)
#define BB_TYPE_INT64		(0x0200 + 64)



#ifdef _DEBUG
#define BB_ASSERT(v)		assert(v)
#else
#define BB_ASSERT(v)		do{}while(0)
#endif



class Bit;
class Binary;
class Sign;


// メモリ効率とSIMDでの並列演算を意図してメモリ上で1bitづつパッキングして配置する事を意図した型
class Bit
{
protected:
	bool	m_value;

public:
	Bit() {}
	template<typename Tp>
	Bit(Tp v) { m_value = (v > 0); }
	Bit(const Bit& bit) { m_value = bit.m_value; }
	Bit(bool v) { m_value = v; }
	inline Bit(const Binary& sign);
	inline Bit(const Sign& sign);

	template<typename Tp>
	Bit& operator=(const Tp& v)    { m_value = (v > 0) ? 0xff : 0x00; return *this; }
	Bit& operator=(const Bit& bit) { m_value = bit.m_value; return *this; }
	Bit& operator=(const bool& v)  { m_value = v; return *this; }
	inline Bit& operator=(const Binary& sign);
	inline Bit& operator=(const Sign& sign);

	template<typename Tp>
	operator Tp() { return m_value ? (Tp)1.0 : (Tp)0.0; }
	operator bool() const { return m_value; }
};


// SSE命令のマスクを意図して、メモリ上で8bitで、0x00 or 0xff の型を定義
class Binary
{
protected:
	std::uint8_t	m_value;

public:
	Binary() {}
	template<typename Tp>
	Binary(Tp v) { m_value = (v > 0) ? 0xff : 0x00; }
	Binary(const Binary& bin) {	m_value = bin.m_value; }
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


// Sign	数値にキャストしたときに -1 or +1 となる bool 型的なものを定義
class Sign
{
protected:
	bool	m_value;

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



template<typename _Tp> class NeuralNetType
{
public:
	typedef _Tp value_type;
	enum {
		type = 0,
		bit_size = 0
	};
};


template<> class NeuralNetType<bool>
{
public:
	typedef float value_type;
	enum {
		type = BB_TYPE_BOOL,
		bit_size = 1
	};
};

template<> class NeuralNetType<Bit>
{
public:
	typedef float value_type;
	enum {
		type = BB_TYPE_BOOL,
		bit_size = 1
	};
};

template<> class NeuralNetType<Binary>
{
public:
	typedef float value_type;
	enum {
		type = BB_TYPE_BINARY,
		bit_size = 8
	};
};


template<> class NeuralNetType<float>
{
public:
	typedef float value_type;
	enum {
		type = BB_TYPE_REAL32,
		bit_size = 32
	};
};

template<> class NeuralNetType<double>
{
public:
	typedef float value_type;
	enum {
		type = BB_TYPE_REAL64,
		bit_size = 32
	};
};


inline int NeuralNet_GetTypeBitSize(int type)
{
	switch (type) {
	case BB_TYPE_BINARY: return 1;
	case BB_TYPE_REAL32: return 32;
	case BB_TYPE_REAL64: return 64;
	}

	return 0;
}


template<typename Tp>
inline void NeuralNet_Write(void* base, size_t index, Tp value)
{
	Tp* ptr = (Tp*)base;
	ptr[index] = value;
}

template<>
inline void NeuralNet_Write<bool>(void* base, size_t index, bool value)
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

template<>
inline void NeuralNet_Write<Bit>(void* base, size_t index, Bit value)
{
	NeuralNet_Write<bool>(base, index, (bool)value);
}



template<typename Tp>
inline Tp NeuralNet_Read(const void *base, size_t index)
{
	const Tp* ptr = (Tp*)base;
	return ptr[index];
}

template <>
inline bool NeuralNet_Read<bool>(const void *base, size_t index)
{
	const std::uint8_t* ptr = (std::uint8_t*)base;
	std::uint8_t mask = (std::uint8_t)(1 << (index % 8));
	return ((ptr[index / 8] & mask) != 0);
}


template<>
inline Bit NeuralNet_Read<Bit>(const void* base, size_t index)
{
	return (Bit)NeuralNet_Read<bool>(base, index);
}


}

