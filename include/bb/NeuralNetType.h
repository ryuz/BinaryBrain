// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

namespace bb {


// 現在、実数とバイナリを実装中。将来は整数も実装するかも
// 実数は float と double が切り替えできる程度にはテンプレートで
// 書いたつもりだが、double の需要は無いと思われる
// 将来整数をやる場合はいろいろ必要と思う


#define BB_TYPE_BINARY		(1)
#define BB_TYPE_REAL32		(32)
#define BB_TYPE_REAL64		(64)


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
		type = BB_TYPE_BINARY,
		bit_size = 32
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


}

