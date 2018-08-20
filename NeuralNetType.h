// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

namespace bb {

#define NN_TYPE_BINARY		(1)
#define NN_TYPE_REAL32		(32)


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
		type = NN_TYPE_BINARY,
		bit_size = 32
	};
};

template<> class NeuralNetType<float>
{
public:
	typedef float value_type;
	enum {
		type = NN_TYPE_REAL32,
		bit_size = 32
	};
};


inline int NeuralNet_GetTypeBitSize(int type)
{
	switch (type) {
	case NN_TYPE_BINARY: return 1;
	case NN_TYPE_REAL32: return 32;
	}

	return 0;
}


}

