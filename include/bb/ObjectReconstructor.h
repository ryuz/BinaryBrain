// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2021 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#ifdef BB_OBJECT_RECONSTRUCTION


#include <iostream>
#include <memory>
#include <string>
#include <array>
#include <map>


#include "bb/Object.h"

#include "bb/Tensor.h" 

#include "bb/Sequential.h"

#include "bb/RealToBinary.h" 
#include "bb/BinaryToReal.h" 
#include "bb/BitEncode.h"

#include "bb/BinaryLutN.h"
#include "bb/DifferentiableLutN.h"
#include "bb/MicroMlpAffine.h"
#include "bb/MicroMlp.h"

#include "bb/DenseAffine.h"
#include "bb/DepthwiseDenseAffine.h"

#include "bb/Convolution2d.h"
#include "bb/ConvolutionCol2Im.h"
#include "bb/ConvolutionIm2Col.h"

#include "bb/MaxPooling.h"
#include "bb/StochasticMaxPooling2x2.h"

#include "bb/Binarize.h"
#include "bb/Sigmoid.h"
#include "bb/ReLU.h"
#include "bb/HardTanh.h"


#include "bb/BatchNormalization.h" 

#include "bb/Shuffle.h"


namespace bb {


// ファイルからのオブジェクトの再構築
// なお、再構築サポートするとすべてのオブジェクトのコードをリンクすることになるので
// BB_OBJECT_RECONSTRUCTION を定義したときのみ有効になる



// 再構築の対象にしたいものはひとまず手動で足すこととする

#define BB_OBJECT_CREATE(...) \
    do { \
        if ( object_name == __VA_ARGS__::ObjectName() ) { return std::shared_ptr<__VA_ARGS__>(new __VA_ARGS__); } \
    } while(0)

#define BB_OBJECT_CREATE_MODEL(...) \
    do { \
        if ( object_name == __VA_ARGS__::ObjectName() ) { return __VA_ARGS__::Create(); } \
    } while(0)


inline std::shared_ptr<Object> Object_Creator(std::string object_name)
{
    BB_OBJECT_CREATE(Tensor);
    BB_OBJECT_CREATE(Tensor_<float>);
    BB_OBJECT_CREATE(Tensor_<double>);
    BB_OBJECT_CREATE(Tensor_<std::int8_t>);
    BB_OBJECT_CREATE(Tensor_<std::int16_t>);
    BB_OBJECT_CREATE(Tensor_<std::int32_t>);
    BB_OBJECT_CREATE(Tensor_<std::int64_t>);
    BB_OBJECT_CREATE(Tensor_<std::uint8_t>);
    BB_OBJECT_CREATE(Tensor_<std::uint16_t>);
    BB_OBJECT_CREATE(Tensor_<std::uint32_t>);
    BB_OBJECT_CREATE(Tensor_<std::uint64_t>);

    BB_OBJECT_CREATE_MODEL(Sequential);

    BB_OBJECT_CREATE_MODEL(RealToBinary<float, float>);
    BB_OBJECT_CREATE_MODEL(RealToBinary<Bit, float>);
    BB_OBJECT_CREATE_MODEL(BinaryToReal<float, float>);
    BB_OBJECT_CREATE_MODEL(BinaryToReal<Bit, float>);
    BB_OBJECT_CREATE_MODEL(BitEncode<float, float>);
    BB_OBJECT_CREATE_MODEL(BitEncode<Bit, float>);

    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<6, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<6, Bit, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<5, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<5, Bit, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<4, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<4, Bit, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<3, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<3, Bit, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<2, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<2, Bit, float>);
    
    BB_OBJECT_CREATE_MODEL(BinaryLutN<6, float, float>);
    BB_OBJECT_CREATE_MODEL(BinaryLutN<6, Bit, float>);
    BB_OBJECT_CREATE_MODEL(BinaryLutN<5, float, float>);
    BB_OBJECT_CREATE_MODEL(BinaryLutN<5, Bit, float>);
    BB_OBJECT_CREATE_MODEL(BinaryLutN<4, float, float>);
    BB_OBJECT_CREATE_MODEL(BinaryLutN<4, Bit, float>);
    BB_OBJECT_CREATE_MODEL(BinaryLutN<3, float, float>);
    BB_OBJECT_CREATE_MODEL(BinaryLutN<3, Bit, float>);
    BB_OBJECT_CREATE_MODEL(BinaryLutN<2, float, float>);
    BB_OBJECT_CREATE_MODEL(BinaryLutN<2, Bit, float>);
    
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<6, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<6, 16, Bit, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<5, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<5, 16, Bit, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<4, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<4, 16, Bit, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<3, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<3, 16, Bit, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<2, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlpAffine<2, 16, Bit, float>);

    BB_OBJECT_CREATE_MODEL(MicroMlp<6, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlp<6, 16, Bit, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlp<5, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlp<5, 16, Bit, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlp<4, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlp<4, 16, Bit, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlp<3, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlp<3, 16, Bit, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlp<2, 16, float, float>);
    BB_OBJECT_CREATE_MODEL(MicroMlp<2, 16, Bit, float>);

    BB_OBJECT_CREATE_MODEL(DenseAffine<float>);
    BB_OBJECT_CREATE_MODEL(DepthwiseDenseAffine<float>);

    BB_OBJECT_CREATE_MODEL(Convolution2d<float, float>);
    BB_OBJECT_CREATE_MODEL(Convolution2d<Bit, float>);
    BB_OBJECT_CREATE_MODEL(ConvolutionCol2Im<float, float>);
    BB_OBJECT_CREATE_MODEL(ConvolutionCol2Im<Bit, float>);
    BB_OBJECT_CREATE_MODEL(ConvolutionIm2Col<float, float>);
    BB_OBJECT_CREATE_MODEL(ConvolutionIm2Col<Bit, float>);
    
    BB_OBJECT_CREATE_MODEL(MaxPooling<float, float>);
    BB_OBJECT_CREATE_MODEL(MaxPooling<Bit, float>);
    BB_OBJECT_CREATE_MODEL(StochasticMaxPooling2x2<float, float>);
    BB_OBJECT_CREATE_MODEL(StochasticMaxPooling2x2<Bit, float>);
    
    BB_OBJECT_CREATE_MODEL(Binarize<Bit, float>);
    BB_OBJECT_CREATE_MODEL(Binarize<float, float>);
    BB_OBJECT_CREATE_MODEL(Sigmoid<Bit, float>);
    BB_OBJECT_CREATE_MODEL(Sigmoid<float, float>);
    BB_OBJECT_CREATE_MODEL(ReLU<Bit, float>);
    BB_OBJECT_CREATE_MODEL(ReLU<float, float>);
    BB_OBJECT_CREATE_MODEL(HardTanh<Bit, float>);
    BB_OBJECT_CREATE_MODEL(HardTanh<float, float>);

    BB_OBJECT_CREATE_MODEL(BatchNormalization<float>);

    BB_OBJECT_CREATE_MODEL(Shuffle);

    return std::shared_ptr<Object>();
}


inline std::shared_ptr<Object> Object_Reconstruct(std::istream &is)
{
    auto object_name = Object::ReadHeader(is);
    BB_ASSERT(object_name.size() > 0);

    auto obj_ptr = Object_Creator(object_name);
    BB_ASSERT(obj_ptr);

    obj_ptr->LoadObjectData(is);

    return obj_ptr;
}

#ifdef BB_PYBIND11
inline pybind11::tuple Object_ReconstructPy(pybind11::bytes data)
{
    std::istringstream is((std::string)data, std::istringstream::binary);
    auto obj = Object_Reconstruct(is);
    return pybind11::make_tuple((std::size_t)is.tellg(), obj);
}

inline std::shared_ptr<Object> Object_CreatePy(pybind11::bytes data)
{
    std::istringstream is((std::string)data, std::istringstream::binary);
    return Object_Reconstruct(is);
}
#endif


}


#else   // BB_OBJECT_RECONSTRUCTION


#include "bb/Object.h"

namespace bb {

inline std::shared_ptr<Object> Object_Reconstruct(std::istream &is)
{
    return std::shared_ptr<Object>();
}

}


#endif   // BB_OBJECT_RECONSTRUCTION



// end of file
