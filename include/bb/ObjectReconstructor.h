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

#include "bb/DataType.h"

#include "bb/Object.h"

#include "bb/Tensor.h" 
#include "bb/FrameBuffer.h"

#include "bb/Sequential.h"

#include "bb/BinaryModulation.h"
#include "bb/RealToBinary.h" 
#include "bb/BinaryToReal.h" 
#include "bb/BitEncode.h"
#include "bb/Reduce.h"

#include "bb/BinaryLutN.h"
#include "bb/StochasticLutN.h"
#include "bb/DifferentiableLutN.h"
#include "bb/DifferentiableLutDiscreteN.h"
#include "bb/MicroMlpAffine.h"
#include "bb/MicroMlp.h"

#include "bb/DenseAffine.h"
#include "bb/DepthwiseDenseAffine.h"

#include "bb/Convolution2d.h"
#include "bb/ConvolutionCol2Im.h"
#include "bb/ConvolutionIm2Col.h"

#include "bb/MaxPooling.h"
#include "bb/StochasticMaxPooling2x2.h"
#include "bb/UpSampling.h"

#include "bb/Binarize.h"
#include "bb/Sigmoid.h"
#include "bb/ReLU.h"
#include "bb/HardTanh.h"

#include "bb/BatchNormalization.h" 
#include "bb/Dropout.h"
#include "bb/Shuffle.h"

#include "bb/OptimizerSgd.h"
#include "bb/OptimizerAdaGrad.h"
#include "bb/OptimizerAdam.h"

#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/LossMeanSquaredError.h"

#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/MetricsBinaryAccuracy.h"
#include "bb/MetricsMeanSquaredError.h"



namespace bb {


// ファイルからのオブジェクトの再構築
// なお、再構築サポートするとすべてのオブジェクトのコードをリンクすることになるので
// BB_OBJECT_RECONSTRUCTION を定義したときのみ有効になる



// 再構築の対象にしたいものはひとまず手動で足すこととする

#define BB_OBJECT_NEW(...) \
    do { \
        if ( object_name == __VA_ARGS__::ObjectName() ) { return std::shared_ptr<__VA_ARGS__>(new __VA_ARGS__); } \
    } while(0)

#define BB_OBJECT_CREATE(...) \
    do { \
        if ( object_name == __VA_ARGS__::ObjectName() ) { return __VA_ARGS__::Create(); } \
    } while(0)


inline std::shared_ptr<Object> Object_Creator(std::string object_name)
{
    BB_OBJECT_NEW(Tensor);
    BB_OBJECT_NEW(Tensor_<float>);
    BB_OBJECT_NEW(Tensor_<double>);
    BB_OBJECT_NEW(Tensor_<std::int8_t>);
    BB_OBJECT_NEW(Tensor_<std::int16_t>);
    BB_OBJECT_NEW(Tensor_<std::int32_t>);
    BB_OBJECT_NEW(Tensor_<std::int64_t>);
    BB_OBJECT_NEW(Tensor_<std::uint8_t>);
    BB_OBJECT_NEW(Tensor_<std::uint16_t>);
    BB_OBJECT_NEW(Tensor_<std::uint32_t>);
    BB_OBJECT_NEW(Tensor_<std::uint64_t>);
    
    BB_OBJECT_NEW(FrameBuffer);
    

    BB_OBJECT_CREATE(Sequential);

    BB_OBJECT_CREATE(BinaryModulation<float, float>);
    BB_OBJECT_CREATE(BinaryModulation<bb::Bit, float>);
    BB_OBJECT_CREATE(RealToBinary<float, float>);
    BB_OBJECT_CREATE(RealToBinary<bb::Bit, float>);
    BB_OBJECT_CREATE(BinaryToReal<float, float>);
    BB_OBJECT_CREATE(BinaryToReal<bb::Bit, float>);
    BB_OBJECT_CREATE(BitEncode<float, float>);
    BB_OBJECT_CREATE(BitEncode<bb::Bit, float>);
    BB_OBJECT_CREATE(Reduce<float, float>);
    BB_OBJECT_CREATE(Reduce<bb::Bit, float>);
    
    BB_OBJECT_CREATE(BinaryLutN<6, float, float>);
    BB_OBJECT_CREATE(BinaryLutN<6, bb::Bit, float>);
    BB_OBJECT_CREATE(BinaryLutN<5, float, float>);
    BB_OBJECT_CREATE(BinaryLutN<5, bb::Bit, float>);
    BB_OBJECT_CREATE(BinaryLutN<4, float, float>);
    BB_OBJECT_CREATE(BinaryLutN<4, bb::Bit, float>);
    BB_OBJECT_CREATE(BinaryLutN<3, float, float>);
    BB_OBJECT_CREATE(BinaryLutN<3, bb::Bit, float>);
    BB_OBJECT_CREATE(BinaryLutN<2, float, float>);
    BB_OBJECT_CREATE(BinaryLutN<2, bb::Bit, float>);

    BB_OBJECT_CREATE(StochasticLutN<6, float, float>);
    BB_OBJECT_CREATE(StochasticLutN<6, bb::Bit, float>);
    BB_OBJECT_CREATE(StochasticLutN<5, float, float>);
    BB_OBJECT_CREATE(StochasticLutN<5, bb::Bit, float>);
    BB_OBJECT_CREATE(StochasticLutN<4, float, float>);
    BB_OBJECT_CREATE(StochasticLutN<4, bb::Bit, float>);
    BB_OBJECT_CREATE(StochasticLutN<3, float, float>);
    BB_OBJECT_CREATE(StochasticLutN<3, bb::Bit, float>);
    BB_OBJECT_CREATE(StochasticLutN<2, float, float>);
    BB_OBJECT_CREATE(StochasticLutN<2, bb::Bit, float>);

    BB_OBJECT_CREATE(DifferentiableLutN<6, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutN<6, bb::Bit, float>);
    BB_OBJECT_CREATE(DifferentiableLutN<5, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutN<5, bb::Bit, float>);
    BB_OBJECT_CREATE(DifferentiableLutN<4, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutN<4, bb::Bit, float>);
    BB_OBJECT_CREATE(DifferentiableLutN<3, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutN<3, bb::Bit, float>);
    BB_OBJECT_CREATE(DifferentiableLutN<2, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutN<2, bb::Bit, float>);
    
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<6, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<6, bb::Bit, float>);
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<5, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<5, bb::Bit, float>);
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<4, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<4, bb::Bit, float>);
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<3, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<3, bb::Bit, float>);
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<2, float, float>);
    BB_OBJECT_CREATE(DifferentiableLutDiscreteN<2, bb::Bit, float>);
    
    BB_OBJECT_CREATE(MicroMlpAffine<6, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlpAffine<6, 16, bb::Bit, float>);
    BB_OBJECT_CREATE(MicroMlpAffine<5, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlpAffine<5, 16, bb::Bit, float>);
    BB_OBJECT_CREATE(MicroMlpAffine<4, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlpAffine<4, 16, bb::Bit, float>);
    BB_OBJECT_CREATE(MicroMlpAffine<3, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlpAffine<3, 16, bb::Bit, float>);
    BB_OBJECT_CREATE(MicroMlpAffine<2, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlpAffine<2, 16, bb::Bit, float>);

    BB_OBJECT_CREATE(MicroMlp<6, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlp<6, 16, bb::Bit, float>);
    BB_OBJECT_CREATE(MicroMlp<5, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlp<5, 16, bb::Bit, float>);
    BB_OBJECT_CREATE(MicroMlp<4, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlp<4, 16, bb::Bit, float>);
    BB_OBJECT_CREATE(MicroMlp<3, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlp<3, 16, bb::Bit, float>);
    BB_OBJECT_CREATE(MicroMlp<2, 16, float, float>);
    BB_OBJECT_CREATE(MicroMlp<2, 16, bb::Bit, float>);

    BB_OBJECT_CREATE(DenseAffine<float>);
    BB_OBJECT_CREATE(DepthwiseDenseAffine<float>);

    BB_OBJECT_CREATE(Convolution2d<float, float>);
    BB_OBJECT_CREATE(Convolution2d<bb::Bit, float>);
    BB_OBJECT_CREATE(ConvolutionCol2Im<float, float>);
    BB_OBJECT_CREATE(ConvolutionCol2Im<bb::Bit, float>);
    BB_OBJECT_CREATE(ConvolutionIm2Col<float, float>);
    BB_OBJECT_CREATE(ConvolutionIm2Col<bb::Bit, float>);
    
    BB_OBJECT_CREATE(MaxPooling<float, float>);
    BB_OBJECT_CREATE(MaxPooling<bb::Bit, float>);
    BB_OBJECT_CREATE(StochasticMaxPooling2x2<float, float>);
    BB_OBJECT_CREATE(StochasticMaxPooling2x2<Bit, float>);
    BB_OBJECT_CREATE(UpSampling<float, float>);
    BB_OBJECT_CREATE(UpSampling<bb::Bit, float>);

    BB_OBJECT_CREATE(Binarize<bb::Bit, float>);
    BB_OBJECT_CREATE(Binarize<float, float>);
    BB_OBJECT_CREATE(Sigmoid<bb::Bit, float>);
    BB_OBJECT_CREATE(Sigmoid<float, float>);
    BB_OBJECT_CREATE(ReLU<bb::Bit, float>);
    BB_OBJECT_CREATE(ReLU<float, float>);
    BB_OBJECT_CREATE(HardTanh<bb::Bit, float>);
    BB_OBJECT_CREATE(HardTanh<float, float>);

    BB_OBJECT_CREATE(BatchNormalization<float>);

    BB_OBJECT_CREATE(Dropout<float, float>);
    BB_OBJECT_CREATE(Dropout<bb::Bit, float>);
    
    BB_OBJECT_CREATE(Shuffle);


    BB_OBJECT_CREATE(OptimizerSgd<float>);
    BB_OBJECT_CREATE(OptimizerAdaGrad<float>);
    BB_OBJECT_CREATE(OptimizerAdam<float>);

    BB_OBJECT_CREATE(LossSoftmaxCrossEntropy<float>);
    BB_OBJECT_CREATE(LossMeanSquaredError<float>);

    BB_OBJECT_CREATE(MetricsCategoricalAccuracy<float>);
    BB_OBJECT_CREATE(MetricsBinaryAccuracy<float>);
    BB_OBJECT_CREATE(MetricsMeanSquaredError<float>);


    return std::shared_ptr<Object>();
}


// ストリームからオブジェクトを再構成
inline std::shared_ptr<Object> Object_Reconstruct(std::istream &is)
{
    auto object_name = Object::ReadHeader(is);
    BB_ASSERT(object_name.size() > 0);

    auto obj_ptr = Object_Creator(object_name);

    if ( obj_ptr ) {
        obj_ptr->LoadObjectData(is);
    }

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
