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

#include "bb/NormalDistributionGenerator.h"
#include "bb/UniformDistributionGenerator.h"



namespace bb {


// ファイルからのオブジェクトの再構築
// なお、再構築サポートするとすべてのオブジェクトのコードをリンクすることになるので
// BB_OBJECT_RECONSTRUCTION を定義したときのみ有効になる



// 再構築の対象にしたいものはひとまず手動で足すこととする

#define BB_OBJECT_NEW(class_name) \
    do { \
        if ( object_name == class_name::ObjectName() ) { return std::shared_ptr<class_name>(new class_name); } \
    } while(0)

#define BB_OBJECT_CREATE(class_name) \
    do { \
        if ( object_name == class_name::ObjectName() ) { return class_name::Create(); } \
    } while(0)


inline std::shared_ptr<Object> Object_Creator(std::string object_name)
{
    // Tensor
    using Tensor        = Tensor;
    using Tensor_fp32   = Tensor_<float>;
    using Tensor_fp64   = Tensor_<double>;
    using Tensor_int8   = Tensor_<std::int8_t>;
    using Tensor_int16  = Tensor_<std::int16_t>;
    using Tensor_int32  = Tensor_<std::int32_t>;
    using Tensor_int64  = Tensor_<std::int64_t>;
    using Tensor_uint8  = Tensor_<std::uint8_t>;
    using Tensor_uint16 = Tensor_<std::uint16_t>;
    using Tensor_uint32 = Tensor_<std::uint32_t>;
    using Tensor_uint64 = Tensor_<std::uint64_t>;
    BB_OBJECT_NEW(Tensor);
    BB_OBJECT_NEW(Tensor_fp32);
    BB_OBJECT_NEW(Tensor_fp64);
    BB_OBJECT_NEW(Tensor_int8);
    BB_OBJECT_NEW(Tensor_int16);
    BB_OBJECT_NEW(Tensor_int32);
    BB_OBJECT_NEW(Tensor_int64);
    BB_OBJECT_NEW(Tensor_uint8);
    BB_OBJECT_NEW(Tensor_uint16);
    BB_OBJECT_NEW(Tensor_uint32);
    BB_OBJECT_NEW(Tensor_uint64);
    
    // FrameBuffer
    using FrameBuffer = bb::FrameBuffer;
    BB_OBJECT_NEW(FrameBuffer);
    
    // Variables
    using Variables = bb::Variables;
    BB_OBJECT_NEW(Variables);

    // Sequential
    using Sequential = bb::Sequential;
    BB_OBJECT_CREATE(Sequential);


    // Model
//    using BinaryModulation_fp32_fp32 = bb::BinaryModulation<float, float>;
//    using BinaryModulation_bit_fp32  = bb::BinaryModulation<bb::Bit, float>;
//    BB_OBJECT_CREATE(BinaryModulation_fp32_fp32);
//    BB_OBJECT_CREATE(BinaryModulation_bit_fp32);

    using RealToBinary_fp32_fp32 = bb::RealToBinary<float, float>;
    using RealToBinary_bit_fp32  = bb::RealToBinary<bb::Bit, float>;
    BB_OBJECT_CREATE(RealToBinary_fp32_fp32);
    BB_OBJECT_CREATE(RealToBinary_bit_fp32);

    using BinaryToReal_fp32_fp32 = bb::BinaryToReal<float, float>;
    using BinaryToReal_bit_fp32  = bb::BinaryToReal<bb::Bit, float>;
    BB_OBJECT_CREATE(BinaryToReal_fp32_fp32);
    BB_OBJECT_CREATE(BinaryToReal_bit_fp32);

    using BitEncode_fp32_fp32 = bb::BitEncode<float, float>;
    using BitEncode_bit_fp32  = bb::BitEncode<bb::Bit, float>;
    BB_OBJECT_CREATE(BitEncode_fp32_fp32);
    BB_OBJECT_CREATE(BitEncode_bit_fp32);

    using Reduce_fp32_fp32 = bb::Reduce<float, float>; 
    using Reduce_bit_fp32  = bb::Reduce<bb::Bit, float>; 
    BB_OBJECT_CREATE(Reduce_fp32_fp32);
    BB_OBJECT_CREATE(Reduce_bit_fp32);
    
    using BinaryLut6_fp32_fp32 = bb::BinaryLutN<6, float, float>;
    using BinaryLut5_fp32_fp32 = bb::BinaryLutN<5, float, float>;
    using BinaryLut4_fp32_fp32 = bb::BinaryLutN<4, float, float>;
    using BinaryLut3_fp32_fp32 = bb::BinaryLutN<3, float, float>;
    using BinaryLut2_fp32_fp32 = bb::BinaryLutN<2, float, float>;
    using BinaryLut6_bit_fp32  = bb::BinaryLutN<6, bb::Bit, float>;
    using BinaryLut5_bit_fp32  = bb::BinaryLutN<5, bb::Bit, float>;
    using BinaryLut4_bit_fp32  = bb::BinaryLutN<4, bb::Bit, float>;
    using BinaryLut3_bit_fp32  = bb::BinaryLutN<3, bb::Bit, float>;
    using BinaryLut2_bit_fp32  = bb::BinaryLutN<2, bb::Bit, float>;
    BB_OBJECT_CREATE(BinaryLut6_fp32_fp32);
    BB_OBJECT_CREATE(BinaryLut5_fp32_fp32);
    BB_OBJECT_CREATE(BinaryLut4_fp32_fp32);
    BB_OBJECT_CREATE(BinaryLut3_fp32_fp32);
    BB_OBJECT_CREATE(BinaryLut2_fp32_fp32);
    BB_OBJECT_CREATE(BinaryLut6_bit_fp32);
    BB_OBJECT_CREATE(BinaryLut5_bit_fp32);
    BB_OBJECT_CREATE(BinaryLut4_bit_fp32);
    BB_OBJECT_CREATE(BinaryLut3_bit_fp32);
    BB_OBJECT_CREATE(BinaryLut2_bit_fp32);

    using StochasticLut6_fp32_fp32 = bb::StochasticLutN<6, float, float>;
    using StochasticLut5_fp32_fp32 = bb::StochasticLutN<5, float, float>;
    using StochasticLut4_fp32_fp32 = bb::StochasticLutN<4, float, float>;
    using StochasticLut3_fp32_fp32 = bb::StochasticLutN<3, float, float>;
    using StochasticLut2_fp32_fp32 = bb::StochasticLutN<2, float, float>;
    using StochasticLut6_bit_fp32  = bb::StochasticLutN<6, bb::Bit, float>;
    using StochasticLut5_bit_fp32  = bb::StochasticLutN<5, bb::Bit, float>;
    using StochasticLut4_bit_fp32  = bb::StochasticLutN<4, bb::Bit, float>;
    using StochasticLut3_bit_fp32  = bb::StochasticLutN<3, bb::Bit, float>;
    using StochasticLut2_bit_fp32  = bb::StochasticLutN<2, bb::Bit, float>;
    BB_OBJECT_CREATE(StochasticLut6_fp32_fp32);
    BB_OBJECT_CREATE(StochasticLut5_fp32_fp32);
    BB_OBJECT_CREATE(StochasticLut4_fp32_fp32);
    BB_OBJECT_CREATE(StochasticLut3_fp32_fp32);
    BB_OBJECT_CREATE(StochasticLut2_fp32_fp32);
    BB_OBJECT_CREATE(StochasticLut6_bit_fp32);
    BB_OBJECT_CREATE(StochasticLut5_bit_fp32);
    BB_OBJECT_CREATE(StochasticLut4_bit_fp32);
    BB_OBJECT_CREATE(StochasticLut3_bit_fp32);
    BB_OBJECT_CREATE(StochasticLut2_bit_fp32);

    using DifferentiableLut6_fp32_fp32 = bb::DifferentiableLutN<6, float, float>;
    using DifferentiableLut5_fp32_fp32 = bb::DifferentiableLutN<5, float, float>;
    using DifferentiableLut4_fp32_fp32 = bb::DifferentiableLutN<4, float, float>;
    using DifferentiableLut3_fp32_fp32 = bb::DifferentiableLutN<3, float, float>;
    using DifferentiableLut2_fp32_fp32 = bb::DifferentiableLutN<2, float, float>;
    using DifferentiableLut6_bit_fp32  = bb::DifferentiableLutN<6, bb::Bit, float>;
    using DifferentiableLut5_bit_fp32  = bb::DifferentiableLutN<5, bb::Bit, float>;
    using DifferentiableLut4_bit_fp32  = bb::DifferentiableLutN<4, bb::Bit, float>;
    using DifferentiableLut3_bit_fp32  = bb::DifferentiableLutN<3, bb::Bit, float>;
    using DifferentiableLut2_bit_fp32  = bb::DifferentiableLutN<2, bb::Bit, float>;
    BB_OBJECT_CREATE(DifferentiableLut6_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLut5_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLut4_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLut3_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLut2_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLut6_bit_fp32);
    BB_OBJECT_CREATE(DifferentiableLut5_bit_fp32);
    BB_OBJECT_CREATE(DifferentiableLut4_bit_fp32);
    BB_OBJECT_CREATE(DifferentiableLut3_bit_fp32);
    BB_OBJECT_CREATE(DifferentiableLut2_bit_fp32);
    
    using DifferentiableLutDiscrete6_fp32_fp32 = bb::DifferentiableLutDiscreteN<6, float, float>;
    using DifferentiableLutDiscrete5_fp32_fp32 = bb::DifferentiableLutDiscreteN<5, float, float>;
    using DifferentiableLutDiscrete4_fp32_fp32 = bb::DifferentiableLutDiscreteN<4, float, float>;
    using DifferentiableLutDiscrete3_fp32_fp32 = bb::DifferentiableLutDiscreteN<3, float, float>;
    using DifferentiableLutDiscrete2_fp32_fp32 = bb::DifferentiableLutDiscreteN<2, float, float>;
    using DifferentiableLutDiscrete6_bit_fp32  = bb::DifferentiableLutDiscreteN<6, bb::Bit, float>;
    using DifferentiableLutDiscrete5_bit_fp32  = bb::DifferentiableLutDiscreteN<5, bb::Bit, float>;
    using DifferentiableLutDiscrete4_bit_fp32  = bb::DifferentiableLutDiscreteN<4, bb::Bit, float>;
    using DifferentiableLutDiscrete3_bit_fp32  = bb::DifferentiableLutDiscreteN<3, bb::Bit, float>;
    using DifferentiableLutDiscrete2_bit_fp32  = bb::DifferentiableLutDiscreteN<2, bb::Bit, float>;
    BB_OBJECT_CREATE(DifferentiableLutDiscrete6_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLutDiscrete6_bit_fp32);
    BB_OBJECT_CREATE(DifferentiableLutDiscrete5_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLutDiscrete5_bit_fp32);
    BB_OBJECT_CREATE(DifferentiableLutDiscrete4_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLutDiscrete4_bit_fp32);
    BB_OBJECT_CREATE(DifferentiableLutDiscrete3_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLutDiscrete3_bit_fp32);
    BB_OBJECT_CREATE(DifferentiableLutDiscrete2_fp32_fp32);
    BB_OBJECT_CREATE(DifferentiableLutDiscrete2_bit_fp32);

    using MicroMlpAffine6_16_fp32_fp32 = bb::MicroMlpAffine<6, 16, float, float>;
    using MicroMlpAffine5_16_fp32_fp32 = bb::MicroMlpAffine<5, 16, float, float>;
    using MicroMlpAffine4_16_fp32_fp32 = bb::MicroMlpAffine<4, 16, float, float>;
    using MicroMlpAffine3_16_fp32_fp32 = bb::MicroMlpAffine<3, 16, float, float>;
    using MicroMlpAffine2_16_fp32_fp32 = bb::MicroMlpAffine<2, 16, float, float>;
    using MicroMlpAffine6_16_bit_fp32  = bb::MicroMlpAffine<6, 16, bb::Bit, float>;
    using MicroMlpAffine5_16_bit_fp32  = bb::MicroMlpAffine<5, 16, bb::Bit, float>;
    using MicroMlpAffine4_16_bit_fp32  = bb::MicroMlpAffine<4, 16, bb::Bit, float>;
    using MicroMlpAffine3_16_bit_fp32  = bb::MicroMlpAffine<3, 16, bb::Bit, float>;
    using MicroMlpAffine2_16_bit_fp32  = bb::MicroMlpAffine<2, 16, bb::Bit, float>;
    BB_OBJECT_CREATE(MicroMlpAffine6_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlpAffine5_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlpAffine4_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlpAffine3_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlpAffine2_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlpAffine6_16_bit_fp32);
    BB_OBJECT_CREATE(MicroMlpAffine5_16_bit_fp32);
    BB_OBJECT_CREATE(MicroMlpAffine4_16_bit_fp32);
    BB_OBJECT_CREATE(MicroMlpAffine3_16_bit_fp32);
    BB_OBJECT_CREATE(MicroMlpAffine2_16_bit_fp32);

    using MicroMlp6_16_fp32_fp32 = bb::MicroMlp<6, 16, float, float>;
    using MicroMlp5_16_fp32_fp32 = bb::MicroMlp<5, 16, float, float>;
    using MicroMlp4_16_fp32_fp32 = bb::MicroMlp<4, 16, float, float>;
    using MicroMlp3_16_fp32_fp32 = bb::MicroMlp<3, 16, float, float>;
    using MicroMlp2_16_fp32_fp32 = bb::MicroMlp<2, 16, float, float>;
    using MicroMlp6_16_bit_fp32  = bb::MicroMlp<6, 16, bb::Bit, float>;
    using MicroMlp5_16_bit_fp32  = bb::MicroMlp<5, 16, bb::Bit, float>;
    using MicroMlp4_16_bit_fp32  = bb::MicroMlp<4, 16, bb::Bit, float>;
    using MicroMlp3_16_bit_fp32  = bb::MicroMlp<3, 16, bb::Bit, float>;
    using MicroMlp2_16_bit_fp32  = bb::MicroMlp<2, 16, bb::Bit, float>;
    BB_OBJECT_CREATE(MicroMlp6_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlp5_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlp4_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlp3_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlp2_16_fp32_fp32);
    BB_OBJECT_CREATE(MicroMlp6_16_bit_fp32);
    BB_OBJECT_CREATE(MicroMlp5_16_bit_fp32);
    BB_OBJECT_CREATE(MicroMlp4_16_bit_fp32);
    BB_OBJECT_CREATE(MicroMlp3_16_bit_fp32);
    BB_OBJECT_CREATE(MicroMlp2_16_bit_fp32);

    using DenseAffine_fp32 = bb::DenseAffine<float>;
    BB_OBJECT_CREATE(DenseAffine_fp32);
    
    using DepthwiseDenseAffine_fp32 = bb::DepthwiseDenseAffine<float>;
    BB_OBJECT_CREATE(DepthwiseDenseAffine_fp32);


    using ConvolutionIm2Col_fp32_fp32 = bb::ConvolutionIm2Col<float, float>;
    using ConvolutionIm2Col_bit_fp32  = bb::ConvolutionIm2Col<bb::Bit, float>;
    BB_OBJECT_CREATE(ConvolutionIm2Col_fp32_fp32);
    BB_OBJECT_CREATE(ConvolutionIm2Col_bit_fp32);

    using ConvolutionCol2Im_fp32_fp32 = bb::ConvolutionCol2Im<float, float>;
    using ConvolutionCol2Im_bit_fp32  = bb::ConvolutionCol2Im<bb::Bit, float>;
    BB_OBJECT_CREATE(ConvolutionCol2Im_fp32_fp32);
    BB_OBJECT_CREATE(ConvolutionCol2Im_bit_fp32);

    using Convolution2d_fp32_fp32 = bb::Convolution2d<float, float>;
    using Convolution2d_bit_fp32  = bb::Convolution2d<bb::Bit, float>;
    BB_OBJECT_CREATE(Convolution2d_fp32_fp32);
    BB_OBJECT_CREATE(Convolution2d_bit_fp32);

    using MaxPooling_fp32_fp32 = bb::MaxPooling<float, float>;
    using MaxPooling_bit_fp32  = bb::MaxPooling<bb::Bit, float>;
    BB_OBJECT_CREATE(MaxPooling_fp32_fp32);
    BB_OBJECT_CREATE(MaxPooling_bit_fp32);

//   using StochasticMaxPooling_fp32_fp32 = bb::StochasticMaxPooling<float, float>;
//   using StochasticMaxPooling_bit_fp32  = bb::StochasticMaxPooling<bb::Bit, float>;
//   BB_OBJECT_CREATE(StochasticMaxPooling_fp32_fp32);
//   BB_OBJECT_CREATE(StochasticMaxPooling_bit_fp32);

    using StochasticMaxPooling2x2_fp32_fp32 = bb::StochasticMaxPooling2x2<float, float>;
    using StochasticMaxPooling2x2_bit_fp32  = bb::StochasticMaxPooling2x2<bb::Bit, float>;
    BB_OBJECT_CREATE(StochasticMaxPooling2x2_fp32_fp32);
    BB_OBJECT_CREATE(StochasticMaxPooling2x2_bit_fp32);

    using UpSampling_fp32_fp32 = bb::UpSampling<float, float>;
    using UpSampling_bit_fp32  = bb::UpSampling<bb::Bit, float>;
    BB_OBJECT_CREATE(UpSampling_fp32_fp32);
    BB_OBJECT_CREATE(UpSampling_bit_fp32);


    using Binarize_fp32_fp32 = bb::Binarize<float, float>;
    using Binarize_bit_fp32  = bb::Binarize<bb::Bit, float>;
    BB_OBJECT_CREATE(Binarize_bit_fp32);
    BB_OBJECT_CREATE(Binarize_fp32_fp32);

    using Sigmoid_fp32_fp32 = bb::Sigmoid<float, float>;
    using Sigmoid_bit_fp32  = bb::Sigmoid<bb::Bit, float>;
    BB_OBJECT_CREATE(Sigmoid_bit_fp32);
    BB_OBJECT_CREATE(Sigmoid_fp32_fp32);

    using ReLU_fp32_fp32 = bb::ReLU<float, float>;
    using ReLU_bit_fp32  = bb::ReLU<bb::Bit, float>;
    BB_OBJECT_CREATE(ReLU_bit_fp32);
    BB_OBJECT_CREATE(ReLU_fp32_fp32);

    using HardTanh_fp32_fp32 = bb::HardTanh<float, float>;
    using HardTanh_bit_fp32  = bb::HardTanh<bb::Bit, float>;
    BB_OBJECT_CREATE(HardTanh_bit_fp32);
    BB_OBJECT_CREATE(HardTanh_fp32_fp32);

    using BatchNormalization_fp32 = bb::BatchNormalization<float>;
    BB_OBJECT_CREATE(BatchNormalization_fp32);

    using StochasticBatchNormalization_fp32 = bb::StochasticBatchNormalization<float>;
    BB_OBJECT_CREATE(StochasticBatchNormalization_fp32);

    using Dropout_fp32_fp32 = bb::Dropout<float, float>;
    using Dropout_bit_fp32  = bb::Dropout<bb::Bit, float>;
    BB_OBJECT_CREATE(Dropout_fp32_fp32);
    BB_OBJECT_CREATE(Dropout_bit_fp32);
    
    using Shuffle = bb::Shuffle;
    BB_OBJECT_CREATE(Shuffle);


    // OptimizerSgd
    using OptimizerSgd_fp32     = bb::OptimizerSgd<float>;
    using OptimizerAdam_fp32    = bb::OptimizerAdam<float>;
    using OptimizerAdaGrad_fp32 = bb::OptimizerAdaGrad<float>;
    BB_OBJECT_CREATE(OptimizerSgd_fp32);
    BB_OBJECT_CREATE(OptimizerAdaGrad_fp32);
    BB_OBJECT_CREATE(OptimizerAdam_fp32);

    // Loss
    using LossMeanSquaredError_fp32    = bb::LossMeanSquaredError<float>;
    using LossSoftmaxCrossEntropy_fp32 = bb::LossSoftmaxCrossEntropy<float>;
    BB_OBJECT_CREATE(LossSoftmaxCrossEntropy_fp32);
    BB_OBJECT_CREATE(LossMeanSquaredError_fp32);

    // Metrics
    using MetricsCategoricalAccuracy_fp32 = bb::MetricsCategoricalAccuracy<float>;
    using MetricsBinaryAccuracy_fp32      = bb::MetricsBinaryAccuracy<float>;
    using MetricsMeanSquaredError_fp32    = bb::MetricsMeanSquaredError<float>;
    BB_OBJECT_CREATE(MetricsCategoricalAccuracy_fp32);
    BB_OBJECT_CREATE(MetricsBinaryAccuracy_fp32);
    BB_OBJECT_CREATE(MetricsMeanSquaredError_fp32);

    // ValueGenerator                          
    using ValueGenerator_fp32               = bb::ValueGenerator<float>;
    using NormalDistributionGenerator_fp32  = bb::NormalDistributionGenerator<float>;
    using UniformDistributionGenerator_fp32 = bb::UniformDistributionGenerator<float>;
    BB_OBJECT_CREATE(NormalDistributionGenerator_fp32);
    BB_OBJECT_CREATE(UniformDistributionGenerator_fp32);


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
