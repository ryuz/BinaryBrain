// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2021 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#ifndef BB_PYBIND11
#define BB_PYBIND11
#endif

#ifndef BB_OBJECT_RECONSTRUCTION
#define BB_OBJECT_RECONSTRUCTION
#endif


#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "bb/Object.h"

#include "bb/Version.h"
#include "bb/DataType.h"

#include "bb/Tensor.h"
#include "bb/FrameBuffer.h"
#include "bb/Variables.h"

#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/BinaryModulation.h"
#include "bb/BitEncode.h"
#include "bb/Reduce.h"

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/DepthwiseDenseAffine.h"
#include "bb/DifferentiableLutN.h"
#include "bb/DifferentiableLutDiscreteN.h"
#include "bb/MicroMlp.h"
#include "bb/BinaryLutN.h"

#include "bb/Convolution2d.h"
#include "bb/MaxPooling.h"
#include "bb/StochasticMaxPooling.h"
#include "bb/StochasticMaxPooling2x2.h"
#include "bb/UpSampling.h"

#include "bb/Binarize.h"
#include "bb/Sigmoid.h"
#include "bb/ReLU.h"
#include "bb/HardTanh.h"

#include "bb/BatchNormalization.h"
#include "bb/StochasticBatchNormalization.h"
#include "bb/Dropout.h"
#include "bb/Shuffle.h"

#include "bb/LossFunction.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/LossMeanSquaredError.h"

#include "bb/MetricsFunction.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/MetricsBinaryAccuracy.h"
#include "bb/MetricsMeanSquaredError.h"

#include "bb/Optimizer.h"
#include "bb/OptimizerSgd.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerAdaGrad.h"

#include "bb/ExportVerilog.h"


#include "bb/ValueGenerator.h"
#include "bb/NormalDistributionGenerator.h"
#include "bb/UniformDistributionGenerator.h"

#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/LoadCifar10.h"



#ifdef BB_WITH_CUDA
#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#endif



// ---------------------------------
//  type definition
// ---------------------------------

// Object
using Object                                 = bb::Object;
                                             
// container                                 
using Tensor                                 = bb::Tensor;
using FrameBuffer                            = bb::FrameBuffer;
using Variables                              = bb::Variables;
                                             
                                             
// model                                     
using Model                                  = bb::Model;
using Sequential                             = bb::Sequential;

using BinaryModulation_fp32_fp32             = bb::BinaryModulation<float, float>;
using BinaryModulation_bit_fp32              = bb::BinaryModulation<bb::Bit, float>;
using RealToBinary_fp32_fp32                 = bb::RealToBinary<float, float>;
using RealToBinary_bit_fp32                  = bb::RealToBinary<bb::Bit, float>;
using BinaryToReal_fp32_fp32                 = bb::BinaryToReal<float, float>;
using BinaryToReal_bit_fp32                  = bb::BinaryToReal<bb::Bit, float>;
using BitEncode_fp32_fp32                    = bb::BitEncode<float, float>;
using BitEncode_bit_fp32                     = bb::BitEncode<bb::Bit, float>;
using Reduce_fp32_fp32                       = bb::Reduce<float, float>; 
using Reduce_bit_fp32                        = bb::Reduce<bb::Bit, float>; 

using DenseAffine_fp32                       = bb::DenseAffine<float>;
using DepthwiseDenseAffine_fp32              = bb::DepthwiseDenseAffine<float>;
                                             
using SparseModel                            = bb::SparseModel;
                                             

using BinaryLutModel                         = bb::BinaryLutModel;
                                       
using BinaryLut6_fp32_fp32                   = bb::BinaryLutN<6, float, float>;
using BinaryLut5_fp32_fp32                   = bb::BinaryLutN<5, float, float>;
using BinaryLut4_fp32_fp32                   = bb::BinaryLutN<4, float, float>;
using BinaryLut3_fp32_fp32                   = bb::BinaryLutN<3, float, float>;
using BinaryLut2_fp32_fp32                   = bb::BinaryLutN<2, float, float>;
using BinaryLut6_bit_fp32                    = bb::BinaryLutN<6, bb::Bit, float>;
using BinaryLut5_bit_fp32                    = bb::BinaryLutN<5, bb::Bit, float>;
using BinaryLut4_bit_fp32                    = bb::BinaryLutN<4, bb::Bit, float>;
using BinaryLut3_bit_fp32                    = bb::BinaryLutN<3, bb::Bit, float>;
using BinaryLut2_bit_fp32                    = bb::BinaryLutN<2, bb::Bit, float>;

                                        
using StochasticLutModel                     = bb::StochasticLutModel;

using StochasticLut6_fp32_fp32               = bb::StochasticLutN<6, float, float>;
using StochasticLut5_fp32_fp32               = bb::StochasticLutN<5, float, float>;
using StochasticLut4_fp32_fp32               = bb::StochasticLutN<4, float, float>;
using StochasticLut3_fp32_fp32               = bb::StochasticLutN<3, float, float>;
using StochasticLut2_fp32_fp32               = bb::StochasticLutN<2, float, float>;
using StochasticLut6_bit_fp32                = bb::StochasticLutN<6, bb::Bit, float>;
using StochasticLut5_bit_fp32                = bb::StochasticLutN<5, bb::Bit, float>;
using StochasticLut4_bit_fp32                = bb::StochasticLutN<4, bb::Bit, float>;
using StochasticLut3_bit_fp32                = bb::StochasticLutN<3, bb::Bit, float>;
using StochasticLut2_bit_fp32                = bb::StochasticLutN<2, bb::Bit, float>;

using DifferentiableLutModel                 = bb::DifferentiableLutModel;

using DifferentiableLut6_fp32_fp32           = bb::DifferentiableLutN<6, float, float>;
using DifferentiableLut5_fp32_fp32           = bb::DifferentiableLutN<5, float, float>;
using DifferentiableLut4_fp32_fp32           = bb::DifferentiableLutN<4, float, float>;
using DifferentiableLut3_fp32_fp32           = bb::DifferentiableLutN<3, float, float>;
using DifferentiableLut2_fp32_fp32           = bb::DifferentiableLutN<2, float, float>;
using DifferentiableLut6_bit_fp32            = bb::DifferentiableLutN<6, bb::Bit, float>;
using DifferentiableLut5_bit_fp32            = bb::DifferentiableLutN<5, bb::Bit, float>;
using DifferentiableLut4_bit_fp32            = bb::DifferentiableLutN<4, bb::Bit, float>;
using DifferentiableLut3_bit_fp32            = bb::DifferentiableLutN<3, bb::Bit, float>;
using DifferentiableLut2_bit_fp32            = bb::DifferentiableLutN<2, bb::Bit, float>;

using DifferentiableLutDiscrete6_fp32_fp32   = bb::DifferentiableLutDiscreteN<6, float, float>;
using DifferentiableLutDiscrete5_fp32_fp32   = bb::DifferentiableLutDiscreteN<5, float, float>;
using DifferentiableLutDiscrete4_fp32_fp32   = bb::DifferentiableLutDiscreteN<4, float, float>;
using DifferentiableLutDiscrete3_fp32_fp32   = bb::DifferentiableLutDiscreteN<3, float, float>;
using DifferentiableLutDiscrete2_fp32_fp32   = bb::DifferentiableLutDiscreteN<2, float, float>;
using DifferentiableLutDiscrete6_bit_fp32    = bb::DifferentiableLutDiscreteN<6, bb::Bit, float>;
using DifferentiableLutDiscrete5_bit_fp32    = bb::DifferentiableLutDiscreteN<5, bb::Bit, float>;
using DifferentiableLutDiscrete4_bit_fp32    = bb::DifferentiableLutDiscreteN<4, bb::Bit, float>;
using DifferentiableLutDiscrete3_bit_fp32    = bb::DifferentiableLutDiscreteN<3, bb::Bit, float>;
using DifferentiableLutDiscrete2_bit_fp32    = bb::DifferentiableLutDiscreteN<2, bb::Bit, float>;

using MicroMlp6_16_fp32_fp32                 = bb::MicroMlp<6, 16, float, float>;
using MicroMlp5_16_fp32_fp32                 = bb::MicroMlp<5, 16, float, float>;
using MicroMlp4_16_fp32_fp32                 = bb::MicroMlp<4, 16, float, float>;
using MicroMlp3_16_fp32_fp32                 = bb::MicroMlp<3, 16, float, float>;
using MicroMlp2_16_fp32_fp32                 = bb::MicroMlp<2, 16, float, float>;
using MicroMlp6_16_bit_fp32                  = bb::MicroMlp<6, 16, bb::Bit, float>;
using MicroMlp5_16_bit_fp32                  = bb::MicroMlp<5, 16, bb::Bit, float>;
using MicroMlp4_16_bit_fp32                  = bb::MicroMlp<4, 16, bb::Bit, float>;
using MicroMlp3_16_bit_fp32                  = bb::MicroMlp<3, 16, bb::Bit, float>;
using MicroMlp2_16_bit_fp32                  = bb::MicroMlp<2, 16, bb::Bit, float>;


using Filter2d                               = bb::Filter2d;

using ConvolutionCol2Im_fp32_fp32            = bb::ConvolutionCol2Im <float, float>;
using ConvolutionCol2Im_bit_fp32             = bb::ConvolutionCol2Im <bb::Bit, float>;
using ConvolutionIm2Col_fp32_fp32            = bb::ConvolutionIm2Col <float, float>;
using ConvolutionIm2Col_bit_fp32             = bb::ConvolutionIm2Col <bb::Bit, float>;
using Convolution2d_fp32_fp32                = bb::Convolution2d<float, float>;
using Convolution2d_bit_fp32                 = bb::Convolution2d<bb::Bit, float>;

using MaxPooling_fp32_fp32                   = bb::MaxPooling<float, float>;
using MaxPooling_bit_fp32                    = bb::MaxPooling<bb::Bit, float>;

using StochasticMaxPooling_fp32_fp32         = bb::StochasticMaxPooling<float, float>;
using StochasticMaxPooling_bit_fp32          = bb::StochasticMaxPooling<bb::Bit, float>;
using StochasticMaxPooling2x2_fp32_fp32      = bb::StochasticMaxPooling2x2<float, float>;
using StochasticMaxPooling2x2_bit_fp32       = bb::StochasticMaxPooling2x2<bb::Bit, float>;

using UpSampling_fp32_fp32                   = bb::UpSampling<float, float>;
using UpSampling_bit_fp32                    = bb::UpSampling<bb::Bit, float>;

using Activation                             = bb::Activation;
using Binarize_fp32_fp32                     = bb::Binarize<float, float>;
using Binarize_bit_fp32                      = bb::Binarize<bb::Bit, float>;
using Sigmoid_fp32_fp32                      = bb::Sigmoid<float, float>;
using Sigmoid_bit_fp32                       = bb::Sigmoid<bb::Bit, float>;
using ReLU_fp32_fp32                         = bb::ReLU<float, float>;
using ReLU_bit_fp32                          = bb::ReLU<bb::Bit, float>;
using HardTanh_fp32_fp32                     = bb::HardTanh<float, float>;
using HardTanh_bit_fp32                      = bb::HardTanh<bb::Bit, float>;

using BatchNormalization_fp32                = bb::BatchNormalization<float>;
using StochasticBatchNormalization_fp32      = bb::StochasticBatchNormalization<float>;
using Dropout_fp32_fp32                      = bb::Dropout<float, float>;
using Dropout_bit_fp32                       = bb::Dropout<bb::Bit, float>;
using Shuffle                                = bb::Shuffle;
                                        
using LossFunction                           = bb::LossFunction;
using LossMeanSquaredError_fp32              = bb::LossMeanSquaredError<float>;
using LossSoftmaxCrossEntropy_fp32           = bb::LossSoftmaxCrossEntropy<float>;
                                        
using MetricsFunction                        = bb::MetricsFunction;
using MetricsCategoricalAccuracy_fp32        = bb::MetricsCategoricalAccuracy<float>;
using MetricsBinaryAccuracy_fp32             = bb::MetricsBinaryAccuracy<float>;
using MetricsMeanSquaredError_fp32           = bb::MetricsMeanSquaredError<float>;

using Optimizer                              = bb::Optimizer;
using OptimizerSgd_fp32                      = bb::OptimizerSgd<float>;
using OptimizerAdam_fp32                     = bb::OptimizerAdam<float>;
using OptimizerAdaGrad_fp32                  = bb::OptimizerAdaGrad<float>;
                                        
using ValueGenerator_fp32                    = bb::ValueGenerator<float>;
using NormalDistributionGenerator_fp32       = bb::NormalDistributionGenerator<float>;
using UniformDistributionGenerator_fp32      = bb::UniformDistributionGenerator<float>;

using TrainData_fp32                         = bb::TrainData<float>;
using LoadMnist_fp32                         = bb::LoadMnist<float>;
using LoadCifar10_fp32                       = bb::LoadCifar10<float>;

//using RunStatus                            = bb::RunStatus;
//using Runner                               = bb::Runner<float>;



// ---------------------------------
//  functions
// ---------------------------------

int GetDeviceCount(void)
{
#if BB_WITH_CUDA
    return bbcu_GetDeviceCount();
#else
    return 0;
#endif
}

void SetDevice(int device)
{
#if BB_WITH_CUDA
    bbcu_SetDevice(device);
#endif
}

std::string GetDevicePropertiesString(int device)
{
#if BB_WITH_CUDA
    return bbcu::GetDevicePropertiesString(device);
#else
    return "host only\n"
#endif
}

inline std::string GetDevicePropertiesName(int device=0)
{
    std::map<std::string, std::int64_t> prop;

#if BB_WITH_CUDA
    int dev_count = bbcu_GetDeviceCount();
    if ( device < dev_count ) {
        cudaDeviceProp dev_prop;
        BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev_prop, device));
        return dev_prop.name;
    }
#endif

    return "no CUDA";
}

inline std::map<std::string, std::int64_t> GetDeviceProperties(int device=0)
{
    std::map<std::string, std::int64_t> prop;

#if BB_WITH_CUDA
    int dev_count = bbcu_GetDeviceCount();
    if ( device < dev_count ) {
        cudaDeviceProp dev_prop;
        BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev_prop, device));
    
        prop["totalGlobalMem"]           = (std::int64_t)dev_prop.totalGlobalMem          ;
        prop["sharedMemPerBlock"]        = (std::int64_t)dev_prop.sharedMemPerBlock       ;
        prop["regsPerBlock"]             = (std::int64_t)dev_prop.regsPerBlock            ;
        prop["warpSize"]                 = (std::int64_t)dev_prop.warpSize                ;
        prop["memPitch"]                 = (std::int64_t)dev_prop.memPitch                ;
        prop["maxThreadsPerBlock"]       = (std::int64_t)dev_prop.maxThreadsPerBlock      ;
        prop["maxThreadsDim[0]"]         = (std::int64_t)dev_prop.maxThreadsDim[0]        ;
        prop["maxThreadsDim[1]"]         = (std::int64_t)dev_prop.maxThreadsDim[1]        ;
        prop["maxThreadsDim[2]"]         = (std::int64_t)dev_prop.maxThreadsDim[2]        ;
        prop["maxGridSize[0]"]           = (std::int64_t)dev_prop.maxGridSize[0]          ;
        prop["maxGridSize[1]"]           = (std::int64_t)dev_prop.maxGridSize[1]          ;
        prop["maxGridSize[2]"]           = (std::int64_t)dev_prop.maxGridSize[2]          ;
        prop["clockRate"]                = (std::int64_t)dev_prop.clockRate               ;
        prop["totalConstMem"]            = (std::int64_t)dev_prop.totalConstMem           ;
        prop["major"]                    = (std::int64_t)dev_prop.major                   ;
        prop["minor"]                    = (std::int64_t)dev_prop.minor                   ;
        prop["textureAlignment"]         = (std::int64_t)dev_prop.textureAlignment        ;
        prop["deviceOverlap"]            = (std::int64_t)dev_prop.deviceOverlap           ;
        prop["multiProcessorCount"]      = (std::int64_t)dev_prop.multiProcessorCount     ;
        prop["kernelExecTimeoutEnabled"] = (std::int64_t)dev_prop.kernelExecTimeoutEnabled;
        prop["integrated"]               = (std::int64_t)dev_prop.integrated              ;
        prop["canMapHostMemory"]         = (std::int64_t)dev_prop.canMapHostMemory        ;
        prop["computeMode"]              = (std::int64_t)dev_prop.computeMode             ;
    }
#endif
    return prop;
}




std::string MakeVerilog_LutLayers(std::string module_name, std::vector< std::shared_ptr< bb::Model > > layers)
{
    std::stringstream ss;
    bb::ExportVerilog_LutModels(ss, module_name, layers);
    return ss.str();
}


std::string MakeVerilog_LutConvLayers(std::string module_name, std::vector< std::shared_ptr< bb::Model > > layers)
{
    std::stringstream ss;
    bb::ExportVerilog_LutCnnLayersAxi4s(ss, module_name, layers);
    return ss.str();
}






//////////////////////////////////////]
// PyBind11 module
//////////////////////////////////////]


#define DEF_CAST_FROM_OBJECT(class_name)    \
        .def("cast_from_object", [](std::shared_ptr<Object> obj) { return std::dynamic_pointer_cast<class_name>(obj); })

#define DEF_OBJECT_PICKLE(class_name)   \
        .def(py::pickle( \
                [](const class_name &obj) { return py::make_tuple(obj.DumpObjectBytes()); }, \
                [](py::tuple t) { return std::dynamic_pointer_cast<class_name>(bb::Object_CreatePy(t[0].cast<py::bytes>())); }))

#define PYCLASS_OBJECT(class_name, superclass_name) \
                py::class_< class_name, superclass_name, std::shared_ptr<class_name> >(m, #class_name) \
                    DEF_CAST_FROM_OBJECT(class_name) \
                    DEF_OBJECT_PICKLE(class_name)


namespace py = pybind11;
PYBIND11_MODULE(core, m) {
    m.doc() = "BinaryBrain ver " + bb::GetVersionString();

    // ------------------------------------
    //  Attribute
    // ------------------------------------

    m.attr("__version__") = py::cast(BB_VERSION);

    m.attr("TYPE_BIT")    = BB_TYPE_BIT;
    m.attr("TYPE_BINARY") = BB_TYPE_BINARY;
    m.attr("TYPE_FP16")   = BB_TYPE_FP16;
    m.attr("TYPE_FP32")   = BB_TYPE_FP32;
    m.attr("TYPE_FP64")   = BB_TYPE_FP64;
    m.attr("TYPE_INT8")   = BB_TYPE_INT8;
    m.attr("TYPE_INT16")  = BB_TYPE_INT16;
    m.attr("TYPE_INT32")  = BB_TYPE_INT32;
    m.attr("TYPE_INT64")  = BB_TYPE_INT64;
    m.attr("TYPE_UINT8")  = BB_TYPE_UINT8;
    m.attr("TYPE_UINT16") = BB_TYPE_UINT16;
    m.attr("TYPE_UINT32") = BB_TYPE_UINT32;
    m.attr("TYPE_UINT64") = BB_TYPE_UINT64;
    
    m.def("dtype_get_bit_size", &bb::DataType_GetBitSize);
    m.def("dtype_get_byte_size", &bb::DataType_GetByteSize);
    


    // ------------------------------------
    //  Object
    // ------------------------------------


    py::class_< Object, std::shared_ptr<Object> >(m, "Object")
        DEF_OBJECT_PICKLE(Object)
        .def("get_object_name", &Object::GetObjectName)
        .def("dump_object", &Object::DumpObjectBytes)
        .def("load_object", &Object::LoadObjectBytes)
        .def_static("write_header", &Object::WriteHeaderPy)
        .def_static("read_header", &Object::ReadHeaderPy)
//      .def("_dump_object_data", &Object::DumpObjectDataBytes)
//      .def("_load_object_data", &Object::LoadObjectDataBytes)
        ;

    m.def("object_reconstruct", &bb::Object_ReconstructPy);



    // ------------------------------------
    //  Container
    // ------------------------------------

    // Tensor
    py::class_< Tensor, Object, std::shared_ptr<Tensor> >(m, "Tensor")
        DEF_OBJECT_PICKLE(Tensor)
        .def(py::init< bb::indices_t, int, bool >(),
            py::arg("shape"),
            py::arg("type")=BB_TYPE_FP32,
            py::arg("host_only")=false)
        .def("is_host_only", &Tensor::IsHostOnly)
        .def("get_type", &Tensor::GetType)
        .def("get_shape", &Tensor::GetShape)
        .def(py::self + py::self)
        .def(py::self + double())
        .def(double() + py::self)
        .def(py::self - py::self)
        .def(py::self - double())
        .def(double() - py::self)
        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self / py::self)
        .def(py::self / double())
        .def(double() / py::self)
        .def(py::self += py::self)
        .def(py::self += double())
        .def(py::self -= py::self)
        .def(py::self -= double())
        .def(py::self *= py::self)
        .def(py::self *= double())
        .def(py::self /= py::self)
        .def(py::self /= double())
        .def("numpy_int8",   &Tensor::Numpy<std::int8_t>)
        .def("numpy_int16",  &Tensor::Numpy<std::int16_t>)
        .def("numpy_int32",  &Tensor::Numpy<std::int32_t>)
        .def("numpy_int64",  &Tensor::Numpy<std::int64_t>)
        .def("numpy_uint8",  &Tensor::Numpy<std::int8_t>)
        .def("numpy_uint16", &Tensor::Numpy<std::uint16_t>)
        .def("numpy_uint32", &Tensor::Numpy<std::uint32_t>)
        .def("numpy_uint64", &Tensor::Numpy<std::uint64_t>)
        .def("numpy_fp32",   &Tensor::Numpy<float>)
        .def("numpy_fp64",   &Tensor::Numpy<double>)
        .def_static("from_numpy_int8",   &Tensor::FromNumpy<std::int8_t>)
        .def_static("from_numpy_int16",  &Tensor::FromNumpy<std::int16_t>)
        .def_static("from_numpy_int32",  &Tensor::FromNumpy<std::int32_t>)
        .def_static("from_numpy_int64",  &Tensor::FromNumpy<std::int64_t>)
        .def_static("from_numpy_uint8",  &Tensor::FromNumpy<std::uint8_t>)
        .def_static("from_numpy_uint16", &Tensor::FromNumpy<std::uint16_t>)
        .def_static("from_numpy_uint32", &Tensor::FromNumpy<std::uint32_t>)
        .def_static("from_numpy_uint64", &Tensor::FromNumpy<std::uint64_t>)
        .def_static("from_numpy_fp32",   &Tensor::FromNumpy<float>)
        .def_static("from_numpy_fp64",   &Tensor::FromNumpy<double>)
        ;

    // FrameBuffer
    py::class_< FrameBuffer, Object, std::shared_ptr<FrameBuffer> >(m, "FrameBuffer")
        DEF_OBJECT_PICKLE(FrameBuffer)
        .def(py::init< bb::index_t, bb::indices_t, int, bool>(),
            py::arg("frame_size") = 0,
            py::arg("shape") = bb::indices_t(),
            py::arg("data_type") = 0,
            py::arg("host_only") = false)
    
        .def("resize",  (void (FrameBuffer::*)(bb::index_t, bb::indices_t, int))&bb::FrameBuffer::Resize,
                py::arg("frame_size"),
                py::arg("shape"),
                py::arg("data_type") = BB_TYPE_FP32)
        .def("is_host_only", &FrameBuffer::IsHostOnly)
        .def("get_type", &FrameBuffer::GetType)
        .def("get_frame_size", &FrameBuffer::GetFrameSize)
        .def("get_frame_stride", &FrameBuffer::GetFrameStride)
        .def("get_node_size", &FrameBuffer::GetNodeSize)
        .def("get_node_shape", &FrameBuffer::GetShape)
        .def("range", &FrameBuffer::Range)
        .def("concatenate", &FrameBuffer::Concatenate)
        .def(py::self + py::self)
        .def(py::self + double())
        .def(double() + py::self)
        .def(py::self - py::self)
        .def(py::self - double())
        .def(double() - py::self)
        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self / py::self)
        .def(py::self / double())
        .def(double() / py::self)
        .def(py::self += py::self)
        .def(py::self += double())
        .def(py::self -= py::self)
        .def(py::self -= double())
        .def(py::self *= py::self)
        .def(py::self *= double())
        .def(py::self /= py::self)
        .def(py::self /= double())
        .def("numpy_int8",   &FrameBuffer::Numpy<std::int8_t>)
        .def("numpy_int16",  &FrameBuffer::Numpy<std::int16_t>)
        .def("numpy_int32",  &FrameBuffer::Numpy<std::int32_t>)
        .def("numpy_int64",  &FrameBuffer::Numpy<std::int64_t>)
        .def("numpy_uint8",  &FrameBuffer::Numpy<std::int8_t>)
        .def("numpy_uint16", &FrameBuffer::Numpy<std::uint16_t>)
        .def("numpy_uint32", &FrameBuffer::Numpy<std::uint32_t>)
        .def("numpy_uint64", &FrameBuffer::Numpy<std::uint64_t>)
        .def("numpy_fp32",   &FrameBuffer::Numpy<float>)
        .def("numpy_fp64",   &FrameBuffer::Numpy<double>)
        .def_static("from_numpy_int8",   &FrameBuffer::FromNumpy<std::int8_t>)
        .def_static("from_numpy_int16",  &FrameBuffer::FromNumpy<std::int16_t>)
        .def_static("from_numpy_int32",  &FrameBuffer::FromNumpy<std::int32_t>)
        .def_static("from_numpy_int64",  &FrameBuffer::FromNumpy<std::int64_t>)
        .def_static("from_numpy_uint8",  &FrameBuffer::FromNumpy<std::uint8_t>)
        .def_static("from_numpy_uint16", &FrameBuffer::FromNumpy<std::uint16_t>)
        .def_static("from_numpy_uint32", &FrameBuffer::FromNumpy<std::uint32_t>)
        .def_static("from_numpy_uint64", &FrameBuffer::FromNumpy<std::uint64_t>)
        .def_static("from_numpy_fp32",   &FrameBuffer::FromNumpy<float>)
        .def_static("from_numpy_fp64",   &FrameBuffer::FromNumpy<double>)
        .def_static("calc_frame_stride", &FrameBuffer::CalcFrameStride)
        ;
    
    // Variables
    py::class_< Variables, Object, std::shared_ptr<Variables> >(m, "Variables")
        DEF_OBJECT_PICKLE(Variables)
        .def(py::init<>())
        .def("push_back", (void (Variables::*)(Variables const &))&Variables::PushBack)
        ;



    // ------------------------------------
    //  Models
    // ------------------------------------
    
#define PYCLASS_MODEL(class_name, superclass_name)  PYCLASS_OBJECT(class_name, superclass_name)

    // model
    PYCLASS_MODEL(Model, Object)
        .def("get_name", &Model::GetName)
        .def("set_name", &Model::SetName)
        .def("is_named", &Model::IsNamed)
        .def("get_model_name", &Model::GetModelName)
        .def("get_info", &Model::GetInfoString,
                py::arg("depth")    = 0,
                py::arg("columns")  = 70,
                py::arg("nest")     = 0)
        .def("send_command",  &Model::SendCommand, "SendCommand",
                py::arg("command"),
                py::arg("send_to") = "all")
        .def("get_input_shape", &Model::GetInputShape)
        .def("set_input_shape", &Model::SetInputShape)
        .def("get_output_shape", &Model::GetOutputShape)
        .def("get_input_node_size", &Model::GetInputNodeSize)
        .def("get_output_node_size", &Model::GetOutputNodeSize)
        .def("get_parameters", &Model::GetParameters)
        .def("get_gradients", &Model::GetGradients)
        .def("forward_node",  &Model::ForwardNode)
        .def("forward",  &Model::Forward)
        .def("backward", &Model::Backward)
        .def("dump", &Model::DumpBytes)
        .def("load", &Model::LoadBytes)
        .def("save_binary", &Model::SaveBinary)
        .def("load_binary", &Model::LoadBinary)
        .def("save_json", &Model::SaveJson)
        .def("load_json", &Model::LoadJson);
    

    PYCLASS_MODEL(Sequential, Model)
        .def_static("create",   &Sequential::Create)
        .def("add",             &Sequential::Add)
        ;


    PYCLASS_MODEL(BitEncode_fp32_fp32, Model)
        .def_static("create",   &BitEncode_fp32_fp32::CreatePy);
    PYCLASS_MODEL(BitEncode_bit_fp32, Model)
        .def_static("create",   &BitEncode_fp32_fp32::CreatePy);
    

    PYCLASS_MODEL(Shuffle, Model)
        .def_static("create",   &Shuffle::CreatePy);
    
    PYCLASS_MODEL(BinaryModulation_fp32_fp32, Model)
        .def_static("create", &BinaryModulation_fp32_fp32::CreatePy,
                py::arg("layer"),
                py::arg("output_shape")              = bb::indices_t(),
                py::arg("depth_modulation_size")     = 1,
                py::arg("training_modulation_size")  = 1,
                py::arg("training_value_generator")  = nullptr,
                py::arg("training_framewise")        = true,
                py::arg("training_input_range_lo")   = 0.0f,
                py::arg("training_input_range_hi")   = 1.0f,
                py::arg("inference_modulation_size") = -1,
                py::arg("inference_value_generator") = nullptr,
                py::arg("inference_framewise")       = true,
                py::arg("inference_input_range_lo")  = 0.0f,
                py::arg("inference_input_range_hi")  = 1.0f);

    PYCLASS_MODEL(BinaryModulation_bit_fp32, Model)
        .def_static("create", &BinaryModulation_bit_fp32::CreatePy,
                py::arg("layer"),
                py::arg("output_shape")              = bb::indices_t(),
                py::arg("depth_modulation_size")     = 1,
                py::arg("training_modulation_size")  = 1,
                py::arg("training_value_generator")  = nullptr,
                py::arg("training_framewise")        = true,
                py::arg("training_input_range_lo")   = 0.0f,
                py::arg("training_input_range_hi")   = 1.0f,
                py::arg("inference_modulation_size") = -1,
                py::arg("inference_value_generator") = nullptr,
                py::arg("inference_framewise")       = true,
                py::arg("inference_input_range_lo")  = 0.0f,
                py::arg("inference_input_range_hi")  = 1.0f);

    PYCLASS_MODEL(RealToBinary_fp32_fp32, Model)
        .def_static("create", &RealToBinary_fp32_fp32::CreatePy,
                py::arg("frame_modulation_size") = 1,
                py::arg("depth_modulation_size") = 1,
                py::arg("value_generator")  = nullptr,
                py::arg("framewise")        = false,
                py::arg("input_range_lo")   = 0.0f,
                py::arg("input_range_hi")   = 1.0f);

    PYCLASS_MODEL(RealToBinary_bit_fp32, Model)
        .def_static("create", &RealToBinary_bit_fp32::CreatePy,
                py::arg("frame_modulation_size") = 1,
                py::arg("depth_modulation_size") = 1,
                py::arg("value_generator")  = nullptr,
                py::arg("framewise")        = false,
                py::arg("input_range_lo")   = 0.0f,
                py::arg("input_range_hi")   = 1.0f);

    PYCLASS_MODEL(BinaryToReal_fp32_fp32, Model)
        .def_static("create", &BinaryToReal_fp32_fp32::CreatePy,
                py::arg("frame_integration_size") = 1,
                py::arg("depth_integration_size") = 0,
                py::arg("output_shape")           = bb::indices_t());

    PYCLASS_MODEL(BinaryToReal_bit_fp32, Model)
        .def_static("create", &BinaryToReal_bit_fp32::CreatePy,
                py::arg("frame_integration_size") = 1,
                py::arg("depth_integration_size") = 0,
                py::arg("output_shape")          = bb::indices_t());

    PYCLASS_MODEL(Reduce_fp32_fp32, Model)
        .def_static("create",   &Reduce_fp32_fp32::CreatePy);
    PYCLASS_MODEL(Reduce_bit_fp32, Model)
        .def_static("create",   &Reduce_bit_fp32::CreatePy);


    // DenseAffine
    PYCLASS_MODEL(DenseAffine_fp32, Model)
        .def_static("create",   &DenseAffine_fp32::CreatePy, "create",
            py::arg("output_shape"),
            py::arg("initialize_std") = 0.01f,
            py::arg("initializer")    = "he",
            py::arg("seed")           = 1)
        .def("W", ((Tensor& (DenseAffine_fp32::*)())&DenseAffine_fp32::W))
        .def("b", ((Tensor& (DenseAffine_fp32::*)())&DenseAffine_fp32::b))
        .def("dW", ((Tensor& (DenseAffine_fp32::*)())&DenseAffine_fp32::dW))
        .def("db", ((Tensor& (DenseAffine_fp32::*)())&DenseAffine_fp32::db));
    

    // DepthwiseDenseAffine
    PYCLASS_MODEL(DepthwiseDenseAffine_fp32, Model)
        .def_static("create",   &DepthwiseDenseAffine_fp32::CreatePy, "create",
            py::arg("output_shape"),
            py::arg("input_point_size")=0,
            py::arg("depth_size")=0,
            py::arg("initialize_std")= 0.01f,
            py::arg("initializer")="he",
            py::arg("seed")= 1)
        .def("W", ((Tensor& (DepthwiseDenseAffine_fp32::*)())&DepthwiseDenseAffine_fp32::W))
        .def("b", ((Tensor& (DepthwiseDenseAffine_fp32::*)())&DepthwiseDenseAffine_fp32::b))
        .def("dW", ((Tensor& (DepthwiseDenseAffine_fp32::*)())&DepthwiseDenseAffine_fp32::dW))
        .def("db", ((Tensor& (DepthwiseDenseAffine_fp32::*)())&DepthwiseDenseAffine_fp32::db));


    // SparseModel
    PYCLASS_MODEL(SparseModel, Model)
        .def("get_connection_size", &SparseModel::GetConnectionSize)
        .def("set_connection", &SparseModel::SetConnectionIndices)
        .def("get_connection", &SparseModel::GetConnectionIndices)
        .def("set_connection_index", &SparseModel::SetConnectionIndex)
        .def("get_connection_index", &SparseModel::GetConnectionIndex)
        .def("get_node_connection_size", &SparseModel::GetNodeConnectionSize)
        .def("set_node_connection_index", &SparseModel::SetNodeConnectionIndex)
        .def("get_node_connection_index", &SparseModel::GetNodeConnectionIndex)
        .def("get_lut_table_size", &SparseModel::GetLutTableSize)
        .def("get_lut_table", &SparseModel::GetLutTable);

    // BinaryLUT
    PYCLASS_MODEL(BinaryLutModel, SparseModel)
        .def("get_lut_table_size", &BinaryLutModel::GetLutTableSize)
        .def("get_lut_table", &BinaryLutModel::GetLutTable)
        .def("set_lut_table", &BinaryLutModel::SetLutTable)
        .def("import_layer", &BinaryLutModel::ImportLayer);

    PYCLASS_MODEL(BinaryLut6_fp32_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut6_fp32_fp32::CreatePy);
    PYCLASS_MODEL(BinaryLut5_fp32_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut5_fp32_fp32::CreatePy);
    PYCLASS_MODEL(BinaryLut4_fp32_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut4_fp32_fp32::CreatePy);
    PYCLASS_MODEL(BinaryLut3_fp32_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut3_fp32_fp32::CreatePy);
    PYCLASS_MODEL(BinaryLut2_fp32_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut2_fp32_fp32::CreatePy);

    PYCLASS_MODEL(BinaryLut6_bit_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut6_bit_fp32::CreatePy);
    PYCLASS_MODEL(BinaryLut5_bit_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut5_bit_fp32::CreatePy);
    PYCLASS_MODEL(BinaryLut4_bit_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut4_bit_fp32::CreatePy);
    PYCLASS_MODEL(BinaryLut3_bit_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut3_bit_fp32::CreatePy);
    PYCLASS_MODEL(BinaryLut2_bit_fp32, BinaryLutModel)
        .def_static("create", &BinaryLut2_bit_fp32::CreatePy);

    // StochasticLutModel
    PYCLASS_MODEL(StochasticLutModel, SparseModel)
        .def("W",  ((Tensor& (StochasticLutModel::*)())&StochasticLutModel::W))
        .def("dW", ((Tensor& (StochasticLutModel::*)())&StochasticLutModel::dW));

    // StochasticLut
    PYCLASS_MODEL(StochasticLut6_fp32_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut6_fp32_fp32::CreatePy);
    PYCLASS_MODEL(StochasticLut5_fp32_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut5_fp32_fp32::CreatePy);
    PYCLASS_MODEL(StochasticLut4_fp32_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut4_fp32_fp32::CreatePy);
    PYCLASS_MODEL(StochasticLut3_fp32_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut3_fp32_fp32::CreatePy);
    PYCLASS_MODEL(StochasticLut2_fp32_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut2_fp32_fp32::CreatePy);

    PYCLASS_MODEL(StochasticLut6_bit_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut6_bit_fp32::CreatePy);
    PYCLASS_MODEL(StochasticLut5_bit_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut5_bit_fp32::CreatePy);
    PYCLASS_MODEL(StochasticLut4_bit_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut4_bit_fp32::CreatePy);
    PYCLASS_MODEL(StochasticLut3_bit_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut3_bit_fp32::CreatePy);
    PYCLASS_MODEL(StochasticLut2_bit_fp32, StochasticLutModel)
        .def_static("create", &StochasticLut2_bit_fp32::CreatePy);


    // DifferentiableModel
    PYCLASS_MODEL(DifferentiableLutModel, StochasticLutModel)
        .def("get_mean",  &DifferentiableLutModel::GetMean)
        .def("get_var",   &DifferentiableLutModel::GetVar)
        .def("get_gamma", &DifferentiableLutModel::GetGamma)
        .def("get_beta",  &DifferentiableLutModel::GetBeta);
    
    // DifferentiableLut
    PYCLASS_MODEL(DifferentiableLut6_fp32_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut6_fp32_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLut5_fp32_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut5_fp32_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLut4_fp32_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut4_fp32_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLut3_fp32_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut3_fp32_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLut2_fp32_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut2_fp32_fp32::CreatePy);

    PYCLASS_MODEL(DifferentiableLut6_bit_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut6_bit_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLut5_bit_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut5_bit_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLut4_bit_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut4_bit_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLut3_bit_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut3_bit_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLut2_bit_fp32, DifferentiableLutModel)
        .def_static("create", &DifferentiableLut2_bit_fp32::CreatePy);


    // DifferentiableLutDiscrete
    PYCLASS_MODEL(DifferentiableLutDiscrete6_fp32_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete6_fp32_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLutDiscrete5_fp32_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete5_fp32_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLutDiscrete4_fp32_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete4_fp32_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLutDiscrete3_fp32_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete3_fp32_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLutDiscrete2_fp32_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete2_fp32_fp32::CreatePy);

    PYCLASS_MODEL(DifferentiableLutDiscrete6_bit_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete6_bit_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLutDiscrete5_bit_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete5_bit_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLutDiscrete4_bit_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete4_bit_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLutDiscrete3_bit_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete3_bit_fp32::CreatePy);
    PYCLASS_MODEL(DifferentiableLutDiscrete2_bit_fp32, StochasticLutModel)
        .def_static("create", &DifferentiableLutDiscrete2_bit_fp32::CreatePy);

    // MicroMlp
    PYCLASS_MODEL(MicroMlp6_16_fp32_fp32, SparseModel)
        .def_static("create", &MicroMlp6_16_fp32_fp32::CreatePy);
    PYCLASS_MODEL(MicroMlp5_16_fp32_fp32, SparseModel)
        .def_static("create", &MicroMlp5_16_fp32_fp32::CreatePy);
    PYCLASS_MODEL(MicroMlp4_16_fp32_fp32, SparseModel)
        .def_static("create", &MicroMlp4_16_fp32_fp32::CreatePy);
    PYCLASS_MODEL(MicroMlp3_16_fp32_fp32, SparseModel)
        .def_static("create", &MicroMlp3_16_fp32_fp32::CreatePy);
    PYCLASS_MODEL(MicroMlp2_16_fp32_fp32, SparseModel)
        .def_static("create", &MicroMlp2_16_fp32_fp32::CreatePy);

    PYCLASS_MODEL(MicroMlp6_16_bit_fp32, SparseModel)
        .def_static("create", &MicroMlp6_16_bit_fp32::CreatePy);
    PYCLASS_MODEL(MicroMlp5_16_bit_fp32, SparseModel)
        .def_static("create", &MicroMlp5_16_bit_fp32::CreatePy);
    PYCLASS_MODEL(MicroMlp4_16_bit_fp32, SparseModel)
        .def_static("create", &MicroMlp4_16_bit_fp32::CreatePy);
    PYCLASS_MODEL(MicroMlp3_16_bit_fp32, SparseModel)
        .def_static("create", &MicroMlp3_16_bit_fp32::CreatePy);
    PYCLASS_MODEL(MicroMlp2_16_bit_fp32, SparseModel)
        .def_static("create", &MicroMlp2_16_bit_fp32::CreatePy);


    // filter
    PYCLASS_MODEL(Filter2d, Model)
        ;

    PYCLASS_MODEL(ConvolutionIm2Col_fp32_fp32, Model)
        .def_static("create", &ConvolutionIm2Col_fp32_fp32::CreatePy)
        .def("get_filter_size_h", &ConvolutionIm2Col_fp32_fp32::GetFilterSizeH)
        .def("get_filter_size_w", &ConvolutionIm2Col_fp32_fp32::GetFilterSizeW)
        .def("get_stride_y",      &ConvolutionIm2Col_fp32_fp32::GetStrideY)
        .def("get_stride_x",      &ConvolutionIm2Col_fp32_fp32::GetStrideX)
        .def("get_padding",       &ConvolutionIm2Col_fp32_fp32::GetPadding)
        .def("get_border_mode",   &ConvolutionIm2Col_fp32_fp32::GetBorderMode)
        .def("get_border_value",  &ConvolutionIm2Col_fp32_fp32::GetBorderValue);

    PYCLASS_MODEL(ConvolutionIm2Col_bit_fp32, Model)
        .def_static("create", &ConvolutionIm2Col_bit_fp32::CreatePy)
        .def("get_filter_size_h", &ConvolutionIm2Col_bit_fp32::GetFilterSizeH)
        .def("get_filter_size_w", &ConvolutionIm2Col_bit_fp32::GetFilterSizeW)
        .def("get_stride_y",      &ConvolutionIm2Col_bit_fp32::GetStrideY)
        .def("get_stride_x",      &ConvolutionIm2Col_bit_fp32::GetStrideX)
        .def("get_padding",       &ConvolutionIm2Col_bit_fp32::GetPadding)
        .def("get_border_mode",   &ConvolutionIm2Col_bit_fp32::GetBorderMode)
        .def("get_border_value",  &ConvolutionIm2Col_bit_fp32::GetBorderValue);

    PYCLASS_MODEL(ConvolutionCol2Im_fp32_fp32, Model)
        .def_static("create", &ConvolutionCol2Im_fp32_fp32::CreatePy)
        .def("set_output_size", &ConvolutionCol2Im_fp32_fp32::SetOutputSize);
    PYCLASS_MODEL(ConvolutionCol2Im_bit_fp32, Model)
        .def_static("create", &ConvolutionCol2Im_bit_fp32::CreatePy)
        .def("set_output_size", &ConvolutionCol2Im_bit_fp32::SetOutputSize);

    PYCLASS_MODEL(Convolution2d_fp32_fp32, Filter2d)
        .def_static("create", &Convolution2d_fp32_fp32::CreatePy)
        .def("get_sub_layer", &Convolution2d_fp32_fp32::GetSubLayer);
    PYCLASS_MODEL(Convolution2d_bit_fp32, Filter2d)
        .def_static("create", &Convolution2d_bit_fp32::CreatePy)
        .def("get_sub_layer", &Convolution2d_bit_fp32::GetSubLayer);

    PYCLASS_MODEL(MaxPooling_fp32_fp32, Filter2d)
        .def_static("create", &MaxPooling_fp32_fp32::CreatePy);
    PYCLASS_MODEL(MaxPooling_bit_fp32, Filter2d)
        .def_static("create", &MaxPooling_bit_fp32::CreatePy);

    PYCLASS_MODEL(StochasticMaxPooling_fp32_fp32, Filter2d)
        .def_static("create", &StochasticMaxPooling_fp32_fp32::Create);
    PYCLASS_MODEL(StochasticMaxPooling_bit_fp32, Filter2d)
        .def_static("create", &StochasticMaxPooling_bit_fp32::Create);

    PYCLASS_MODEL(StochasticMaxPooling2x2_fp32_fp32, Filter2d)
        .def_static("create", &StochasticMaxPooling2x2_fp32_fp32::Create);
    PYCLASS_MODEL(StochasticMaxPooling2x2_bit_fp32, Filter2d)
        .def_static("create", &StochasticMaxPooling2x2_bit_fp32::Create);

    PYCLASS_MODEL(UpSampling_fp32_fp32, Model)
        .def_static("create", &UpSampling_fp32_fp32::CreatePy);
    PYCLASS_MODEL(UpSampling_bit_fp32, Model)
        .def_static("create", &UpSampling_bit_fp32::CreatePy);
    
    
    // activation
    PYCLASS_MODEL(Activation, Model)
        ;

    PYCLASS_MODEL(Binarize_fp32_fp32, Activation)
        .def_static("create", &Binarize_fp32_fp32::CreatePy,
                py::arg("binary_th")    =  0.0f,
                py::arg("hardtanh_min") = -1.0f,
                py::arg("hardtanh_max") = +1.0f);    
    PYCLASS_MODEL(Binarize_bit_fp32, Activation)
        .def_static("create", &Binarize_bit_fp32::CreatePy,
                py::arg("binary_th")    =  0.0f,
                py::arg("hardtanh_min") = -1.0f,
                py::arg("hardtanh_max") = +1.0f);

    PYCLASS_MODEL(Sigmoid_fp32_fp32, Binarize_fp32_fp32)
        .def_static("create",   &Sigmoid_fp32_fp32::Create);
    PYCLASS_MODEL(Sigmoid_bit_fp32, Binarize_bit_fp32)
        .def_static("create",   &Sigmoid_bit_fp32::Create);

    PYCLASS_MODEL(ReLU_fp32_fp32, Binarize_fp32_fp32)
        .def_static("create",   &ReLU_fp32_fp32::Create);
    PYCLASS_MODEL(ReLU_bit_fp32, Binarize_bit_fp32)
        .def_static("create",   &ReLU_bit_fp32::Create);

    PYCLASS_MODEL(HardTanh_fp32_fp32, Binarize_fp32_fp32)
        .def_static("create", &HardTanh_fp32_fp32::CreatePy);
    PYCLASS_MODEL(HardTanh_bit_fp32, Binarize_bit_fp32)
        .def_static("create", &HardTanh_bit_fp32::CreatePy);

    
    PYCLASS_MODEL(Dropout_fp32_fp32, Activation)
        .def_static("create", &Dropout_fp32_fp32::CreatePy,
                py::arg("rate") = 0.5,
                py::arg("seed") = 1);
    PYCLASS_MODEL(Dropout_bit_fp32, Activation)
        .def_static("create", &Dropout_bit_fp32::CreatePy,
                py::arg("rate") = 0.5,
                py::arg("seed") = 1);

    PYCLASS_MODEL(BatchNormalization_fp32, Activation)
        .def_static("create", &BatchNormalization_fp32::CreatePy,
                py::arg("momentum")  = 0.9f,
                py::arg("gamma")     = 1.0f,
                py::arg("beta")      = 0.0f,
                py::arg("fix_gamma") = false,
                py::arg("fix_beta")  = false);

    PYCLASS_MODEL(StochasticBatchNormalization_fp32, Activation)
        .def_static("create", &StochasticBatchNormalization_fp32::CreatePy,
                py::arg("momentum")  = 0.9,
                py::arg("gamma")     = 0.2,
                py::arg("beta")      = 0.5);



    // ------------------------------------
    //  Loss Functions
    // ------------------------------------

#define PYCLASS_LOSS(class_name, superclass_name)  PYCLASS_OBJECT(class_name, superclass_name)

    PYCLASS_LOSS(LossFunction, Object)
        .def("clear",          &LossFunction::Clear)
        .def("get_loss",       &LossFunction::GetLoss)
        .def("calculate_loss", &LossFunction::CalculateLoss,
            py::arg("y_buf"),
            py::arg("t_buf"),
            py::arg("mini_batch_size"));

    PYCLASS_LOSS(LossSoftmaxCrossEntropy_fp32, LossFunction)
        .def_static("create", &LossSoftmaxCrossEntropy_fp32::Create);

    PYCLASS_LOSS(LossMeanSquaredError_fp32, LossFunction)
        .def_static("create", &LossMeanSquaredError_fp32::Create);



    // ------------------------------------
    //  Metrics Functions
    // ------------------------------------

#define PYCLASS_METRICS(class_name, superclass_name)  PYCLASS_OBJECT(class_name, superclass_name)

    PYCLASS_METRICS(MetricsFunction, Object)
        .def("clear",              &MetricsFunction::Clear)
        .def("get_metrics",        &MetricsFunction::GetMetrics)
        .def("calculate_metrics",  &MetricsFunction::CalculateMetrics)
        .def("get_metrics_string", &MetricsFunction::GetMetricsString);

    PYCLASS_METRICS(MetricsCategoricalAccuracy_fp32, MetricsFunction)
        .def_static("create", &MetricsCategoricalAccuracy_fp32::Create);

    PYCLASS_METRICS(MetricsBinaryAccuracy_fp32, MetricsFunction)
        .def_static("create", &MetricsBinaryAccuracy_fp32::Create);

    PYCLASS_METRICS(MetricsMeanSquaredError_fp32, MetricsFunction)
        .def_static("create", &MetricsMeanSquaredError_fp32::Create);


    // ------------------------------------
    //  Optimizer
    // ------------------------------------

#define PYCLASS_OPTIMIZER(class_name, superclass_name)  PYCLASS_OBJECT(class_name, superclass_name)

    PYCLASS_OPTIMIZER(Optimizer, Object)
        .def("set_variables", &Optimizer::SetVariables)
        .def("update",        &Optimizer::Update);
  
    PYCLASS_OPTIMIZER(OptimizerSgd_fp32, Optimizer)
        .def_static("create", (std::shared_ptr<OptimizerSgd_fp32> (*)(float))&OptimizerSgd_fp32::Create, "create",
            py::arg("learning_rate") = 0.01f);
    
    PYCLASS_OPTIMIZER(OptimizerAdaGrad_fp32, Optimizer)
        .def_static("Create", (std::shared_ptr<OptimizerAdaGrad_fp32> (*)(float))&OptimizerAdaGrad_fp32::Create,
            py::arg("learning_rate") = 0.01f);

    PYCLASS_OPTIMIZER(OptimizerAdam_fp32, Optimizer)
        .def_static("create", &OptimizerAdam_fp32::CreatePy,
            py::arg("learning_rate") = 0.001f,
            py::arg("beta1")         = 0.9f,
            py::arg("beta2")         = 0.999f);
    
    

    // ------------------------------------
    //  ValueGenerator
    // ------------------------------------

#define PYCLASS_VALUEGENERATOR(class_name, superclass_name)  PYCLASS_OBJECT(class_name, superclass_name)

    PYCLASS_VALUEGENERATOR(ValueGenerator_fp32, Object);
    
    PYCLASS_VALUEGENERATOR(NormalDistributionGenerator_fp32, ValueGenerator_fp32)
        .def_static("create", &NormalDistributionGenerator_fp32::Create,
            py::arg("mean")   = 0.0f,
            py::arg("stddev") = 1.0f,
            py::arg("seed")   = 1);
    
    PYCLASS_VALUEGENERATOR(UniformDistributionGenerator_fp32, ValueGenerator_fp32)
        .def_static("Create", &UniformDistributionGenerator_fp32::Create,
            py::arg("a")    = 0.0f,
            py::arg("b")    = 1.0f,
            py::arg("seed") = 1);



    // ------------------------------------
    //  Dataset
    // ------------------------------------

    // TrainData
    py::class_< TrainData_fp32 >(m, "TrainData_fp32")
        .def_readwrite("x_shape", &TrainData_fp32::x_shape)
        .def_readwrite("t_shape", &TrainData_fp32::t_shape)
        .def_readwrite("x_train", &TrainData_fp32::x_train)
        .def_readwrite("t_train", &TrainData_fp32::t_train)
        .def_readwrite("x_test",  &TrainData_fp32::x_test)
        .def_readwrite("t_test",  &TrainData_fp32::t_test)
        .def("empty", &TrainData_fp32::empty);

    // LoadMNIST
    py::class_< LoadMnist_fp32 >(m, "LoadMnist_fp32")
        .def_static("load", &LoadMnist_fp32::Load,
            py::arg("max_train") = -1,
            py::arg("max_test")  = -1,
            py::arg("num_class") = 10);
    
    // LoadCifar10
    py::class_< LoadCifar10_fp32 >(m, "LoadCifar10_fp32")
        .def_static("load", &LoadCifar10_fp32::Load,
            py::arg("num") = 5);

    /*
    // RunStatus
    py::class_< RunStatus >(m, "RunStatus")
        .def_static("WriteJson", &RunStatus::WriteJson)
        .def_static("ReadJson",  &RunStatus::ReadJson);


    // Runnner
    py::class_< Runner, std::shared_ptr<Runner> >(m, "Runner")
        .def_static("create", &Runner::CreateEx,
            py::arg("name"),
            py::arg("net"),
            py::arg("lossFunc"),
            py::arg("metricsFunc"),
            py::arg("optimizer"),
            py::arg("max_run_size") = 0,
            py::arg("print_progress") = true,
            py::arg("print_progress_loss") = true,
            py::arg("print_progress_accuracy") = true,
            py::arg("log_write") = true,
            py::arg("log_append") = true,
            py::arg("file_read") = false,
            py::arg("file_write") = false,
            py::arg("write_serial") = false,
            py::arg("initial_evaluation") = false,
            py::arg("seed") = 1)
        .def("fitting", &Runner::Fitting,
            py::arg("td"),
            py::arg("epoch_size"),
            py::arg("batch_size"));
    */


    // ------------------------------------
    //  System
    // ------------------------------------

    // version
    m.def("get_version_string", &bb::GetVersionString);

    // verilog
    m.def("make_verilog_lut_layers",     &MakeVerilog_LutLayers);
    m.def("make_verilog_lut_cnv_layers", &MakeVerilog_LutConvLayers);

    // OpenMP
    m.def("omp_set_num_threads", &omp_set_num_threads);

    // CUDA device
    py::class_< bb::Manager >(m, "Manager")
        .def_static("is_device_available", &bb::Manager::IsDeviceAvailable)
        .def_static("set_host_only", &bb::Manager::SetHostOnly);

    m.def("get_device_count",             &GetDeviceCount);
    m.def("set_device",                   &SetDevice,                 py::arg("device") = 0);
    m.def("get_device_name",              &GetDevicePropertiesName,   py::arg("device") = 0);
    m.def("get_device_properties",        &GetDeviceProperties,       py::arg("device") = 0);
    m.def("get_device_properties_string", &GetDevicePropertiesString, py::arg("device") = 0);
}



// end of file
