// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>


#include "bb/Version.h"
#include "bb/DataType.h"

#include "bb/Tensor.h"
#include "bb/FrameBuffer.h"
#include "bb/Variables.h"

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/SparseLutN.h"
#include "bb/SparseLutDiscreteN.h"
#include "bb/BinaryLutN.h"
#include "bb/Reduce.h"
#include "bb/LoweringConvolution.h"
#include "bb/BinaryModulation.h"
#include "bb/Sigmoid.h"
#include "bb/ReLU.h"
#include "bb/BatchNormalization.h"

#include "bb/LossFunction.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/LossMeanSquaredError.h"

#include "bb/MetricsFunction.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/MetricsMeanSquaredError.h"

#include "bb/Optimizer.h"
#include "bb/OptimizerSgd.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerAdaGrad.h"

#include "bb/ValueGenerator.h"
#include "bb/NormalDistributionGenerator.h"
#include "bb/UniformDistributionGenerator.h"

#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/LoadCifar10.h"
#include "bb/ExportVerilog.h"


using Tensor                       = bb::Tensor;
using FrameBuffer                  = bb::FrameBuffer;
using Variables                    = bb::Variables;

using Model                        = bb::Model;
using SparseLayer                  = bb::SparseLayer;
using Sequential                   = bb::Sequential;
using DenseAffine                  = bb::DenseAffine<float>;
using LutLayer                     = bb::LutLayer<float, float>;
using LutLayerBit                  = bb::LutLayer<bb::Bit, float>;
using BinaryLut6                   = bb::BinaryLutN<6, float, float>;
using BinaryLut6Bit                = bb::BinaryLutN<6, bb::Bit, float>;
using SparseLut2                   = bb::SparseLutN<2, float, float>;
using SparseLut2Bit                = bb::SparseLutN<2, bb::Bit, float>;
using SparseLut4                   = bb::SparseLutN<4, float, float>;
using SparseLut4Bit                = bb::SparseLutN<4, bb::Bit, float>;
using SparseLut6                   = bb::SparseLutN<6, float, float>;
using SparseLut6Bit                = bb::SparseLutN<6, bb::Bit, float>;
using Reduce                       = bb::Reduce<float, float>; 
using BinaryModulation             = bb::BinaryModulation<float, float>;
using BinaryModulationBit          = bb::BinaryModulation<bb::Bit, float>;

using Filter2d                     = bb::Filter2d<float, float>;
using Filter2dBit                  = bb::Filter2d<bb::Bit, float>;
using LoweringConvolution          = bb::LoweringConvolution<float, float>;
using LoweringConvolutionBit       = bb::LoweringConvolution<bb::Bit, float>;
using MaxPooling                   = bb::MaxPooling<float, float>;
using MaxPoolingBit                = bb::MaxPooling<bb::Bit, float>;

using Activation                   = bb::Activation;
using Binarize                     = bb::Binarize<float, float>;
using BinarizeBit                  = bb::Binarize<bb::Bit, float>;
using Sigmoid                      = bb::Sigmoid<float>;
using ReLU                         = bb::ReLU<float, float>;
using ReLUBit                      = bb::ReLU<bb::Bit, float>;
using BatchNormalization           = bb::BatchNormalization<float>;

using LossFunction                 = bb::LossFunction;
using LossMeanSquaredError         = bb::LossMeanSquaredError<float>;
using LossSoftmaxCrossEntropy      = bb::LossSoftmaxCrossEntropy<float>;

using MetricsFunction              = bb::MetricsFunction;
using MetricsCategoricalAccuracy   = bb::MetricsCategoricalAccuracy<float>;
using MetricsMeanSquaredError      = bb::MetricsMeanSquaredError<float>;

using Optimizer                    = bb::Optimizer;
using OptimizerSgd                 = bb::OptimizerSgd<float>;
using OptimizerAdam                = bb::OptimizerAdam<float>;
using OptimizerAdaGrad             = bb::OptimizerAdaGrad<float>;

using ValueGenerator               = bb::ValueGenerator<float>;
using NormalDistributionGenerator  = bb::NormalDistributionGenerator<float>;
using UniformDistributionGenerator = bb::UniformDistributionGenerator<float>;

using TrainData                    = bb::TrainData<float>;
using LoadMnist                    = bb::LoadMnist<float>;
using LoadCifar10                  = bb::LoadCifar10<float>;
using RunStatus                    = bb::RunStatus;
using Runner                       = bb::Runner<float>;



std::string MakeVerilog_FromLut(std::string module_name, std::vector< std::shared_ptr< bb::LutLayer<float, float> > > layers)
{
    std::stringstream ss;
    bb::ExportVerilog_LutLayers<float, float>(ss, module_name, layers);
    return ss.str();
}

std::string MakeVerilog_FromLutBit(std::string module_name, std::vector< std::shared_ptr< bb::LutLayer<bb::Bit, float> > > layers)
{
    std::stringstream ss;
    bb::ExportVerilog_LutLayers<bb::Bit, float>(ss, module_name, layers);
    return ss.str();
}


std::string MakeVerilogAxi4s_FromLutFilter2d(std::string module_name, std::vector< std::shared_ptr< bb::Filter2d<float, float> > > layers)
{
    std::stringstream ss;
    bb::ExportVerilog_LutCnnLayersAxi4s(ss, module_name, layers);
    return ss.str();
}

std::string MakeVerilogAxi4s_FromLutFilter2dBit(std::string module_name, std::vector< std::shared_ptr< bb::Filter2d<bb::Bit, float> > > layers)
{
    std::stringstream ss;
    bb::ExportVerilog_LutCnnLayersAxi4s(ss, module_name, layers);
    return ss.str();
}


namespace py = pybind11;
PYBIND11_MODULE(core, m) {
    m.doc() = "BinaryBrain ver " + bb::GetVersionString();

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

    m.attr("BB_BORDER_CONSTANT")    = BB_BORDER_CONSTANT;
    m.attr("BB_BORDER_CONSTANT")    = BB_BORDER_CONSTANT;
    m.attr("BB_BORDER_REFLECT")     = BB_BORDER_REFLECT;
    m.attr("BB_BORDER_REFLECT_101") = BB_BORDER_REFLECT_101;
    m.attr("BB_BORDER_REPLICATE")   = BB_BORDER_REPLICATE;
    m.attr("BB_BORDER_WRAP")        = BB_BORDER_WRAP;


    // Tensor
    py::class_< Tensor >(m, "Tensor")
        .def("get_type", &Tensor::GetType)
        .def("get_shape", &Tensor::GetShape)
        .def("set_data", &Tensor::SetData<float>,
R"(set data to tensor

    set data to tensor

Args:
    data(List[List[float]]): data
)"
            )
        .def("get_data", &Tensor::GetData<float>,
R"(get data from tensor

    set data to tensor

Returns:
    tensor data
)"
            )
        .def("set_data_int32", &Tensor::SetData<int>,
R"(set data to tensor

    set data to tensor

Args:
    data(List[List[int]]): data
)"
            )
        .def("get_data_int32", &Tensor::GetData<int>,
R"(get data from tensor

    set data to tensor

Returns:
    tensor data
)"
            );


    // FrameBuffer
    py::class_< FrameBuffer >(m, "FrameBuffer")
        .def(py::init< bb::index_t, bb::indices_t, int, bool>(),
R"(FrameBuffer object constructor

Manegement frame memory on CPU or GPU

Args:
    frame_size(int): size of frames
    shape(List[int]): shape of frame
    data_type(int): frame type  TYPE_BIT or TYPE_FP32
    host_only(bool): only use host(CPU) memory.
)",
            py::arg("frame_size") = 0,
            py::arg("shape") = bb::indices_t(),
            py::arg("data_type") = 0,
            py::arg("host_only") = false)
        .def("resize",  (void (FrameBuffer::*)(bb::index_t, bb::indices_t, int))&bb::FrameBuffer::Resize,
                "resize",
                py::arg("frame_size"),
                py::arg("shape"),
                py::arg("data_type") = BB_TYPE_FP32)
        .def("get_type", &FrameBuffer::GetType)
        .def("get_frame_size", &FrameBuffer::GetFrameSize)
        .def("get_node_size", &FrameBuffer::GetNodeSize)
        .def("get_node_shape", &FrameBuffer::GetShape)
        .def("range", &FrameBuffer::Range)
        .def("concatenate", &FrameBuffer::Concatenate)

        .def("set_data", &FrameBuffer::SetData<float>,
R"(set data to frames

    set data to frames

Args:
    data(List[List[float]]): data
    offset(int): offset
)",
                py::arg("data"),
                py::arg("offset") = 0)

        .def("get_data", &FrameBuffer::GetData<float>,
R"(get data from frames

    set data to frames

Args:
    size(int): size (If you specify 0 or less, it will be the size to the end)
    offset(int): offset
)",
                py::arg("size") = 0,
                py::arg("offset") = 0);


    // Variables
    py::class_< Variables, std::shared_ptr<Variables> >(m, "Variables");



    // ------------------------------------
    //  Models
    // ------------------------------------
    
    // model
    py::class_< Model, std::shared_ptr<Model> >(m, "Model")
        .def("set_input_shape", &Model::SetInputShape)
        .def("get_info", &Model::GetInfoString,
R"(get a information of model structure.

get a information string of model network structure.

Args:
   depth(int): depth of network structure
   columns(str): size of column 
   nest(str): nest counter
Returns:
   str: strings of model information)",
                py::arg("depth")    = 0,
                py::arg("columns")  = 70,
                py::arg("nest")     = 0)
        .def("get_input_shape", &Model::GetInputShape)
        .def("get_output_shape", &Model::GetOutputShape)
        .def("get_parameters", &Model::GetParameters)
        .def("get_gradients", &Model::GetGradients)
        .def("forward_node",  &Model::ForwardNode)
        .def("forward",  &Model::Forward, "Forward",
                py::arg("x_buf"),
                py::arg("train") = true)
        .def("backward", &Model::Backward, "Backward")
        .def("send_command",  &Model::SendCommand, "SendCommand",
                py::arg("command"),
                py::arg("send_to") = "all");


    // DenseAffine
    py::class_< DenseAffine, Model, std::shared_ptr<DenseAffine> >(m, "DenseAffine")
        .def_static("create",   &DenseAffine::CreateEx, "create",
            py::arg("output_shape"),
            py::arg("initialize_std") = 0.01f,
            py::arg("initializer")    = "he",
            py::arg("seed")           = 1)
        .def("W", ((Tensor& (DenseAffine::*)())&DenseAffine::W))
        .def("b", ((Tensor& (DenseAffine::*)())&DenseAffine::b))
        .def("dW", ((Tensor& (DenseAffine::*)())&DenseAffine::dW))
        .def("db", ((Tensor& (DenseAffine::*)())&DenseAffine::db));
    

    // SparseLayer
    py::class_< SparseLayer, Model, std::shared_ptr<SparseLayer> >(m, "SparseLayer")
         .def("get_connection_size", &SparseLayer::GetConnectionSize)
         .def("set_connection", &SparseLayer::SetConnectionIndices)
         .def("get_connection", &SparseLayer::GetConnectionIndices)
         .def("set_connection_index", &SparseLayer::SetConnectionIndex)
         .def("get_connection_index", &SparseLayer::GetConnectionIndex)
         .def("get_node_connection_size", &SparseLayer::GetNodeConnectionSize)
         .def("set_node_connection_index", &SparseLayer::SetNodeConnectionIndex)
         .def("get_node_connection_index", &SparseLayer::GetNodeConnectionIndex);


    py::class_< LutLayer, SparseLayer, std::shared_ptr<LutLayer> >(m, "LutLayer")
        .def("import_parameter", &LutLayer::ImportLayer);

    py::class_< LutLayerBit, SparseLayer, std::shared_ptr<LutLayerBit> >(m, "LutLayerBit")
        .def("import_parameter", &LutLayerBit::ImportLayer);
       
    py::class_< Sequential, Model, std::shared_ptr<Sequential> >(m, "Sequential")
        .def_static("create",   &Sequential::Create)
        .def("add",             &Sequential::Add);

    py::class_< Reduce, Model, std::shared_ptr<Reduce> >(m, "Reduce")
        .def_static("create",   &Reduce::CreateEx);

    py::class_< BinaryModulation, Model, std::shared_ptr<BinaryModulation> >(m, "BinaryModulation")
        .def_static("create", &BinaryModulation::CreateEx,
                py::arg("layer"),
                py::arg("output_shape")              = bb::indices_t(),
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

    py::class_< BinaryModulationBit, Model, std::shared_ptr<BinaryModulationBit> >(m, "BinaryModulationBit")
        .def_static("create", &BinaryModulationBit::CreateEx,
                py::arg("layer"),
                py::arg("output_shape")              = bb::indices_t(),
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

    py::class_< BinaryLut6, LutLayer, std::shared_ptr<BinaryLut6> >(m, "BinaryLut6")
        .def_static("create", &BinaryLut6::CreateEx,
R"(create BinaryLut6 object

    Args:
        output_shape (List[int]): shape of output frame
        seed (int): seed of random
)",
                py::arg("output_shape"),
                py::arg("seed") = 1);
    
    py::class_< BinaryLut6Bit, LutLayerBit, std::shared_ptr<BinaryLut6Bit> >(m, "BinaryLut6Bit")
        .def_static("create", &BinaryLut6Bit::CreateEx, "create",
                py::arg("output_shape"),
                py::arg("seed") = 1);
    
    py::class_< SparseLut6, SparseLayer, std::shared_ptr<SparseLut6> >(m, "SparseLut6")
        .def_static("create", &SparseLut6::CreateEx, "create SparseLut6",
                py::arg("output_shape"),
                py::arg("batch_norm") = true,
                py::arg("connection") = "",
                py::arg("seed") = 1);

    
    // filter
    py::class_< Filter2d, Model, std::shared_ptr<Filter2d> >(m, "Filter2d");

    py::class_< Filter2dBit, Model, std::shared_ptr<Filter2dBit> >(m, "Filter2dBit");

    py::class_< LoweringConvolution, Filter2d, std::shared_ptr<LoweringConvolution> >(m, "LoweringConvolution")
        .def_static("create", &LoweringConvolution::CreateEx,
                py::arg("layer"),
                py::arg("filter_h_size"),
                py::arg("filter_w_size"),
                py::arg("y_stride")      = 1,
                py::arg("x_stride")      = 1,
                py::arg("padding")       = "valid",
                py::arg("border_mode")   = BB_BORDER_REFLECT_101,
                py::arg("border_value")  = 0.0f);

    py::class_< LoweringConvolutionBit, Filter2dBit, std::shared_ptr<LoweringConvolutionBit> >(m, "LoweringConvolutionBit")
        .def_static("create", &LoweringConvolutionBit::CreateEx,
                py::arg("layer"),
                py::arg("filter_h_size"),
                py::arg("filter_w_size"),
                py::arg("y_stride")      = 1,
                py::arg("x_stride")      = 1,
                py::arg("padding")       = "valid",
                py::arg("border_mode")   = BB_BORDER_REFLECT_101,
                py::arg("border_value")  = 0.0f);
    
    py::class_< MaxPooling, Filter2d, std::shared_ptr<MaxPooling> >(m, "MaxPooling")
        .def_static("create", &MaxPooling::CreateEx,
                py::arg("filter_h_size"),
                py::arg("filter_w_size"));


    // activation
    py::class_< Activation, Model, std::shared_ptr<Activation> >(m, "Activation");

    py::class_< Binarize, Activation, std::shared_ptr<Binarize> >(m, "Binarize")
        .def_static("create", &Binarize::CreateEx,
                py::arg("binary_th")    =  0.0f,
                py::arg("hardtanh_min") = -1.0f,
                py::arg("hardtanh_max") = +1.0f);
    
    py::class_< BinarizeBit, Activation, std::shared_ptr<BinarizeBit> >(m, "BinarizeBit")
        .def_static("create", &BinarizeBit::CreateEx,
                py::arg("binary_th")    =  0.0f,
                py::arg("hardtanh_min") = -1.0f,
                py::arg("hardtanh_max") = +1.0f);

    py::class_< Sigmoid, Binarize, std::shared_ptr<Sigmoid> >(m, "Sigmoid")
        .def_static("create",   &Sigmoid::Create);

    py::class_< ReLU, Binarize, std::shared_ptr<ReLU> >(m, "ReLU")
        .def_static("create",   &ReLU::Create);

    py::class_< ReLUBit, BinarizeBit, std::shared_ptr<ReLUBit> >(m, "ReLUBit")
        .def_static("create",   &ReLUBit::Create);

    py::class_< BatchNormalization, Activation, std::shared_ptr<BatchNormalization> >(m, "BatchNormalization")
        .def_static("create", &BatchNormalization::CreateEx,
                py::arg("momentum")  = 0.9f,
                py::arg("gamma")     = 1.0f,
                py::arg("beta")      = 0.0f,
                py::arg("fix_gamma") = false,
                py::arg("fix_beta")  = false);

    // Loss Functions
    py::class_< LossFunction, std::shared_ptr<LossFunction> >(m, "LossFunction")
        .def("clear",          &LossFunction::Clear)
        .def("get_loss",       &LossFunction::GetLoss)
        .def("calculate_loss", &LossFunction::CalculateLoss,
            py::arg("y_buf"),
            py::arg("t_buf"),
            py::arg("mini_batch_size"));

    py::class_< LossSoftmaxCrossEntropy, LossFunction, std::shared_ptr<LossSoftmaxCrossEntropy> >(m, "LossSoftmaxCrossEntropy")
        .def_static("create", &LossSoftmaxCrossEntropy::Create);

    py::class_< LossMeanSquaredError, LossFunction, std::shared_ptr<LossMeanSquaredError> >(m, "LossMeanSquaredError")
        .def_static("create", &LossMeanSquaredError::Create);


    // Metrics Functions
    py::class_< MetricsFunction, std::shared_ptr<MetricsFunction> >(m, "MetricsFunction")
        .def("clear",              &MetricsFunction::Clear)
        .def("get_metrics",        &MetricsFunction::GetMetrics)
        .def("calculate_metrics",  &MetricsFunction::CalculateMetrics)
        .def("get_metrics_string", &MetricsFunction::GetMetricsString);

    py::class_< MetricsCategoricalAccuracy, MetricsFunction, std::shared_ptr<MetricsCategoricalAccuracy> >(m, "MetricsCategoricalAccuracy")
        .def_static("create", &MetricsCategoricalAccuracy::Create);

    py::class_< MetricsMeanSquaredError, MetricsFunction, std::shared_ptr<MetricsMeanSquaredError> >(m, "MetricsMeanSquaredError")
        .def_static("create", &MetricsMeanSquaredError::Create);


    // Optimizer
    py::class_< Optimizer, std::shared_ptr<Optimizer> >(m, "Optimizer")
        .def("set_variables", &Optimizer::SetVariables)
        .def("update",        &Optimizer::Update);
    
    py::class_< OptimizerSgd, Optimizer, std::shared_ptr<OptimizerSgd> >(m, "OptimizerSgd")
        .def_static("create", (std::shared_ptr<OptimizerSgd> (*)(float))&OptimizerSgd::Create, "create",
            py::arg("learning_rate") = 0.01f);
    
    py::class_< OptimizerAdam, Optimizer, std::shared_ptr<OptimizerAdam> >(m, "OptimizerAdam")
        .def_static("create", &OptimizerAdam::CreateEx,
            py::arg("learning_rate") = 0.001f,
            py::arg("beta1")         = 0.9f,
            py::arg("beta2")         = 0.999f); 
    
    py::class_< OptimizerAdaGrad, Optimizer, std::shared_ptr<OptimizerAdaGrad> >(m, "OptimizerAdaGrad")
        .def_static("Create", (std::shared_ptr<OptimizerAdaGrad> (*)(float))&OptimizerAdaGrad::Create,
            py::arg("learning_rate") = 0.01f);


    // ValueGenerator
    py::class_< ValueGenerator, std::shared_ptr<ValueGenerator> >(m, "ValueGenerator");
    
    py::class_< NormalDistributionGenerator, ValueGenerator, std::shared_ptr<NormalDistributionGenerator> >(m, "NormalDistributionGenerator")
        .def_static("create", &NormalDistributionGenerator::Create,
            py::arg("mean")   = 0.0f,
            py::arg("stddev") = 1.0f,
            py::arg("seed")   = 1);
    
    py::class_< UniformDistributionGenerator, ValueGenerator, std::shared_ptr<UniformDistributionGenerator> >(m, "UniformDistributionGenerator")
        .def_static("Create", &UniformDistributionGenerator::Create,
            py::arg("a")    = 0.0f,
            py::arg("b")    = 1.0f,
            py::arg("seed") = 1);


    // TrainData
    py::class_< TrainData >(m, "TrainData")
        .def_readwrite("x_shape", &TrainData::x_shape)
        .def_readwrite("t_shape", &TrainData::t_shape)
        .def_readwrite("x_train", &TrainData::x_train)
        .def_readwrite("t_train", &TrainData::t_train)
        .def_readwrite("x_test",  &TrainData::x_test)
        .def_readwrite("t_test",  &TrainData::t_test)
        .def("empty", &TrainData::empty);

    // LoadMNIST
    py::class_< LoadMnist >(m, "LoadMnist")
        .def_static("load", &LoadMnist::Load,
            py::arg("max_train") = -1,
            py::arg("max_test")  = -1,
            py::arg("num_class") = 10);
    
    // LoadCifar10
    py::class_< LoadCifar10 >(m, "LoadCifar10")
        .def_static("load", &LoadCifar10::Load,
            py::arg("num") = 5);

    
    // RunStatus
    py::class_< RunStatus >(m, "RunStatus")
        .def_static("WriteJson", &RunStatus::WriteJson)
        .def_static("ReadJson",  &RunStatus::ReadJson);


    // Runnner
    py::class_< Runner, std::shared_ptr<Runner> >(m, "CRunner")
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

    // verilog
    m.def("make_verilog_from_lut", &MakeVerilog_FromLut);
    m.def("make_verilog_from_lut_bit", &MakeVerilog_FromLutBit);
    m.def("make_verilog_axi4s_from_lut_cnn", &MakeVerilogAxi4s_FromLutFilter2d);
    m.def("make_verilog_axi4s_from_lut_cnn_bit", &MakeVerilogAxi4s_FromLutFilter2dBit);

    m.def("get_version", &bb::GetVersionString);
}


// end of file
