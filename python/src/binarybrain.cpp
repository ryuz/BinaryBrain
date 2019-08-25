

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#define BB_ASSERT_EXCEPTION

#include "bb/DataType.h"

#include "bb/Tensor.h"
#include "bb/FrameBuffer.h"
#include "bb/Variables.h"

#include "bb/Sequential.h"
#include "bb/SparseLutN.h"
#include "bb/SparseLutDiscreteN.h"
#include "bb/BinaryLutN.h"
#include "bb/Reduce.h"
#include "bb/BinaryModulation.h"

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
#include "bb/ExportVerilog.h"



#include "bb/Sequential.h"
#include "bb/SparseLutN.h"
#include "bb/BinaryModulation.h"

#include "bb/Runner.h"
#include "bb/LoadMnist.h"


using Tensor                       = bb::Tensor;
using FrameBuffer                  = bb::FrameBuffer;
using Variables                    = bb::Variables;

using Model                        = bb::Model;
using SparseLayer                  = bb::SparseLayer;
using Sequential                   = bb::Sequential;
using SparseLut6                   = bb::SparseLutN<6, float>;
using Reduce                       = bb::Reduce<float, float>; 
using BinaryModulation             = bb::BinaryModulation<float, float>;

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
using Runner                       = bb::Runner<float>;
using LoadMnist                    = bb::LoadMnist<float>;

/*
struct type
{
    int const   bit    = BB_TYPE_BIT;
    int const   binary = BB_TYPE_BINARY;
    int const   fp16   = BB_TYPE_FP16;
    int const   fp32   = BB_TYPE_FP32;
    int const   fp64   = BB_TYPE_FP64;
    int const   int8   = BB_TYPE_INT8;
    int const   int16  = BB_TYPE_INT16;
    int const   int32  = BB_TYPE_INT32;
    int const   int64  = BB_TYPE_INT64;
    int const   uint8  = BB_TYPE_UINT8;
    int const   uint16 = BB_TYPE_UINT16;
    int const   uint32 = BB_TYPE_UINT32;
    int const   uint64 = BB_TYPE_UINT64;
};
*/

namespace py = pybind11;
PYBIND11_MODULE(binarybrain, m) {
    m.doc() = "binarybrain plugin";
    
    /*
    py::class_< type >(m, "type")
        .def_readonly("bit",    &type::bit)
        .def_readonly("binary", &type::binary)
        .def_readonly("fp16",   &type::fp16)
        .def_readonly("fp32",   &type::fp32)
        .def_readonly("fp64",   &type::fp64)
        .def_readonly("int8",   &type::int8)
        .def_readonly("int16",  &type::int16)
        .def_readonly("int32",  &type::int32)
        .def_readonly("int64",  &type::int64)
        .def_readonly("uint8",  &type::uint8)
        .def_readonly("uint16", &type::uint16)
        .def_readonly("uint32", &type::uint32)
        .def_readonly("uint64", &type::uint64);
    */

    m.attr("bit")    = BB_TYPE_BIT;
    m.attr("binary") = BB_TYPE_BINARY;
    m.attr("fp16")   = BB_TYPE_FP16;
    m.attr("fp32")   = BB_TYPE_FP32;
    m.attr("fp64")   = BB_TYPE_FP64;
    m.attr("int8")   = BB_TYPE_INT8;
    m.attr("int16")  = BB_TYPE_INT16;
    m.attr("int32")  = BB_TYPE_INT32;
    m.attr("int64")  = BB_TYPE_INT64;
    m.attr("uint8")  = BB_TYPE_UINT8;
    m.attr("uint16") = BB_TYPE_UINT16;
    m.attr("uint32") = BB_TYPE_UINT32;
    m.attr("uint64") = BB_TYPE_UINT64;

    py::class_< Tensor >(m, "Tensor");

    py::class_< FrameBuffer >(m, "FrameBuffer")
        .def(py::init<int, bb::index_t, bb::indices_t, bool>(),
            py::arg("data_type"),
            py::arg("frame_size"),
            py::arg("shape"),
            py::arg("hostOnly") = false)
        //int data_type, index_t frame_size, indices_t shape, bool
        .def("Resize",    (void (FrameBuffer::*)(int, bb::index_t, bb::indices_t))&bb::FrameBuffer::Resize)
        .def("GetRange",  &FrameBuffer::GetRange)
        .def("SetVector", (void (FrameBuffer::*)(std::vector< std::vector<float> > const &, bb::index_t))&FrameBuffer::SetVector<float>,
                "set vector",
                py::arg("data"),
                py::arg("offset") = 0);

    py::class_< Variables, std::shared_ptr<Variables> >(m, "Variables");

    // Models
    py::class_< Model, std::shared_ptr<Model> >(m, "Model")
        .def("SetInputShape", &Model::SetInputShape)
        .def("GetInfoString", &Model::GetInfoString, "get network information",
                py::arg("depth")    = 0,
                py::arg("columns")  = 70,
                py::arg("nest")     = 0)
        .def("GetParameters", &Model::GetParameters)
        .def("GetGradients", &Model::GetGradients)
        .def("Forward",  &Model::Forward, "Forward",
                py::arg("x_buf"),
                py::arg("train") = true)
        .def("Backward", &Model::Backward, "Backward");

    py::class_< SparseLayer, Model, std::shared_ptr<SparseLayer> >(m, "SparseLayer");

    py::class_< Sequential, Model, std::shared_ptr<Sequential> >(m, "Sequential")
        .def_static("Create", &Sequential::Create)
        .def("GetClassName",  &Sequential::GetClassName)
        .def("SetInputShape", &Sequential::SetInputShape)
        .def("Add",           &Sequential::Add);

    py::class_< Reduce, Model, std::shared_ptr<Reduce> >(m, "Reduce")
        .def_static("Create", &Reduce::CreateEx)
        .def("GetClassName",  &Reduce::GetClassName)
        .def("SetInputShape", &Reduce::SetInputShape);

    py::class_< BinaryModulation, Model, std::shared_ptr<BinaryModulation> >(m, "BinaryModulation")
        .def_static("Create", &BinaryModulation::CreateEx,
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
                py::arg("inference_input_range_hi")  = 1.0f)
        .def("GetClassName",  &BinaryModulation::GetClassName)
        .def("SetInputShape", &BinaryModulation::SetInputShape);

    py::class_< SparseLut6, SparseLayer, std::shared_ptr<SparseLut6> >(m, "SparseLut6")
        .def_static("Create", &SparseLut6::CreateEx, "create SparseLut6",
                py::arg("output_shape"),
                py::arg("batch_norm") = true,
                py::arg("connection") = "",
                py::arg("seed") = 1
            )
        .def("GetClassName", &SparseLut6::GetClassName)
        .def("SetInputShape", &SparseLut6::SetInputShape);


    // Loss Functions
    py::class_< LossFunction, std::shared_ptr<LossFunction> >(m, "LossFunction")
        .def("Clear",         &LossFunction::Clear)
        .def("GetLoss",       &LossFunction::GetLoss)
        .def("CalculateLoss", &LossFunction::CalculateLoss,
            py::arg("y_buf"),
            py::arg("t_buf"),
            py::arg("mini_batch_size"));

    py::class_< LossSoftmaxCrossEntropy, LossFunction, std::shared_ptr<LossSoftmaxCrossEntropy> >(m, "LossSoftmaxCrossEntropy")
        .def_static("Create", &LossSoftmaxCrossEntropy::Create);

    py::class_< LossMeanSquaredError, LossFunction, std::shared_ptr<LossMeanSquaredError> >(m, "LossMeanSquaredError")
        .def_static("Create", &LossMeanSquaredError::Create);


    // Metrics Functions
    py::class_< MetricsFunction, std::shared_ptr<MetricsFunction> >(m, "MetricsFunction")
        .def("Clear",            &MetricsFunction::Clear)
        .def("GetMetrics",       &MetricsFunction::GetMetrics)
        .def("CalculateMetrics", &MetricsFunction::CalculateMetrics);

    py::class_< MetricsCategoricalAccuracy, MetricsFunction, std::shared_ptr<MetricsCategoricalAccuracy> >(m, "MetricsCategoricalAccuracy")
        .def_static("Create", &MetricsCategoricalAccuracy::Create);

    py::class_< MetricsMeanSquaredError, MetricsFunction, std::shared_ptr<MetricsMeanSquaredError> >(m, "MetricsMeanSquaredError")
        .def_static("Create", &MetricsMeanSquaredError::Create);


    // Optimizer
    py::class_< Optimizer, std::shared_ptr<Optimizer> >(m, "Optimizer")
        .def("SetVariables", &Optimizer::SetVariables)
        .def("Update",       &Optimizer::Update);
    
    py::class_< OptimizerSgd, Optimizer, std::shared_ptr<OptimizerSgd> >(m, "OptimizerSgd")
        .def_static("Create", (std::shared_ptr<OptimizerSgd> (*)(float))&OptimizerSgd::Create, "create",
            py::arg("learning_rate") = 0.01f);
    
    py::class_< OptimizerAdam, Optimizer, std::shared_ptr<OptimizerAdam> >(m, "OptimizerAdam")
        .def_static("Create", &OptimizerAdam::CreateEx,
            py::arg("learning_rate") = 0.001f,
            py::arg("beta1")         = 0.9f,
            py::arg("beta2")         = 0.999f); 
    
    py::class_< OptimizerAdaGrad, Optimizer, std::shared_ptr<OptimizerAdaGrad> >(m, "OptimizerAdaGrad")
        .def_static("Create", (std::shared_ptr<OptimizerAdaGrad> (*)(float))&OptimizerAdaGrad::Create,
            py::arg("learning_rate") = 0.01f);


    // ValueGenerator
    py::class_< ValueGenerator, std::shared_ptr<ValueGenerator> >(m, "ValueGenerator");
    
    py::class_< NormalDistributionGenerator, ValueGenerator, std::shared_ptr<NormalDistributionGenerator> >(m, "NormalDistributionGenerator")
        .def_static("Create", &NormalDistributionGenerator::Create,
            py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f, py::arg("seed") = 1);
    
    py::class_< UniformDistributionGenerator, ValueGenerator, std::shared_ptr<UniformDistributionGenerator> >(m, "UniformDistributionGenerator")
        .def_static("Create", &UniformDistributionGenerator::Create,
            py::arg("a") = 0.0f, py::arg("b") = 1.0f, py::arg("seed") = 1);


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
        .def_static("Load", &LoadMnist::Load,
            py::arg("max_train") = -1,
            py::arg("max_test")  = -1,
            py::arg("num_class") = 10);
    
    // Runnner
    py::class_< Runner, std::shared_ptr<Runner> >(m, "Runner")
        .def_static("Create", &Runner::CreateEx,
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
        .def("Fitting", &Runner::Fitting,
            py::arg("td"), py::arg("epoch_size"), py::arg("batch_size"));

}


// end of file
