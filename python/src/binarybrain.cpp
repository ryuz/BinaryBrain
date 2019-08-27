

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


namespace py = pybind11;
PYBIND11_MODULE(binarybrain, m) {
    m.doc() = "binarybrain plugin";

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

    py::class_< Tensor >(m, "Tensor");

    py::class_< FrameBuffer >(m, "FrameBuffer")
        .def(py::init< bool >(),
            py::arg("hostOnly") = false)
        .def(py::init< bb::index_t, bb::indices_t, int, bool>(),
            py::arg("data_type"),
            py::arg("frame_size"),
            py::arg("shape"),
            py::arg("hostOnly") = false)
        .def("resize",  (void (FrameBuffer::*)(bb::index_t, bb::indices_t, int))&bb::FrameBuffer::Resize,
                "resize",
                py::arg("frame_size"),
                py::arg("shape"),
                py::arg("data_type") = BB_TYPE_FP32)
        .def("get_range", &FrameBuffer::GetRange)
        .def("set_data",  (void (FrameBuffer::*)(std::vector< std::vector<float> > const &, bb::index_t))&FrameBuffer::SetVector<float>,
                "set data",
                py::arg("data"),
                py::arg("offset") = 0);

    py::class_< Variables, std::shared_ptr<Variables> >(m, "Variables");

    // Models
    py::class_< Model, std::shared_ptr<Model> >(m, "Model")
        .def("set_input_shape", &Model::SetInputShape)
        .def("get_info", &Model::GetInfoString, "get network information",
                py::arg("depth")    = 0,
                py::arg("columns")  = 70,
                py::arg("nest")     = 0)
        .def("get_parameters", &Model::GetParameters)
        .def("get_gradients", &Model::GetGradients)
        .def("forward",  &Model::Forward, "Forward",
                py::arg("x_buf"),
                py::arg("train") = true)
        .def("backward", &Model::Backward, "Backward");

    py::class_< SparseLayer, Model, std::shared_ptr<SparseLayer> >(m, "SparseLayer");

    py::class_< Sequential, Model, std::shared_ptr<Sequential> >(m, "Sequential")
        .def_static("create",   &Sequential::Create)
//      .def("get_class_name",  &Sequential::GetClassName)
//      .def("set_input_shape", &Sequential::SetInputShape)
        .def("add",             &Sequential::Add);

    py::class_< Reduce, Model, std::shared_ptr<Reduce> >(m, "Reduce")
        .def_static("create",   &Reduce::CreateEx)
//      .def("get_class_name",  &Reduce::GetClassName)
//      .def("set_input_shape", &Reduce::SetInputShape)
        ;

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
                py::arg("inference_input_range_hi")  = 1.0f)
//      .def("GetClassName",  &BinaryModulation::GetClassName)
//      .def("SetInputShape", &BinaryModulation::SetInputShape)
        ;

    py::class_< SparseLut6, SparseLayer, std::shared_ptr<SparseLut6> >(m, "SparseLut6")
        .def_static("create", &SparseLut6::CreateEx, "create SparseLut6",
                py::arg("output_shape"),
                py::arg("batch_norm") = true,
                py::arg("connection") = "",
                py::arg("seed") = 1
            )
//      .def("GetClassName", &SparseLut6::GetClassName)
//      .def("SetInputShape", &SparseLut6::SetInputShape)
        ;


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
        .def("clear",            &MetricsFunction::Clear)
        .def("get_metrics",       &MetricsFunction::GetMetrics)
        .def("calculate_metrics", &MetricsFunction::CalculateMetrics);

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

}


// end of file
