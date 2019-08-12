

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "bb/Sequential.h"
#include "bb/SparseLutN.h"
#include "bb/SparseLutDiscreteN.h"
#include "bb/BinaryLutN.h"
#include "bb/Reduce.h"
#include "bb/BinaryModulation.h"

#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"

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


using Model                        = bb::Model;
using SparseLayer                  = bb::SparseLayer;
using Sequential                   = bb::Sequential;
using SparseLut6                   = bb::SparseLutN<6, float>;
using Reduce                       = bb::Reduce<float, float>; 
using BinaryModulation             = bb::BinaryModulation<float, float>;

using LossFunction                 = bb::LossFunction;
using LossSoftmaxCrossEntropy      = bb::LossSoftmaxCrossEntropy<float>;

using MetricsFunction              = bb::MetricsFunction;
using MetricsCategoricalAccuracy   = bb::MetricsCategoricalAccuracy<float>;

using Optimizer                    = bb::Optimizer;
using OptimizerAdam                = bb::OptimizerAdam<float>;

using ValueGenerator               = bb::ValueGenerator<float>;
using NormalDistributionGenerator  = bb::NormalDistributionGenerator<float>;
using UniformDistributionGenerator = bb::UniformDistributionGenerator<float>;

using TrainData                    = bb::TrainData<float>;
using Runner                       = bb::Runner<float>;
using LoadMnist                    = bb::LoadMnist<float>;


namespace py = pybind11;
PYBIND11_MODULE(binarybrain, m) {
    m.doc() = "binarybrain plugin";
    

    py::class_< Model, std::shared_ptr<Model> >(m, "Model")
        .def("SetInputShape", &Model::SetInputShape);

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
//      .def("PrintInfo",    &SparseLut6::PrintInfo, py::arg("depth")=0, py::arg("os")=std::cout, py::arg("columns")=70, py::arg("nest")=0)


    py::class_< LossFunction, std::shared_ptr<LossFunction> >(m, "LossFunction");
    py::class_< LossSoftmaxCrossEntropy, LossFunction, std::shared_ptr<LossSoftmaxCrossEntropy> >(m, "LossSoftmaxCrossEntropy")
        .def_static("Create", &LossSoftmaxCrossEntropy::Create);

    py::class_< MetricsFunction, std::shared_ptr<MetricsFunction> >(m, "MetricsFunction");
    py::class_< MetricsCategoricalAccuracy, MetricsFunction, std::shared_ptr<MetricsCategoricalAccuracy> >(m, "MetricsCategoricalAccuracy")
        .def_static("Create", &MetricsCategoricalAccuracy::Create);

    py::class_< Optimizer, std::shared_ptr<Optimizer> >(m, "Optimizer");
    py::class_< OptimizerAdam, Optimizer, std::shared_ptr<OptimizerAdam> >(m, "OptimizerAdam")
        .def_static("Create", &OptimizerAdam::CreateEx,
            py::arg("learning_rate") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f); 


    py::class_< ValueGenerator, std::shared_ptr<ValueGenerator> >(m, "ValueGenerator");
    
    py::class_< NormalDistributionGenerator, ValueGenerator, std::shared_ptr<NormalDistributionGenerator> >(m, "NormalDistributionGenerator")
        .def_static("Create", &NormalDistributionGenerator::Create,
            py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f, py::arg("seed") = 1);
    
    py::class_< UniformDistributionGenerator, ValueGenerator, std::shared_ptr<UniformDistributionGenerator> >(m, "UniformDistributionGenerator")
        .def_static("Create", &UniformDistributionGenerator::Create,
            py::arg("a") = 0.0f, py::arg("b") = 1.0f, py::arg("seed") = 1);



    py::class_< TrainData >(m, "TrainData")
        .def_readwrite("x_shape", &TrainData::x_shape)
        .def_readwrite("t_shape", &TrainData::t_shape)
        .def_readwrite("x_train", &TrainData::x_train)
        .def_readwrite("t_train", &TrainData::t_train)
        .def_readwrite("x_test",  &TrainData::x_test)
        .def_readwrite("t_test",  &TrainData::t_test)
        .def("empty", &TrainData::empty);


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

    py::class_< LoadMnist >(m, "LoadMnist")
        .def_static("Load", &LoadMnist::Load, py::arg("num_class") = 10, py::arg("max_train") = -1, py::arg("max_test") = -1);

}


// end of file
