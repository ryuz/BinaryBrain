import binarybrain as bb
#from binarybrain import TestA


td = bb.LoadMnist.Load()

layer_sl0 = bb.SparseLut6.Create([360]);
layer_sl1 = bb.SparseLut6.Create([60]);
layer_sl2 = bb.SparseLut6.Create([10]);

main_net = bb.Sequential.Create();
main_net.Add(layer_sl0);
main_net.Add(layer_sl1);
main_net.Add(layer_sl2);

main_net.SetInputShape(td.x_shape);

loss      = bb.LossSoftmaxCrossEntropy.Create()
metrics   = bb.MetricsCategoricalAccuracy.Create()
optimizer = bb.OptimizerAdam.Create()
runner = bb.Runner.Create("MNIST MLP", main_net, loss, metrics, optimizer)

runner.Fitting(td, 16, 16)

