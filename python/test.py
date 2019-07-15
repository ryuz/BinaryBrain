import binarybrain as bb


# load MNIST data
td = bb.LoadMnist.Load()

# create layer
layer_sl0 = bb.SparseLut6.Create([1024])
layer_sl1 = bb.SparseLut6.Create([480])
layer_sl2 = bb.SparseLut6.Create([70])

# create network
main_net = bb.Sequential.Create();
main_net.Add(layer_sl0)
main_net.Add(layer_sl1)
main_net.Add(layer_sl2)

# wrapping with binary modulator
net = bb.Sequential.Create()
net.Add(bb.BinaryModulation.Create(main_net, training_modulation_size=15))
net.Add(bb.Reduce.Create(td.t_shape))
net.SetInputShape(td.x_shape)

loss      = bb.LossSoftmaxCrossEntropy.Create()
metrics   = bb.MetricsCategoricalAccuracy.Create()
optimizer = bb.OptimizerAdam.Create()
runner    = bb.Runner.Create("mnist-mlp-sparse-lut6", net, loss, metrics, optimizer)

runner.Fitting(td, epoch_size=16, batch_size=16)

