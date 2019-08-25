from binarybrain import binarybrain as bb
from tqdm import tqdm
import time
import numpy as np

epoch      = 3
mini_batch = 32


# load MNIST data
td = bb.LoadMnist.Load()

# create layer
layer_sl0 = bb.SparseLut6.Create([1024])
layer_sl1 = bb.SparseLut6.Create([480])
layer_sl2 = bb.SparseLut6.Create([70])

# create network
main_net = bb.Sequential.Create()
main_net.Add(layer_sl0)
main_net.Add(layer_sl1)
main_net.Add(layer_sl2)

# wrapping with binary modulator
net = bb.Sequential.Create()
net.Add(bb.BinaryModulation.Create(main_net, training_modulation_size=15))
net.Add(bb.Reduce.Create(td.t_shape))
net.SetInputShape(td.x_shape)

#print(net.GetInfoString())

loss      = bb.LossSoftmaxCrossEntropy.Create()
metrics   = bb.MetricsCategoricalAccuracy.Create()
optimizer = bb.OptimizerAdam.Create()

optimizer.SetVariables(net.GetParameters(), net.GetGradients())

loss.Clear()
metrics.Clear()

batch_size = len(td.x_train)
print('batch_size =', batch_size)

if False:
    runner    = bb.Runner.Create("mnist-mlp-sparse-lut6", net, loss, metrics, optimizer)
    runner.Fitting(td, epoch_size=1, batch_size=16)

x_train = td.x_train
t_train = td.t_train

x_buf = bb.FrameBuffer(bb.fp32, 16, td.x_shape, False)
t_buf = bb.FrameBuffer(bb.fp32, 16, td.t_shape, False)

for epoch_number in range(epoch):
    for index in tqdm(range(0, batch_size, mini_batch)):
        mini_batch_size = min(mini_batch, batch_size-index)
        
        x_buf.Resize(bb.fp32, mini_batch_size, td.x_shape)
        x_buf.SetVector(x_train[index:index+mini_batch_size])
        
        y_buf = net.Forward(x_buf)
        
        t_buf.Resize(bb.fp32, mini_batch_size, td.t_shape)
        t_buf.SetVector(t_train[index:index+mini_batch_size])
        
        dy_buf = loss.CalculateLoss(y_buf, t_buf, mini_batch_size)
        metrics.CalculateMetrics(y_buf, t_buf)
        dx_buf = net.Backward(dy_buf)
        
        optimizer.Update()
    
    print('loss =', loss.GetLoss())
    print('metrics =', metrics.GetMetrics())
