# coding: utf-8

import sys, os
sys.path.append(os.pardir)

import binarybrain as bb
from tqdm import tqdm
import time
import numpy as np


epoch      = 3
mini_batch = 32


# load MNIST data
td = bb.LoadMnist.load()

# create layer
layer_sl0 = bb.SparseLut6.create([1024])
layer_sl1 = bb.SparseLut6.create([480])
layer_sl2 = bb.SparseLut6.create([70])

# create network
main_net = bb.Sequential.create()
main_net.add(layer_sl0)
main_net.add(layer_sl1)
main_net.add(layer_sl2)

# wrapping with binary modulator
net = bb.Sequential.create()
net.add(bb.BinaryModulation.create(main_net, training_modulation_size=15))
net.add(bb.Reduce.create(td.t_shape))
net.set_input_shape(td.x_shape)

print(net.get_info())

loss      = bb.LossSoftmaxCrossEntropy.create()
metrics   = bb.MetricsCategoricalAccuracy.create()
optimizer = bb.OptimizerAdam.create()

optimizer.set_variables(net.get_parameters(), net.get_gradients())

batch_size = len(td.x_train)
print('batch_size =', batch_size)


runner = bb.Runner(net, "mnist-mlp-sparse-lut6", loss, metrics, optimizer)
runner.fitting(td, epoch_size=3, mini_batch_size=16)

sys.exit(0)

if False:
    runner = bb.CRunner.create("mnist-mlp-sparse-lut6", net, loss, metrics, optimizer)
    runner.fitting(td, epoch_size=1, batch_size=16)

loss.clear()
metrics.clear()

x_train = td.x_train
t_train = td.t_train

x_buf = bb.FrameBuffer(bb.TYPE_FP32, 16, td.x_shape, False)
t_buf = bb.FrameBuffer(bb.TYPE_FP32, 16, td.t_shape, False)

for epoch_number in range(epoch):
    for index in tqdm(range(0, batch_size, mini_batch)):
        mini_batch_size = min(mini_batch, batch_size-index)
        
        x_buf.resize(bb.TYPE_FP32, mini_batch_size, td.x_shape)
        x_buf.set_data(x_train[index:index+mini_batch_size])
        
        y_buf = net.forward(x_buf)
        
        t_buf.resize(bb.TYPE_FP32, mini_batch_size, td.t_shape)
        t_buf.set_data(t_train[index:index+mini_batch_size])
        
        dy_buf = loss.calculate_loss(y_buf, t_buf, mini_batch_size)
        metrics.calculate_metrics(y_buf, t_buf)
        dx_buf = net.backward(dy_buf)
        
        optimizer.update()
    
    print('loss =', loss.get_loss())
    print('metrics =', metrics.get_metrics())
