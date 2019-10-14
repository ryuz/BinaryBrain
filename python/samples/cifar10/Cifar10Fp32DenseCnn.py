# coding: utf-8
import binarybrain as bb
import numpy as np



# option
epoch      = 16
mini_batch = 32
file_read  = False
file_write = True


# load data
td = bb.load_cifar10()

batch_size = len(td['x_train'])
print('batch_size =', batch_size)

# main network
cnv0_sub = bb.Sequential.create()
cnv0_sub.add(bb.DenseAffine.create([32]))
cnv0_sub.add(bb.BatchNormalization.create())
cnv0_sub.add(bb.ReLU.create())

cnv1_sub = bb.Sequential.create()
cnv1_sub.add(bb.DenseAffine.create([32]))
cnv1_sub.add(bb.BatchNormalization.create())
cnv1_sub.add(bb.ReLU.create())

cnv2_sub = bb.Sequential.create()
cnv2_sub.add(bb.DenseAffine.create([64]))
cnv2_sub.add(bb.BatchNormalization.create())
cnv2_sub.add(bb.ReLU.create())

cnv3_sub = bb.Sequential.create()
cnv3_sub.add(bb.DenseAffine.create([64]))
cnv3_sub.add(bb.BatchNormalization.create())
cnv3_sub.add(bb.ReLU.create())

net = bb.Sequential.create()
net.add(bb.LoweringConvolution.create(cnv0_sub, 3, 3))
net.add(bb.LoweringConvolution.create(cnv1_sub, 3, 3))
net.add(bb.MaxPooling.create(2, 2))
net.add(bb.LoweringConvolution.create(cnv2_sub, 3, 3))
net.add(bb.LoweringConvolution.create(cnv3_sub, 3, 3))
net.add(bb.MaxPooling.create(2, 2))
net.add(bb.DenseAffine.create([512]))
net.add(bb.BatchNormalization.create())
net.add(bb.ReLU.create())
net.add(bb.DenseAffine.create([10]))

net.set_input_shape(td['x_shape'])

# set no binary mode
net.send_command("binary false")

# print model information
print(net.get_info())

# learning
print('\n[learning]')

loss      = bb.LossSoftmaxCrossEntropy.create()
metrics   = bb.MetricsCategoricalAccuracy.create()
optimizer = bb.OptimizerAdam.create()
optimizer.set_variables(net.get_parameters(), net.get_gradients())

runner = bb.Runner(net, "cifar10-fp32-dense-cnn", loss, metrics, optimizer)
runner.fitting(td, epoch_size=epoch, mini_batch_size=mini_batch, file_read=file_read, file_write=file_write)
