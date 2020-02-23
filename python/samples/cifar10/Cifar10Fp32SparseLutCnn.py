# coding: utf-8

import binarybrain as bb
import numpy as np


# options
epoch      = 64 # 16
mini_batch = 32
file_read  = True # False
file_write = True


# load data
td = bb.load_cifar10()

batch_size = len(td['x_train'])
print('batch_size =', batch_size)


# main network
cnv0_sub = bb.Sequential.create()
cnv0_sub.add(bb.StochasticBatchNormalization.create())
cnv0_sub.add(bb.StochasticLut6.create([512]))
cnv0_sub.add(bb.StochasticBatchNormalization.create())
cnv0_sub.add(bb.StochasticLut6.create([32*6]))
cnv0_sub.add(bb.StochasticBatchNormalization.create())
cnv0_sub.add(bb.StochasticLut6.create([32]))

cnv1_sub = bb.Sequential.create()
cnv1_sub.add(bb.StochasticBatchNormalization.create())
cnv1_sub.add(bb.StochasticLut6.create([512]))
cnv1_sub.add(bb.StochasticBatchNormalization.create())
cnv1_sub.add(bb.StochasticLut6.create([32*6]))
cnv1_sub.add(bb.StochasticBatchNormalization.create())
cnv1_sub.add(bb.StochasticLut6.create([32]))

cnv2_sub = bb.Sequential.create()
cnv2_sub.add(bb.StochasticBatchNormalization.create())
cnv2_sub.add(bb.StochasticLut6.create([512]))
cnv2_sub.add(bb.StochasticBatchNormalization.create())
cnv2_sub.add(bb.StochasticLut6.create([64*6]))
cnv2_sub.add(bb.StochasticBatchNormalization.create())
cnv2_sub.add(bb.StochasticLut6.create([64]))

cnv3_sub = bb.Sequential.create()
cnv3_sub.add(bb.StochasticBatchNormalization.create())
cnv3_sub.add(bb.StochasticLut6.create([512]))
cnv3_sub.add(bb.StochasticBatchNormalization.create())
cnv3_sub.add(bb.StochasticLut6.create([64*6*6]))
cnv3_sub.add(bb.StochasticBatchNormalization.create())
cnv3_sub.add(bb.StochasticLut6.create([64*6]))
cnv3_sub.add(bb.StochasticBatchNormalization.create())
cnv3_sub.add(bb.StochasticLut6.create([64]))

net = bb.Sequential.create()
net.add(bb.LoweringConvolution.create(cnv0_sub, 3, 3))
net.add(bb.LoweringConvolution.create(cnv1_sub, 3, 3))
net.add(bb.MaxPooling.create(2, 2))
net.add(bb.LoweringConvolution.create(cnv2_sub, 3, 3))
net.add(bb.LoweringConvolution.create(cnv3_sub, 3, 3))
net.add(bb.MaxPooling.create(2, 2))
net.add(bb.StochasticBatchNormalization.create())
net.add(bb.StochasticLut6.create([1024*6]))
net.add(bb.StochasticBatchNormalization.create())
net.add(bb.StochasticLut6.create([1024]))
net.add(bb.StochasticBatchNormalization.create())
net.add(bb.StochasticLut6.create([10*6*6*6]))
net.add(bb.StochasticBatchNormalization.create())
net.add(bb.StochasticLut6.create([10*6*6]))
net.add(bb.StochasticBatchNormalization.create())
net.add(bb.StochasticLut6.create([10*6]))
net.add(bb.StochasticBatchNormalization.create())
net.add(bb.StochasticLut6.create([10]))

net.set_input_shape(td['x_shape'])

# set no binary mode
net.send_command("binary false")
net.send_command("lut_binarize false")

# print model information
print(net.get_info())


# learning
print('\n[learning]')

loss      = bb.LossSoftmaxCrossEntropy.create()
metrics   = bb.MetricsCategoricalAccuracy.create()
optimizer = bb.OptimizerAdam.create()
optimizer.set_variables(net.get_parameters(), net.get_gradients())
data_augmentation = bb.image_data_augmentation(shift_range=5.0, rotation_range=10.0, scale_range=0.1, rate=0.8)

runner = bb.Runner(net, "cifar10-fp32-sparse-lut-cnn", loss, metrics, optimizer, data_augmentation=data_augmentation)
runner.fitting(td, epoch_size=epoch, mini_batch_size=mini_batch, file_read=file_read, file_write=file_write)
