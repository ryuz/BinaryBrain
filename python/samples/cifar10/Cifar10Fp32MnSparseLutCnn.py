# coding: utf-8

import binarybrain as bb
import numpy as np


# options
epoch       = 16
mini_batch  = 16


# load data
td = bb.load_cifar10()

batch_size = len(td['x_train'])
print('batch_size =', batch_size)


# main network
#cnv0_sub = bb.Sequential.create()
#cnv0_sub.add(bb.StochasticBatchNormalization.create())
#cnv0_sub.add(bb.StochasticLut6.create([32*6]))
#cnv0_sub.add(bb.StochasticBatchNormalization.create())
#cnv0_sub.add(bb.StochasticLut6.create([32]))
#
#cnv1_sub = bb.Sequential.create()
#cnv1_sub.add(bb.StochasticBatchNormalization.create())
#cnv1_sub.add(bb.StochasticLut6.create([32*6]))
#cnv1_sub.add(bb.StochasticBatchNormalization.create())
#cnv1_sub.add(bb.StochasticLut6.create([32]))
#
#cnv2_sub = bb.Sequential.create()
#cnv2_sub.add(bb.StochasticBatchNormalization.create())
#cnv2_sub.add(bb.StochasticLut6.create([32*6]))
#cnv2_sub.add(bb.StochasticBatchNormalization.create())
#cnv2_sub.add(bb.StochasticLut6.create([32]))

#cnv3_sub = bb.Sequential.create()
#cnv3_sub.add(bb.StochasticBatchNormalization.create())
#cnv3_sub.add(bb.StochasticLut6.create([32*6]))
#cnv3_sub.add(bb.StochasticBatchNormalization.create())
#cnv3_sub.add(bb.StochasticLut6.create([32]))

cnv0_sub0 = bb.Sequential.create()
cnv0_sub0.add(bb.StochasticBatchNormalization.create())
cnv0_sub0.add(bb.StochasticLut6.create([1, 1, 32*6], 'pointwise'))
cnv0_sub0.add(bb.StochasticBatchNormalization.create())
cnv0_sub0.add(bb.StochasticLut6.create([1, 1, 32], 'pointwise'))
cnv0_sub1 = bb.Sequential.create()
cnv0_sub1.add(bb.StochasticBatchNormalization.create())
cnv0_sub1.add(bb.StochasticLut6.create([6, 1, 32], 'depthwise'))
cnv0_sub1.add(bb.StochasticBatchNormalization.create())
cnv0_sub1.add(bb.StochasticLut6.create([1, 1, 32], 'depthwise'))

cnv1_sub0 = bb.Sequential.create()
cnv1_sub0.add(bb.StochasticBatchNormalization.create())
cnv1_sub0.add(bb.StochasticLut6.create([1, 1, 32*6], 'pointwise'))
cnv1_sub0.add(bb.StochasticBatchNormalization.create())
cnv1_sub0.add(bb.StochasticLut6.create([1, 1, 32], 'pointwise'))
cnv1_sub1 = bb.Sequential.create()
cnv1_sub1.add(bb.StochasticBatchNormalization.create())
cnv1_sub1.add(bb.StochasticLut6.create([6, 1, 32], 'depthwise'))
cnv1_sub1.add(bb.StochasticBatchNormalization.create())
cnv1_sub1.add(bb.StochasticLut6.create([1, 1, 32], 'depthwise'))
cnv1_sub2 = bb.Sequential.create()
cnv1_sub2.add(bb.StochasticBatchNormalization.create())
cnv1_sub2.add(bb.StochasticLut6.create([1, 1, 32*6], 'pointwise'))
cnv1_sub2.add(bb.StochasticBatchNormalization.create())
cnv1_sub2.add(bb.StochasticLut6.create([1, 1, 32], 'pointwise'))

cnv2_sub0 = bb.Sequential.create()
cnv2_sub0.add(bb.StochasticBatchNormalization.create())
cnv2_sub0.add(bb.StochasticLut6.create([1, 1, 64*6], 'pointwise'))
cnv2_sub0.add(bb.StochasticBatchNormalization.create())
cnv2_sub0.add(bb.StochasticLut6.create([1, 1, 64], 'pointwise'))
cnv2_sub1 = bb.Sequential.create()
cnv2_sub1.add(bb.StochasticBatchNormalization.create())
cnv2_sub1.add(bb.StochasticLut6.create([6, 1, 64], 'depthwise'))
cnv2_sub1.add(bb.StochasticBatchNormalization.create())
cnv2_sub1.add(bb.StochasticLut6.create([1, 1, 64], 'depthwise'))

cnv3_sub0 = bb.Sequential.create()
cnv3_sub0.add(bb.StochasticBatchNormalization.create())
cnv3_sub0.add(bb.StochasticLut6.create([1, 1, 64*6], 'pointwise'))
cnv3_sub0.add(bb.StochasticBatchNormalization.create())
cnv3_sub0.add(bb.StochasticLut6.create([1, 1, 64], 'pointwise'))
cnv3_sub1 = bb.Sequential.create()
cnv3_sub1.add(bb.StochasticBatchNormalization.create())
cnv3_sub1.add(bb.StochasticLut6.create([6, 1, 64], 'depthwise'))
cnv3_sub1.add(bb.StochasticBatchNormalization.create())
cnv3_sub1.add(bb.StochasticLut6.create([1, 1, 64], 'depthwise'))
cnv3_sub2 = bb.Sequential.create()
cnv3_sub2.add(bb.StochasticBatchNormalization.create())
cnv3_sub2.add(bb.StochasticLut6.create([1, 1, 64*6], 'pointwise'))
cnv3_sub2.add(bb.StochasticBatchNormalization.create())
cnv3_sub2.add(bb.StochasticLut6.create([1, 1, 64], 'pointwise'))

net = bb.Sequential.create()
#net.add(bb.LoweringConvolution.create(cnv0_sub, 3, 3))
net.add(bb.LoweringConvolution.create(cnv0_sub0, 1, 1))
net.add(bb.LoweringConvolution.create(cnv0_sub1, 3, 3))
#net.add(bb.LoweringConvolution.create(cnv1_sub, 1, 1))
net.add(bb.LoweringConvolution.create(cnv1_sub0, 1, 1))
net.add(bb.LoweringConvolution.create(cnv1_sub1, 3, 3))
net.add(bb.LoweringConvolution.create(cnv1_sub2, 1, 1))
net.add(bb.MaxPooling.create(2, 2))
#net.add(bb.LoweringConvolution.create(cnv2_sub, 1, 1))
net.add(bb.LoweringConvolution.create(cnv2_sub0, 1, 1))
net.add(bb.LoweringConvolution.create(cnv2_sub1, 3, 3))

#net.add(bb.LoweringConvolution.create(cnv3_sub, 1, 1))
net.add(bb.LoweringConvolution.create(cnv3_sub0, 1, 1))
net.add(bb.LoweringConvolution.create(cnv3_sub1, 3, 3))
net.add(bb.LoweringConvolution.create(cnv3_sub2, 1, 1))

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

runner = bb.Runner(net, "cifar10-fp32-mn-sparse-lut-cnn", loss, metrics, optimizer)
runner.fitting(td, epoch_size=epoch, mini_batch_size=mini_batch, file_read=True, file_write=True)
