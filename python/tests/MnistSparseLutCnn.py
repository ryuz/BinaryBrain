# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import binarybrain as bb
from tqdm import tqdm
import time
import numpy as np



def main():
    epoch                    = 1
    mini_batch               = 32
    training_modulation_size = 15

    # load MNIST data
    td = bb.LoadMnist.load()

    # create layer
    layer_cnv0_sl0 = bb.SparseLut6.create([192])
    layer_cnv0_sl1 = bb.SparseLut6.create([32])
    layer_cnv1_sl0 = bb.SparseLut6.create([192])
    layer_cnv1_sl1 = bb.SparseLut6.create([32])
    layer_cnv2_sl0 = bb.SparseLut6.create([384])
    layer_cnv2_sl1 = bb.SparseLut6.create([64])
    layer_cnv3_sl0 = bb.SparseLut6.create([384])
    layer_cnv3_sl1 = bb.SparseLut6.create([64])
    layer_sl4      = bb.SparseLut6.create([420])
    layer_sl5      = bb.SparseLut6.create([70])
    
    # main network
    cnv0_sub = bb.Sequential.create()
    cnv0_sub.add(layer_cnv0_sl0)
    cnv0_sub.add(layer_cnv0_sl1)
    
    cnv1_sub = bb.Sequential.create()
    cnv1_sub.add(layer_cnv1_sl0)
    cnv1_sub.add(layer_cnv1_sl1)
    
    cnv2_sub = bb.Sequential.create()
    cnv2_sub.add(layer_cnv2_sl0)
    cnv2_sub.add(layer_cnv2_sl1)
    
    cnv3_sub = bb.Sequential.create()
    cnv3_sub.add(layer_cnv3_sl0)
    cnv3_sub.add(layer_cnv3_sl1)
    
    main_net = bb.Sequential.create()
    main_net.add(bb.LoweringConvolution.create(cnv0_sub, 3, 3))
    main_net.add(bb.LoweringConvolution.create(cnv1_sub, 3, 3))
    main_net.add(bb.MaxPooling.create(2, 2))
    main_net.add(bb.LoweringConvolution.create(cnv2_sub, 3, 3))
    main_net.add(bb.LoweringConvolution.create(cnv3_sub, 3, 3))
    main_net.add(bb.MaxPooling.create(2, 2))
    main_net.add(layer_sl4)
    main_net.add(layer_sl5)
    
    # wrapping with binary modulator
    net = bb.Sequential.create()
    net.add(bb.BinaryModulation.create(main_net, training_modulation_size=training_modulation_size))
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


if __name__ == '__main__':
    main()

