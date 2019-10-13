# coding: utf-8

import binarybrain as bb
import numpy as np


def main():
    binary_mode               = False
    epoch                     = 8
    mini_batch                = 32
    training_modulation_size  = 3
    inference_modulation_size = 3
    
    # load data
    td = bb.load_cifar10()
    
    batch_size = len(td['x_train'])
    print('batch_size =', batch_size)
    
    
    
    ############################
    # Learning
    ############################
    
    # create layer
    layer_cnv0_sl0 = bb.SparseLut6.create([192])
    layer_cnv0_sl1 = bb.SparseLut6.create([32])
    
    layer_cnv1_sl0 = bb.SparseLut6.create([192])
    layer_cnv1_sl1 = bb.SparseLut6.create([32])
    
    layer_cnv2_sl0 = bb.SparseLut6.create([384])
    layer_cnv2_sl1 = bb.SparseLut6.create([64])
    
    layer_cnv3_sl0 = bb.SparseLut6.create([384])
    layer_cnv3_sl1 = bb.SparseLut6.create([64])
    
    layer_sl4      = bb.SparseLut6.create([3072])
    layer_sl5      = bb.SparseLut6.create([512])
    
    layer_sl6      = bb.SparseLut6.create([360])
    layer_sl7      = bb.SparseLut6.create([60])
    layer_sl8      = bb.SparseLut6.create([10])
    
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
    main_net.add(layer_sl6)
    main_net.add(layer_sl7)
    main_net.add(layer_sl8)
    
    
    # wrapping with binary modulator
    net = bb.Sequential.create()
    net.add(bb.BinaryModulation.create(main_net, training_modulation_size=training_modulation_size))
    net.add(bb.Reduce.create(td['t_shape']))
    net.set_input_shape(td['x_shape'])
    
    # set no binary mode
    if binary_mode:
        net.send_command("binary true")
    else:
        net.send_command("binary false")
    
    # print model information
    print(net.get_info())
    
    
    # learning
    print('\n[learning]')
    
    loss      = bb.LossSoftmaxCrossEntropy.create()
    metrics   = bb.MetricsCategoricalAccuracy.create()
    optimizer = bb.OptimizerAdam.create()
    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    
    runner = bb.Runner(net, "cifar10-sparse-lut6-cnn", loss, metrics, optimizer)
    runner.fitting(td, epoch_size=epoch, mini_batch_size=mini_batch, file_read=True, file_write=True)
    
    


if __name__ == '__main__':
    main()

