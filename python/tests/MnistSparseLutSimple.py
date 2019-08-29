# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import binarybrain as bb
import numpy as np

def main():
    # config
    epoch                     = 4
    mini_batch                = 32
    training_modulation_size  = 3
    inference_modulation_size = 3
    
    
    # load MNIST data
    td = bb.LoadMnist.load()
    
    batch_size = len(td.x_train)
    print('batch_size =', batch_size)
    
    
    
    ############################
    # Learning
    ############################
    
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
    net.add(bb.BinaryModulation.create(main_net, training_modulation_size=training_modulation_size))
    net.add(bb.Reduce.create(td.t_shape))
    net.set_input_shape(td.x_shape)
    
    # print model information
    print(net.get_info())
    
    # learning
    print('\n[learning]')
    loss      = bb.LossSoftmaxCrossEntropy.create()
    metrics   = bb.MetricsCategoricalAccuracy.create()
    optimizer = bb.OptimizerAdam.create()
    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    
    runner = bb.Runner(net, "mnist-sparse-lut6-simple", loss, metrics, optimizer)
    runner.fitting(td, epoch_size=epoch, mini_batch_size=mini_batch)
    
    
    ################################
    # convert to FPGA
    ################################
    
    print('\n[convert to Binary LUT]')
    
    # LUT-network
    layer_bl0 = bb.BinaryLut6.create(layer_sl0.get_output_shape())
    layer_bl1 = bb.BinaryLut6.create(layer_sl1.get_output_shape())
    layer_bl2 = bb.BinaryLut6.create(layer_sl2.get_output_shape())
    
    lut_net = bb.Sequential.create()
    lut_net.add(layer_bl0)
    lut_net.add(layer_bl1)
    lut_net.add(layer_bl2)
    
    # evaluate network
    eval_net = bb.Sequential.create()
    eval_net.add(bb.BinaryModulation.create(lut_net, inference_modulation_size=inference_modulation_size))
    eval_net.add(bb.Reduce.create(td.t_shape))
    
    # set input shape
    eval_net.set_input_shape(td.x_shape)
    
    # parameter copy
    print('parameter copy to binary LUT-Network')
    layer_bl0.import_parameter(layer_sl0)
    layer_bl1.import_parameter(layer_sl1)
    layer_bl2.import_parameter(layer_sl2)
    
    # evaluate network
    print('evaluate LUT-Network')
    lut_runner = bb.Runner(eval_net, "mnist-binary-lut6-simple",
                    bb.LossSoftmaxCrossEntropy.create(),
                    bb.MetricsCategoricalAccuracy.create())
    lut_runner.evaluation(td, mini_batch_size=mini_batch)
    
    # write Verilog
    print('write verilog file')
    with open('MnistLutSimple.v', 'w') as f:
        f.write('`timescale 1ns / 1ps\n\n')
        f.write(bb.get_verilog_from_lut('MnistLutSimple', [layer_bl0, layer_bl1, layer_bl2]))


if __name__ == '__main__':
    main()

