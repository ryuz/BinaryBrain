# coding: utf-8

import binarybrain as bb
import numpy as np


def main():
    epoch                     = 4
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
    layer_cnv0_sl0 = bb.SparseLut6Bit.create([192])
    layer_cnv0_sl1 = bb.SparseLut6Bit.create([32])

    layer_cnv1_sl0 = bb.SparseLut6Bit.create([1152])
    layer_cnv1_sl1 = bb.SparseLut6Bit.create([192])
    layer_cnv1_sl2 = bb.SparseLut6Bit.create([32])
    
    layer_cnv2_sl0 = bb.SparseLut6Bit.create([2304])
    layer_cnv2_sl1 = bb.SparseLut6Bit.create([384])
    layer_cnv2_sl2 = bb.SparseLut6Bit.create([64])
    
    layer_cnv3_sl0 = bb.SparseLut6Bit.create([2384])
    layer_cnv3_sl1 = bb.SparseLut6Bit.create([384])
    layer_cnv3_sl2 = bb.SparseLut6Bit.create([64])

    layer_sl4      = bb.SparseLut6Bit.create([18432])
    layer_sl5      = bb.SparseLut6Bit.create([3072])
    layer_sl6      = bb.SparseLut6Bit.create([512])

    layer_sl7      = bb.SparseLut6Bit.create([2160])
    layer_sl8      = bb.SparseLut6Bit.create([360])
    layer_sl9      = bb.SparseLut6Bit.create([60])
    layer_sl10     = bb.SparseLut6Bit.create([10])
    
    # main network
    cnv0_sub = bb.Sequential.create()
    cnv0_sub.add(layer_cnv0_sl0)
    cnv0_sub.add(layer_cnv0_sl1)
    
    cnv1_sub = bb.Sequential.create()
    cnv1_sub.add(layer_cnv1_sl0)
    cnv1_sub.add(layer_cnv1_sl1)
    cnv1_sub.add(layer_cnv1_sl2)
    
    cnv2_sub = bb.Sequential.create()
    cnv2_sub.add(layer_cnv2_sl0)
    cnv2_sub.add(layer_cnv2_sl1)
    cnv2_sub.add(layer_cnv2_sl2)
    
    cnv3_sub = bb.Sequential.create()
    cnv3_sub.add(layer_cnv3_sl0)
    cnv3_sub.add(layer_cnv3_sl1)
    cnv3_sub.add(layer_cnv3_sl2)
    
    main_net = bb.Sequential.create()
    main_net.add(bb.LoweringConvolutionBit.create(cnv0_sub, 3, 3))
    main_net.add(bb.LoweringConvolutionBit.create(cnv1_sub, 3, 3))
    main_net.add(bb.MaxPoolingBit.create(2, 2))
    main_net.add(bb.LoweringConvolutionBit.create(cnv2_sub, 3, 3))
    main_net.add(bb.LoweringConvolutionBit.create(cnv3_sub, 3, 3))
    main_net.add(bb.MaxPoolingBit.create(2, 2))
    main_net.add(layer_sl4)
    main_net.add(layer_sl5)
    main_net.add(layer_sl6)
    main_net.add(layer_sl7)
    main_net.add(layer_sl8)
    main_net.add(layer_sl9)
    main_net.add(layer_sl10)
    
    # wrapping with binary modulator
    net = bb.Sequential.create()
    net.add(bb.BinaryModulationBit.create(main_net, training_modulation_size=training_modulation_size))
    net.add(bb.Reduce.create(td['t_shape']))
    net.set_input_shape(td['x_shape'])
    
    # set binary mode
    net.send_command("binary true")

    # print model information
    print(net.get_info())
    
    
    # learning
    print('\n[learning]')
    
    loss      = bb.LossSoftmaxCrossEntropy.create()
    metrics   = bb.MetricsCategoricalAccuracy.create()
    optimizer = bb.OptimizerAdam.create()
    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    
    runner = bb.Runner(net, "cifar10-sparse-lut6-cnn", loss, metrics, optimizer)
    runner.fitting(td, epoch_size=epoch, mini_batch_size=mini_batch)
    
    
'''    
    ################################
    # convert to FPGA
    ################################
    
    print('\n[convert to Binary LUT]')
    
    # LUT-network
    layer_cnv0_bl0 = bb.BinaryLut6.create(layer_cnv0_sl0.get_output_shape())
    layer_cnv0_bl1 = bb.BinaryLut6.create(layer_cnv0_sl1.get_output_shape())
    layer_cnv1_bl0 = bb.BinaryLut6.create(layer_cnv1_sl0.get_output_shape())
    layer_cnv1_bl1 = bb.BinaryLut6.create(layer_cnv1_sl1.get_output_shape())
    layer_cnv2_bl0 = bb.BinaryLut6.create(layer_cnv2_sl0.get_output_shape())
    layer_cnv2_bl1 = bb.BinaryLut6.create(layer_cnv2_sl1.get_output_shape())
    layer_cnv3_bl0 = bb.BinaryLut6.create(layer_cnv3_sl0.get_output_shape())
    layer_cnv3_bl1 = bb.BinaryLut6.create(layer_cnv3_sl1.get_output_shape())
    layer_bl4      = bb.BinaryLut6.create(layer_sl4.get_output_shape())
    layer_bl5      = bb.BinaryLut6.create(layer_sl5.get_output_shape())
    
    cnv0_sub = bb.Sequential.create()
    cnv0_sub.add(layer_cnv0_bl0)
    cnv0_sub.add(layer_cnv0_bl1)
    
    cnv1_sub = bb.Sequential.create()
    cnv1_sub.add(layer_cnv1_bl0)
    cnv1_sub.add(layer_cnv1_bl1)
    
    cnv2_sub = bb.Sequential.create()
    cnv2_sub.add(layer_cnv2_bl0)
    cnv2_sub.add(layer_cnv2_bl1)
    
    cnv3_sub = bb.Sequential.create()
    cnv3_sub.add(layer_cnv3_bl0)
    cnv3_sub.add(layer_cnv3_bl1)
    
    cnv4_sub = bb.Sequential.create()
    cnv4_sub.add(layer_bl4)
    cnv4_sub.add(layer_bl5)
    
    cnv0 = bb.LoweringConvolution.create(cnv0_sub, 3, 3)
    cnv1 = bb.LoweringConvolution.create(cnv1_sub, 3, 3)
    pol0 = bb.MaxPooling.create(2, 2)
    
    cnv2 = bb.LoweringConvolution.create(cnv2_sub, 3, 3)
    cnv3 = bb.LoweringConvolution.create(cnv3_sub, 3, 3)
    pol1 = bb.MaxPooling.create(2, 2)
    
    cnv4 = bb.LoweringConvolution.create(cnv4_sub, 4, 4)
    
    lut_net = bb.Sequential.create()
    lut_net.add(cnv0)
    lut_net.add(cnv1)
    lut_net.add(pol0)
    lut_net.add(cnv2)
    lut_net.add(cnv3)
    lut_net.add(pol1)
    lut_net.add(cnv4)
    
    # evaluate network
    eval_net = bb.Sequential.create();
    eval_net.add(bb.BinaryModulation.create(lut_net, inference_modulation_size=inference_modulation_size))
    eval_net.add(bb.Reduce.create(td['t_shape']))
    
    # set input shape
    eval_net.set_input_shape(td['x_shape'])
    
    
    # parameter copy
    print('parameter copy to binary LUT-Network')
    layer_cnv0_bl0.import_parameter(layer_cnv0_sl0);
    layer_cnv0_bl1.import_parameter(layer_cnv0_sl1);
    layer_cnv1_bl0.import_parameter(layer_cnv1_sl0);
    layer_cnv1_bl1.import_parameter(layer_cnv1_sl1);
    layer_cnv2_bl0.import_parameter(layer_cnv2_sl0);
    layer_cnv2_bl1.import_parameter(layer_cnv2_sl1);
    layer_cnv3_bl0.import_parameter(layer_cnv3_sl0);
    layer_cnv3_bl1.import_parameter(layer_cnv3_sl1);
    layer_bl4.import_parameter(layer_sl4);
    layer_bl5.import_parameter(layer_sl5);
    
    # evaluate network
    print('evaluate LUT-Network')
    lut_runner = bb.Runner(eval_net, "mnist-binary-lut6-cnn",
                    bb.LossSoftmaxCrossEntropy.create(),
                    bb.MetricsCategoricalAccuracy.create())
    lut_runner.evaluation(td, mini_batch_size=mini_batch)
    
    # write Verilog
    print('write verilog file')
    with open('MnistLutCnn.v', 'w') as f:
        f.write('`timescale 1ns / 1ps\n\n')
        f.write(bb.make_verilog_axi4s_from_lut_cnn('MnistLutCnnCnv0', [cnv0, cnv1, pol0]))
        f.write(bb.make_verilog_axi4s_from_lut_cnn('MnistLutCnnCnv1', [cnv2, cnv3, pol1]))
        f.write(bb.make_verilog_axi4s_from_lut_cnn('MnistLutCnnCnv2', [cnv4]))
'''


if __name__ == '__main__':
    main()

