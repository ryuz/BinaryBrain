# coding: utf-8

import binarybrain as bb
import numpy as np
import os
import sys
from tqdm import tqdm
from collections import OrderedDict


def layer_distillation(x, x_shape, target_net, ref_net, pre_net):
    x_buf = bb.FrameBuffer()
    t_buf = bb.FrameBuffer()

    print(ref_net.get_input_shape())

    target_net.set_input_shape(ref_net.get_input_shape())
    target_net.send_command("binary true")
    
    pre_net.set_input_shape(x_shape)

    batch_size = len(x)
    max_batch_size = 32
    leave = True

    loss = bb.LossMeanSquaredError.create()
    optimizer = bb.OptimizerAdam.create()
    optimizer.set_variables(target_net.get_parameters(), target_net.get_gradients())

#   x_shape = td['x_shape']
#   x_shape = td['x_shape']
    
    for epoch in range(8):
    #   for index in tqdm(range(0, batch_size, max_batch_size)):
        loss.clear()
        with tqdm(range(0, batch_size, max_batch_size), leave=leave) as pbar:
            for index in pbar:
                # calc mini_batch_size
                mini_batch_size = min(max_batch_size, batch_size-index)
                
                # setup x
                x_buf.resize(mini_batch_size, x_shape)
                x_buf.set_data(x[index:index+mini_batch_size])
                
                # forward
                x_buf = pre_net.forward(x_buf, False)
    #            print('\n')
    #            print(x_buf.get_node_shape())
    #            print(ref_net.get_input_shape())
    #            print(target_net.get_input_shape())

                t_buf = ref_net.forward(x_buf, False)
                y_buf = target_net.forward(x_buf, True)
                
                # calc loss
                dy_buf = loss.calculate_loss(y_buf, t_buf, mini_batch_size)

                # backward
                target_net.backward(dy_buf)

                # update
                optimizer.update()
                
                # print progress
                dict = OrderedDict()
                dict['loss'] = loss.get_loss()
                if len(dict) > 0:
                    pbar.set_postfix(dict)


def main():
    epoch                     = 1
    mini_batch                = 32
    training_modulation_size  = 8
    inference_modulation_size = 8
    
    # load MNIST data
    td = bb.load_mnist()
    
    batch_size = len(td['x_train'])
    print('batch_size =', batch_size)
    
    
    
    ############################
    # Learning
    ############################
    
    # create layer
    layer0_affine  = bb.DenseAffine.create([32])
    layer0_norm    = bb.BatchNormalization.create()
    layer0_bin     = bb.Binarize.create()
    layer1_affine  = bb.DenseAffine.create([32])
    layer1_norm    = bb.BatchNormalization.create()
    layer1_bin     = bb.Binarize.create()
    layer2_affine  = bb.DenseAffine.create([64])
    layer2_norm    = bb.BatchNormalization.create()
    layer2_bin     = bb.Binarize.create()
    layer3_affine  = bb.DenseAffine.create([64])
    layer3_norm    = bb.BatchNormalization.create()
    layer3_bin     = bb.Binarize.create()
    layer4_affine  = bb.DenseAffine.create([512])
    layer4_norm    = bb.BatchNormalization.create()
    layer4_bin     = bb.Binarize.create()
    layer5_affine  = bb.DenseAffine.create([10])
    layer5_norm    = bb.BatchNormalization.create()
    layer5_bin     = bb.Binarize.create()

    # main network
    cnv0_sub = bb.Sequential.create()
    cnv0_sub.add(layer0_affine)
    cnv0_sub.add(layer0_norm)
    cnv0_sub.add(layer0_bin)
    layer0_cnv = bb.LoweringConvolution.create(cnv0_sub, 3, 3)

    cnv1_sub = bb.Sequential.create()
    cnv1_sub.add(layer1_affine)
    cnv1_sub.add(layer1_norm)
    cnv1_sub.add(layer1_bin)
    layer1_cnv = bb.LoweringConvolution.create(cnv1_sub, 3, 3)
    
    cnv2_sub = bb.Sequential.create()
    cnv2_sub.add(layer2_affine)
    cnv2_sub.add(layer2_norm)
    cnv2_sub.add(layer2_bin)
    layer2_cnv = bb.LoweringConvolution.create(cnv2_sub, 3, 3)
    
    cnv3_sub = bb.Sequential.create()
    cnv3_sub.add(layer3_affine)
    cnv3_sub.add(layer3_norm)
    cnv3_sub.add(layer3_bin)
    layer3_cnv = bb.LoweringConvolution.create(cnv3_sub, 3, 3)
    


    main_net = bb.Sequential.create()
    main_net.add(layer0_cnv)
    main_net.add(layer1_cnv)
    main_net.add(bb.MaxPooling.create(2, 2))
    main_net.add(layer2_cnv)
    main_net.add(layer3_cnv)
    main_net.add(bb.MaxPooling.create(2, 2))
    main_net.add(layer4_affine)
    main_net.add(layer4_norm)
    main_net.add(layer4_bin)
    main_net.add(layer5_affine)
    main_net.add(layer5_norm)
    main_net.add(layer5_bin)
    
    # wrapping with binary modulator
    net = bb.Sequential.create()
    net.add(bb.BinaryModulation.create(main_net, training_modulation_size=training_modulation_size, inference_modulation_size=inference_modulation_size))
    net.set_input_shape(td['x_shape'])
    
    net.send_command("binary true")

    # print model information
    print(net.get_info())
        
    # learning
    print('\n[learning]')
    
    loss      = bb.LossSoftmaxCrossEntropy.create()
    metrics   = bb.MetricsCategoricalAccuracy.create()
    optimizer = bb.OptimizerAdam.create()
    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    
    runner = bb.Runner(net, "mnist-dense-cnn-binary", loss, metrics, optimizer)
#   runner.fitting(td, epoch_size=epoch, mini_batch_size=mini_batch, file_write=True, file_read=True)
    runner.fitting(td, epoch_size=0, mini_batch_size=mini_batch, file_write=False, file_read=True)

    net_path = 'mnist-dense-cnn-binary'
    os.makedirs(net_path, exist_ok=True)


    if False:
        layer0_affine.load_json(os.path.join(net_path, 'layer0_affine.json'))
        layer1_affine.load_json(os.path.join(net_path, 'layer1_affine.json'))
        layer2_affine.load_json(os.path.join(net_path, 'layer2_affine.json'))
        layer3_affine.load_json(os.path.join(net_path, 'layer3_affine.json'))
        layer4_affine.load_json(os.path.join(net_path, 'layer4_affine.json'))
        layer5_affine.load_json(os.path.join(net_path, 'layer5_affine.json'))

        layer0_norm.load_json(os.path.join(net_path, 'layer0_norm.json'))
        layer1_norm.load_json(os.path.join(net_path, 'layer1_norm.json'))
        layer2_norm.load_json(os.path.join(net_path, 'layer2_norm.json'))
        layer3_norm.load_json(os.path.join(net_path, 'layer3_norm.json'))
        layer4_norm.load_json(os.path.join(net_path, 'layer4_norm.json'))
        layer5_norm.load_json(os.path.join(net_path, 'layer5_norm.json'))

#   runner.fitting(td, epoch_size=0, mini_batch_size=mini_batch, file_write=False, file_read=True, init_eval=True)

    if True:
        layer0_affine.save_json(os.path.join(net_path, 'layer0_affine.json'))
        layer1_affine.save_json(os.path.join(net_path, 'layer1_affine.json'))
        layer2_affine.save_json(os.path.join(net_path, 'layer2_affine.json'))
        layer3_affine.save_json(os.path.join(net_path, 'layer3_affine.json'))
        layer4_affine.save_json(os.path.join(net_path, 'layer4_affine.json'))
        layer5_affine.save_json(os.path.join(net_path, 'layer5_affine.json'))

        layer0_norm.save_json(os.path.join(net_path, 'layer0_norm.json'))
        layer1_norm.save_json(os.path.join(net_path, 'layer1_norm.json'))
        layer2_norm.save_json(os.path.join(net_path, 'layer2_norm.json'))
        layer3_norm.save_json(os.path.join(net_path, 'layer3_norm.json'))
        layer4_norm.save_json(os.path.join(net_path, 'layer4_norm.json'))
        layer5_norm.save_json(os.path.join(net_path, 'layer5_norm.json'))
    

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
    target_cnv0_sub = bb.Sequential.create()
    target_cnv0_sub.add(layer_cnv0_sl0)
    target_cnv0_sub.add(layer_cnv0_sl1)
    target_cnv0 = bb.LoweringConvolution.create(target_cnv0_sub, 3, 3)

    target_cnv1_sub = bb.Sequential.create()
    target_cnv1_sub.add(layer_cnv1_sl0)
    target_cnv1_sub.add(layer_cnv1_sl1)
    target_cnv1 = bb.LoweringConvolution.create(target_cnv1_sub, 3, 3)

    target_cnv2_sub = bb.Sequential.create()
    target_cnv2_sub.add(layer_cnv2_sl0)
    target_cnv2_sub.add(layer_cnv2_sl1)
    target_cnv2 = bb.LoweringConvolution.create(target_cnv2_sub, 3, 3)

    target_cnv3_sub = bb.Sequential.create()
    target_cnv3_sub.add(layer_cnv3_sl0)
    target_cnv3_sub.add(layer_cnv3_sl1)
    target_cnv3 = bb.LoweringConvolution.create(target_cnv3_sub, 3, 3)
    
    pre_net = bb.Sequential.create()
    pre_net.add(bb.RealToBinary.create(8, framewise=True))

    if False:
        layer_distillation(td['x_train'], td['x_shape'], target_cnv0, layer0_cnv, pre_net)
        layer_cnv0_sl0.save_json(os.path.join(net_path, 'layer_cnv0_sl0.json'))
        layer_cnv0_sl0.save_json(os.path.join(net_path, 'layer_cnv0_sl1.json'))

    pre_net.add(layer0_cnv)

    if True:
        layer_distillation(td['x_train'], td['x_shape'], target_cnv1, layer1_cnv, pre_net)
        layer_cnv1_sl0.save_json(os.path.join(net_path, 'layer_cnv1_sl0_.json'))
        layer_cnv1_sl1.save_json(os.path.join(net_path, 'layer_cnv1_sl1_.json'))

    pre_net.add(layer1_cnv)
    pre_net.add(bb.MaxPooling.create(2, 2))

    if True:
        layer_distillation(td['x_train'], td['x_shape'], target_cnv2, layer2_cnv, pre_net)
        layer_cnv2_sl0.save_json(os.path.join(net_path, 'layer_cnv2_sl0.json'))
        layer_cnv2_sl1.save_json(os.path.join(net_path, 'layer_cnv2_sl1.json'))

    pre_net.add(layer2_cnv)

    if True:
        layer_distillation(td['x_train'], td['x_shape'], target_cnv3, layer3_cnv, pre_net)
        layer_cnv3_sl0.save_json(os.path.join(net_path, 'layer_cnv3_sl0.json'))
        layer_cnv3_sl1.save_json(os.path.join(net_path, 'layer_cnv3_sl1.json'))


    
if __name__ == '__main__':
    main()

