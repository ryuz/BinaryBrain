# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import binarybrain as bb
import numpy as np
import cv2
from tqdm import tqdm


def make_image_block(vec):
    vec = np.array(vec)
    img = np.ndarray((28, 0))
    for x in vec:
        img = np.hstack((img, x.reshape(28, -1)))
    return img

def main():
    epoch                    = 8
    mini_batch               = 32
    training_modulation_size = 7
    test_modulation_size     = 7
    
    # load MNIST data
    td = bb.load_mnist()

    # set teaching signnal
    td['t_shape'] = td['x_shape']
    td['t_train'] = td['x_train']
    td['t_test']  = td['x_test']

    # create layer
    layer_enc_sl0 = bb.SparseLut6.create([32*6*6*6])
    layer_enc_sl1 = bb.SparseLut6.create([32*6*6])
    layer_enc_sl2 = bb.SparseLut6.create([32*6])
    layer_enc_sl3 = bb.SparseLut6.create([32])

    layer_dec_sl2 = bb.SparseLut6.create([28*28*6*6])
    layer_dec_sl1 = bb.SparseLut6.create([28*28*6])
    layer_dec_sl0 = bb.SparseLut6.create([28*28], False)    # diable BatchNorm

    # create network
    main_net = bb.Sequential.create()
    main_net.add(layer_enc_sl0)
    main_net.add(layer_enc_sl1)
    main_net.add(layer_enc_sl2)
    main_net.add(layer_enc_sl3)
    main_net.add(layer_dec_sl2)
    main_net.add(layer_dec_sl1)
    main_net.add(layer_dec_sl0)


    # wrapping with binary modulator
    net = bb.Sequential.create()
    net.add(bb.BinaryModulation.create(main_net, training_modulation_size=training_modulation_size))
    net.set_input_shape(td['x_shape'])

    print(net.get_info())

    loss      = bb.LossMeanSquaredError.create()
    metrics   = bb.MetricsMeanSquaredError.create()
    optimizer = bb.OptimizerAdam.create()

    optimizer.set_variables(net.get_parameters(), net.get_gradients())

    batch_size = len(td['x_train'])
    print('batch_size =', batch_size)


#    runner = bb.Runner(net, "mnist-autoencoder-sparse-lut6-simple", loss, metrics, optimizer)
#    runner.fitting(td, epoch_size=epoch, mini_batch_size=mini_batch)

    result_img = None

    x_train = td['x_train']
    t_train = td['t_train']
    x_test  = td['x_test']
    t_test  = td['t_test']
    x_buf = bb.FrameBuffer()
    t_buf = bb.FrameBuffer()
    for epoch_num in range(epoch):
        # train
        for index in tqdm(range(0, batch_size, mini_batch)):
            mini_batch_size = min(mini_batch, batch_size-index)
            
            x_buf.resize(mini_batch_size, td['x_shape'], bb.TYPE_FP32)
            x_buf.set_data(x_train[index:index+mini_batch_size])

            y_buf = net.forward(x_buf)
            
            t_buf.resize(mini_batch_size, td['t_shape'], bb.TYPE_FP32)
            t_buf.set_data(t_train[index:index+mini_batch_size])
            
            dy_buf = loss.calculate_loss(y_buf, t_buf, mini_batch_size)
            metrics.calculate_metrics(y_buf, t_buf)
            dx_buf = net.backward(dy_buf)
            
            optimizer.update()
            cv2.waitKey(1)
        
        print('loss =', loss.get_loss())
        print('metrics =', metrics.get_metrics())
        
        # test
        x_buf.resize(16, td['x_shape'], bb.TYPE_FP32)
        x_buf.set_data(x_test[0:16])
        y_buf = net.forward(x_buf)
        
        if result_img is None:
            x_img = make_image_block(x_buf.get_data())
            cv2.imwrite('mnist-autoencoder-sparse-lut6-simple_x.png', x_img*255)
            result_img = x_img

        y_img = make_image_block(y_buf.get_data())
        cv2.imwrite('mnist-autoencoder-sparse-lut6-simple_%d.png' % epoch_num, y_img*255)
        result_img = np.vstack((result_img, y_img))
        cv2.imshow('result_img', result_img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cv2.imwrite("mnist-autoencoder-sparse-lut6-simple.png", result_img*255)
    
    # LUT-network
    layer_enc_bl0 = bb.BinaryLut6Bit.create(layer_enc_sl0.get_output_shape())
    layer_enc_bl1 = bb.BinaryLut6Bit.create(layer_enc_sl1.get_output_shape())
    layer_enc_bl2 = bb.BinaryLut6Bit.create(layer_enc_sl2.get_output_shape())
    layer_enc_bl3 = bb.BinaryLut6Bit.create(layer_enc_sl3.get_output_shape())
    layer_dec_bl2 = bb.BinaryLut6Bit.create(layer_dec_sl2.get_output_shape())
    layer_dec_bl1 = bb.BinaryLut6Bit.create(layer_dec_sl1.get_output_shape())
    layer_dec_bl0 = bb.BinaryLut6Bit.create(layer_dec_sl0.get_output_shape())
    
    lut_net = bb.Sequential.create()
    lut_net.add(layer_enc_bl0)
    lut_net.add(layer_enc_bl1)
    lut_net.add(layer_enc_bl2)
    lut_net.add(layer_enc_bl3)
    lut_net.add(layer_dec_bl2)
    lut_net.add(layer_dec_bl1)
    lut_net.add(layer_dec_bl0)

    # evaluation network
    eval_net = bb.Sequential.create()
    eval_net.add(bb.BinaryModulationBit.create(lut_net, inference_modulation_size=test_modulation_size))
    eval_net.add(bb.Reduce.create(td['t_shape']))

    # set input shape
    eval_net.set_input_shape(td['x_shape'])

    # import table
    print('parameter copy to binary LUT-Network')
    layer_enc_bl0.import_parameter(layer_enc_sl0)
    layer_enc_bl1.import_parameter(layer_enc_sl1)
    layer_enc_bl2.import_parameter(layer_enc_sl2)
    layer_enc_bl3.import_parameter(layer_enc_sl3)
    layer_dec_bl2.import_parameter(layer_dec_sl2)
    layer_dec_bl1.import_parameter(layer_dec_sl1)
    layer_dec_bl0.import_parameter(layer_dec_sl0)

    # evaluation
    lut_runner = bb.Runner(eval_net, "mnist-autoencpder-binary-lut6-simple",
                    bb.LossMeanSquaredError.create(),
                    bb.MetricsMeanSquaredError.create())
    lut_runner.evaluation(td, mini_batch_size=mini_batch)

    # Verilog 出力
    with open('MnistAeLutSimple.v', 'w') as f:
        f.write('`timescale 1ns / 1ps\n\n')
        f.write(bb.make_verilog_from_lut_bit('MnistAeLutSimple',
            [layer_enc_bl0, layer_enc_bl1, layer_enc_bl2,
            layer_dec_bl2, layer_dec_bl1, layer_dec_bl0]))


if __name__ == '__main__':
    main()

