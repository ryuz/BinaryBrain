# coding: utf-8

import binarybrain as bb
import numpy as np

def main():
    # config
    binary_mode               = True
    epoch                     = 4
    mini_batch                = 32
    training_modulation_size  = 3
    inference_modulation_size = 3
    
    # download MNIST data
    bb.download_mnist()
    
    # load MNIST data
    td = bb.LoadMnist.load()
    
    batch_size = len(td.x_train)
    print('batch_size =', batch_size)
    
    
    ############################
    # Learning
    ############################
        
    # create network
    main_net = bb.Sequential.create()
    main_net.add(bb.DenseAffine.create(output_shape=[1024]))
    main_net.add(bb.BatchNormalization.create())
    main_net.add(bb.ReLU.create())
    main_net.add(bb.DenseAffine.create([512]))
    main_net.add(bb.BatchNormalization.create())
    main_net.add(bb.ReLU.create())
    main_net.add(bb.DenseAffine.create(td.t_shape))
    if binary_mode:
        main_net.add(bb.BatchNormalization.create())
        main_net.add(bb.ReLU.create())
    
    # wrapping with binary modulator
    net = bb.Sequential.create()
    net.add(bb.BinaryModulation.create(main_net, training_modulation_size=training_modulation_size))
    net.add(bb.Reduce.create(td.t_shape))
    net.set_input_shape(td.x_shape)
    
    # print model information
    print(net.get_info())
    
    # set binary mode
    if binary_mode:
        net.send_command("binary true");
    else:
        net.send_command("binary false");
    
    
    # learning
    print('\n[learning]')
    loss      = bb.LossSoftmaxCrossEntropy.create()
    metrics   = bb.MetricsCategoricalAccuracy.create()
    optimizer = bb.OptimizerAdam.create()
    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    
    runner = bb.Runner(net, "mnist-sparse-lut6-simple", loss, metrics, optimizer)
    runner.fitting(td, epoch_size=epoch, mini_batch_size=mini_batch)


if __name__ == '__main__':
    main()

