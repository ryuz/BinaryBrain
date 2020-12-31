# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

import binarybrain as bb


def main():
    # configuration
    data_path             = './data/'
    net_name              = 'MnistDifferentiableLutSimple'
    data_path             = os.path.join('./data/', net_name)
    rtl_sim_path          = '../../verilog/mnist'
    rtl_module_name       = 'MnistLutSimple'
    output_velilog_file   = os.path.join(data_path, net_name + '.v')
    sim_velilog_file      = os.path.join(rtl_sim_path, rtl_module_name + '.v')

    epochs                = 1
    mini_batch_size       = 64
    frame_modulation_size = 15

    # dataset
    dataset_path = './data/'
    dataset_train = torchvision.datasets.MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    dataset_test  = torchvision.datasets.MNIST(root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=mini_batch_size, shuffle=True, num_workers=2)
    loader_test  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=mini_batch_size, shuffle=False, num_workers=2)

    # define network
    net = bb.Sequential([
                bb.RealToBinary(frame_modulation_size=frame_modulation_size),
                bb.DifferentiableLut([1024]),
                bb.DifferentiableLut([420]),
                bb.DifferentiableLut([70]),
                bb.Reduce([10]),
                bb.BinaryToReal(frame_modulation_size=frame_modulation_size)
            ])
    net.set_input_shape([1, 28, 28])

    net.send_command("binary true")

    # load last data
    bb.load_networks(data_path, net)

    # learning
    loss      = bb.LossSoftmaxCrossEntropy()
    metrics   = bb.MetricsCategoricalAccuracy()
    optimizer = bb.OptimizerAdam()

    optimizer.set_variables(net.get_parameters(), net.get_gradients())

    for epoch in range(epochs):
        # training
        with tqdm(loader_train) as t:
            for images, labels in t:
                x_buf = bb.FrameBuffer.from_numpy(np.array(images).astype(np.float32))
                t_buf = bb.FrameBuffer.from_numpy(np.identity(10)[np.array(labels)].astype(np.float32))

                y_buf = net.forward(x_buf, train=True)

                dy_buf = loss.calculate(y_buf, t_buf)
                net.backward(dy_buf)

                optimizer.update()
                
                t.set_postfix(loss=loss.get(), acc=metrics.get())
        
        # test
        for images, labels in loader_test:
            x_buf = bb.FrameBuffer.from_numpy(np.array(images).astype(np.float32))
            t_buf = bb.FrameBuffer.from_numpy(np.identity(10)[np.array(labels)].astype(np.float32))

            y_buf = net.forward(x_buf, train=False)

            loss.calculate(y_buf, t_buf)
            metrics.calculate(y_buf, t_buf)

        print('epoch[%d] : loss=%f accuracy=%f' % (epoch, loss.get(), metrics.get()))

        bb.save_networks(data_path, net, keep_olds=3)
    
    # write verilog
    print('write : %s'%output_velilog_file)
    with open(output_velilog_file, 'w') as f:
        f.write('`timescale 1ns / 1ps\n\n')
        bb.dump_verilog_lut_layers(f, module_name=rtl_module_name, net=net)

    # copy for RTL simulation
    print('copy : %s -> %s'%(output_velilog_file, sim_velilog_file))
    shutil.copyfile(output_velilog_file, sim_velilog_file)

    # make data file for RTL simuration
    print('write : %s'%(os.path.join(rtl_sim_path, 'mnist_test.txt')))
    with open(os.path.join(rtl_sim_path, 'mnist_test.txt'), 'w') as f:
        bb.dump_verilog_readmemb_image_classification(f ,loader_test)


if __name__ == "__main__":
    main()

