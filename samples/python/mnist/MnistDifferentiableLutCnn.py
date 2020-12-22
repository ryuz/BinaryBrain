# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import binarybrain as bb


def main():
    epochs    = 4
    net_name  = 'MnistDifferentiableLutCnn'
    data_path = './data' + net_name
    frame_modulation_size = 7
    
    # dataset
    dataset_train = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    dataset_test  = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=64, shuffle=True, num_workers=2)
    loader_test  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=64, shuffle=False, num_workers=2)
    
    # define network
    net = bb.Sequential([
                bb.RealToBinary(frame_modulation_size=frame_modulation_size),
                bb.Sequential([
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([36*6]),
                            bb.DifferentiableLut([36]),
                        ]),
                        filter_size=(3, 3)),
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([2*36*6]),
                            bb.DifferentiableLut([2*36]),
                        ]),
                        filter_size=(3, 3)),
                    bb.MaxPooling(filter_size=(2, 2)),
                ]),
                bb.Sequential([
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([4*36*6]),
                            bb.DifferentiableLut([4*36]),
                        ]),
                        filter_size=(3, 3)),
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([4*36*6]),
                            bb.DifferentiableLut([4*36]),
                        ]),
                        filter_size=(3, 3)),
                    bb.MaxPooling(filter_size=(2, 2)),
                ]),
                bb.Sequential([
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([6*6*10]),
                            bb.DifferentiableLut([6*10]),
                            bb.DifferentiableLut([10]),
                        ]),
                        filter_size=(4, 4)),
                ]),
                bb.BinaryToReal(frame_modulation_size=frame_modulation_size)
            ])

    net.set_input_shape([1, 28, 28])

    net.send_command("binary true")
    
    print(net.get_info())
    
    bb.load_networks(data_path, net)
    
    # learning
    loss      = bb.LossSoftmaxCrossEntropy()
    metrics   = bb.MetricsCategoricalAccuracy()
    optimizer = bb.OptimizerAdam()
    
    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    
    for epoch in range(epochs):
        loss.clear()
        metrics.clear()
        
        # learning
        with tqdm(loader_train) as t:
            for images, labels in t:

            #  for images, labels in tqdm(loader_train):
                x_buf = bb.FrameBuffer.from_numpy(np.array(images).astype(np.float32))
                t_buf = bb.FrameBuffer.from_numpy(np.identity(10)[np.array(labels)].astype(np.float32))

                y_buf = net.forward(x_buf, train=True)

                dy_buf = loss.calculate(y_buf, t_buf)
                metrics.calculate(y_buf, t_buf)
                net.backward(dy_buf)
                
                optimizer.update()
                
                t.set_postfix(loss=loss.get(), acc=metrics.get())
        
        # test
        loss.clear()
        metrics.clear()
        for images, labels in loader_test:
            x_buf = bb.FrameBuffer.from_numpy(np.array(images).astype(np.float32))
            t_buf = bb.FrameBuffer.from_numpy(np.identity(10)[np.array(labels)].astype(np.float32))
            
            y_buf = net.forward(x_buf, train=False)
            
            loss.calculate(y_buf, t_buf)
            metrics.calculate(y_buf, t_buf)
        
        bb.save_networks(data_path, net)
        
        print('epoch[%d] : loss=%f accuracy=%f' % (epoch, loss.get(), metrics.get()))
    
    # export verilog
    with open(net_name+'.v', 'w') as f:
        f.write('`timescale 1ns / 1ps\n\n')
        f.write(bb.make_verilog_lut_cnv_layers(net_name + 'Cnv0', net[1]))
        f.write(bb.make_verilog_lut_cnv_layers(net_name + 'Cnv1', net[2]))
        f.write(bb.make_verilog_lut_cnv_layers(net_name + 'Cnv2', net[3]))
    

if __name__ == "__main__":
    main()

