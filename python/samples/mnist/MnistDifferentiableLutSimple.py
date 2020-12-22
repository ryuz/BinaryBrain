# -*- coding: utf-8 -*-

import binarybrain as bb
import numpy as np

from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

def main():
    # setting
    epochs    = 4
    net_name  = 'MnistDifferentiableLutSimple'
    data_path = './data/' + net_name
    frame_modulation_size = 15


    # dataset
    dataset_train = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    dataset_test  = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=64, shuffle=True, num_workers=2)
    loader_test  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=64, shuffle=False, num_workers=2)

    # define network
    net = bb.Sequential([
                bb.RealToBinary(frame_modulation_size=frame_modulation_size),
                bb.DifferentiableLut([1024]),
                bb.DifferentiableLut([512]),
                bb.DifferentiableLut([10]),
                bb.BinaryToReal(frame_modulation_size=frame_modulation_size)
            ])
    net.set_input_shape([1, 28, 28])

    net.send_command("binary true")

    bb.load_networks(data_path, net)

    # learning
    loss      = bb.LossSoftmaxCrossEntropy()
    metrics   = bb.MetricsCategoricalAccuracy()
    optimizer = bb.OptimizerAdam()

    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    
    for epoch in range(epochs):
        # learning
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
    
    # export verilog
    bb.export_verilog_lut_layers(network_name + '.v', network_name, net)

    
if __name__ == "__main__":
    main()


