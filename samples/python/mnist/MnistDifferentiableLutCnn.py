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
    net_name              = 'MnistDifferentiableLutCnn'
    data_path             = os.path.join('./data/', net_name)
    rtl_sim_path          = '../../verilog/mnist/tb_mnist_lut_cnn'
    rtl_module_name       = 'MnistLutCnn'
    output_velilog_file   = os.path.join(data_path, rtl_module_name + '.v')
    sim_velilog_file      = os.path.join(rtl_sim_path, rtl_module_name + '.v')

    bin_mode              = True
    frame_modulation_size = 7
    epochs                = 8
    mini_batch_size       = 64

    # dataset
    dataset_path = './data/'
    dataset_train = torchvision.datasets.MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    dataset_test  = torchvision.datasets.MNIST(root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=mini_batch_size, shuffle=True, num_workers=2)
    loader_test  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=mini_batch_size, shuffle=False, num_workers=2)
    
    # select Binary DataType
    bin_dtype = bb.DType.BIT if bin_mode else bb.DType.FP32

    # Define networks
    net = bb.Sequential([
                bb.RealToBinary(frame_modulation_size=frame_modulation_size, bin_dtype=bin_dtype),
                bb.Sequential([
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([36*6], bin_dtype=bin_dtype),
                            bb.DifferentiableLut([36], bin_dtype=bin_dtype),
                        ]),
                        filter_size=(3, 3),
                        fw_dtype=bin_dtype),
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([2*36*6], bin_dtype=bin_dtype),
                            bb.DifferentiableLut([2*36], bin_dtype=bin_dtype),
                        ]),
                        filter_size=(3, 3),
                        fw_dtype=bin_dtype),
                    bb.MaxPooling(filter_size=(2, 2), fw_dtype=bin_dtype),
                ]),
                bb.Sequential([
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([2*36*6], bin_dtype=bin_dtype),
                            bb.DifferentiableLut([2*36], bin_dtype=bin_dtype),
                        ]),
                        filter_size=(3, 3),
                        fw_dtype=bin_dtype),
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([4*36*6], bin_dtype=bin_dtype),
                            bb.DifferentiableLut([4*36], bin_dtype=bin_dtype),
                        ]),
                        filter_size=(3, 3),
                        fw_dtype=bin_dtype),
                    bb.MaxPooling(filter_size=(2, 2), fw_dtype=bin_dtype),
                ]),
                bb.Sequential([
                    bb.Convolution2d(
                        bb.Sequential([
                            bb.DifferentiableLut([6*128], bin_dtype=bin_dtype),
                            bb.DifferentiableLut([128], bin_dtype=bin_dtype),
                            bb.DifferentiableLut([6*6*10], bin_dtype=bin_dtype),
                            bb.DifferentiableLut([6*10], bin_dtype=bin_dtype),
                            bb.DifferentiableLut([10], bin_dtype=bin_dtype),
                        ]),
                        filter_size=(4, 4),
                        fw_dtype=bin_dtype),
                ]),
                bb.BinaryToReal(frame_integration_size=frame_modulation_size, bin_dtype=bin_dtype)
            ])

    net.set_input_shape([1, 28, 28])

    # set binary mode
    if bin_mode:
        net.send_command("binary true")

    # load last training data
    # bb.load_networks(data_path, net)
    
    # learning
    loss      = bb.LossSoftmaxCrossEntropy()
    metrics   = bb.MetricsCategoricalAccuracy()
    optimizer = bb.OptimizerAdam()
    
    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    
    for epoch in range(epochs):
        # training
        loss.clear()
        metrics.clear()
        with tqdm(loader_train) as t:
            for images, labels in t:
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
    with open(output_velilog_file, 'w') as f:
        f.write('`timescale 1ns / 1ps\n\n')
        bb.dump_verilog_lut_cnv_layers(f, rtl_module_name + 'Cnv0', net[1])
        bb.dump_verilog_lut_cnv_layers(f, rtl_module_name + 'Cnv1', net[2])
        bb.dump_verilog_lut_cnv_layers(f, rtl_module_name + 'Cnv2', net[3])

    # Simulation用ファイルに上書きコピー
    shutil.copyfile(output_velilog_file, sim_velilog_file)
    
    # Simulationで使う画像の生成
    def img_geneator():
        for data in dataset_test:
            yield data[0] # 画像とラベルの画像の方を返す

    img = (bb.make_image_tile(480//28+1, 640//28+1, img_geneator())*255).astype(np.uint8)
    bb.write_ppm(os.path.join(rtl_sim_path, 'mnist_test_160x120.ppm'), img[:,:120,:160])
    bb.write_ppm(os.path.join(rtl_sim_path, 'mnist_test_640x480.ppm'), img[:,:480,:640])

if __name__ == "__main__":
    main()

