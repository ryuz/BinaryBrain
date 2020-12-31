
import binarybrain as bb
import numpy as np

from matplotlib import pyplot as plt

import torch
import torchvision
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm

def main():
    print("start")

    dataset_train = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    dataset_test  = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=64, shuffle=True, num_workers=2)
    loader_test  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=64, shuffle=False, num_workers=2)

    net = bb.Sequential([
                bb.DenseAffine([1024]),
                bb.ReLU(),
                bb.DenseAffine([512]),
                bb.ReLU(),
                bb.DenseAffine([10]),
            ])
    net.set_input_shape([28, 28])

    loss      = bb.LossSoftmaxCrossEntropy()
    metrics   = bb.MetricsCategoricalAccuracy()
    optimizer = bb.OptimizerAdam()

    optimizer.set_variables(net.get_parameters(), net.get_gradients())

    for epoch in range(8):
        for images, labels in loader_train:
            x_buf = bb.FrameBuffer.from_numpy(np.array(images).astype(np.float32))
            t_buf = bb.FrameBuffer.from_numpy(np.identity(10)[np.array(labels)].astype(np.float32))
            
            y_buf = net.forward(x_buf, train=True)

            dy_buf = loss.calculate(y_buf, t_buf)
            net.backward(dy_buf)

            optimizer.update()

            
        for images, labels in loader_test:
            x_buf = bb.FrameBuffer.from_numpy(np.array(images).astype(np.float32))
            t_buf = bb.FrameBuffer.from_numpy(np.identity(10)[np.array(labels)].astype(np.float32))
            
            y_buf = net.forward(x_buf, train=False)
            
            loss.calculate(y_buf, t_buf)
            metrics.calculate(y_buf, t_buf)

        print('epoch[%d] : loss=%f accuracy=%f' % (epoch, loss.get(), metrics.get()))


if __name__ == "__main__":
    main()

