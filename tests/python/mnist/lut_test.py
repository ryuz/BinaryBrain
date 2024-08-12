
import os
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

import binarybrain as bb

print(bb.get_version_string())



# configuration
data_path             = './data/'
epochs                = 8
mini_batch_size       = 64
frame_modulation_size = 15


# dataset
dataset_path = './data/'
dataset_train = torchvision.datasets.MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
dataset_test  = torchvision.datasets.MNIST(root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)
loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=mini_batch_size, shuffle=True, num_workers=2)
loader_test  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=mini_batch_size, shuffle=False, num_workers=2)


N=4
L1 = 4
L2 = 4

def eval(N, L1, L2, f):
    # define network
    LUTs = 0
    net = bb.Sequential()
    net.append(bb.RealToBinary(frame_modulation_size=frame_modulation_size))
    for i in range(L1):
        k = L1 - i - 1
        con = 'serial' if i > 0 else 'random'
        net.append(bb.DifferentiableLut([64*(N**k)], connection=con, N=N))
        LUTs += 64*(N**k)
    for i in range(L2-1):
        k = L2 - i - 1
        con = 'serial' if i > 0 else 'random'
        net.append(bb.DifferentiableLut([10*(N**k)], connection=con, N=N))
        LUTs += 10*(N**k)
    net.append(bb.AverageLut([10],     connection='serial', N=N))
    LUTs += 10
    net.append(bb.BinaryToReal(frame_integration_size=frame_modulation_size))
    net.set_input_shape([1, 28, 28])

    net.send_command("binary true")

#   net.print_info()
    BITS = LUTs * (2**N)
    print(f'LUTs = {LUTs}   bits = {BITS}')

    loss      = bb.LossSoftmaxCrossEntropy()
    metrics   = bb.MetricsCategoricalAccuracy()
    optimizer = bb.OptimizerAdam()

    optimizer.set_variables(net.get_parameters(), net.get_gradients())

    max_acc = 0

    for epoch in range(epochs):
        # learning
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
        
        acc  = metrics.get()
        print('epoch[%d] : loss=%f accuracy=%f' % (epoch, loss.get(), acc))
        if acc > max_acc:
            max_acc = acc
    
    f.write(f'LUT{N},{N},{L1},{L2},{max_acc},{LUTs},{BITS}\n')
    
    return max_acc, LUTs, BITS

with open('lut.csv', 'w') as f:
    f.write('LUT,N,L1,L2,accuracy,LUTs,bits\n')
    eval(2, 8, 8, f)
#   eval(2, 4, 4, f)
#   eval(2, 5, 5, f)
#   eval(2, 6, 6, f)
#   eval(3, 4, 4, f)
#   eval(3, 5, 5, f)
#   eval(3, 6, 6, f)
#   eval(4, 3, 3, f)
#   eval(4, 3, 4, f)
#   eval(4, 4, 3, f)
#   eval(4, 4, 4, f)
#    eval(5, 3, 3, f)
#    eval(5, 3, 4, f)
#    eval(5, 4, 3, f)
#    eval(5, 4, 4, f)
#    eval(6, 3, 3, f)
#    eval(6, 3, 4, f)
#    eval(6, 4, 3, f)
#    eval(6, 4, 4, f)