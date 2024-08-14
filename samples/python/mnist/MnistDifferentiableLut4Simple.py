# Interface 2024年10月号 特別付録 FPGA マガジン No.3
# GOWIN Tang Nano 4k 向けの学習サンプル

#%%
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import binarybrain as bb

# %%
mini_batch_size = 64
dataset_path = './data/'
dataset_train = torchvision.datasets.MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
dataset_test  = torchvision.datasets.MNIST(root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)
loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=mini_batch_size, shuffle=True, num_workers=2)
loader_test  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=mini_batch_size, shuffle=False, num_workers=2)

# %%
frame_modulation_size = 31
net = bb.Sequential([
            bb.RealToBinary(frame_modulation_size=frame_modulation_size),
            bb.DifferentiableLut([64*4*2], connection='random', N=4),
            bb.DifferentiableLut([64*4],   connection='serial', N=4),
            bb.DifferentiableLut([64],     connection='serial', N=4),
            bb.DifferentiableLut([10*4*4], connection='random', N=4),
            bb.DifferentiableLut([10*4],   connection='serial', N=4),
            bb.AverageLut       ([10],     connection='serial', N=4),
            bb.BinaryToReal(frame_integration_size=frame_modulation_size)
        ])
net.set_input_shape([1, 28, 28])
net.send_command("binary true")

# %%
loss      = bb.LossSoftmaxCrossEntropy()
metrics   = bb.MetricsCategoricalAccuracy()
optimizer = bb.OptimizerAdam(learning_rate=0.001)
optimizer.set_variables(net.get_parameters(), net.get_gradients())

from tqdm import tqdm
for epoch in range(8):
    # Training
    loss.clear()
    metrics.clear()
    with tqdm(loader_train) as t:
        for images, labels in t:
            # データ用意
            x_buf = bb.FrameBuffer.from_numpy(np.array(images).astype(np.float32))
            t_buf = bb.FrameBuffer.from_numpy(np.identity(10)[np.array(labels)].astype(np.float32))

            # forward 計算
            y_buf = net.forward(x_buf, train=True)
            
            # 損失計算
            dy_buf = loss.calculate(y_buf, t_buf)
            metrics.calculate(y_buf, t_buf)
            
            # backward (誤差逆伝搬)
            net.backward(dy_buf)
            
            # パラメータ更新
            optimizer.update()
            
            # 進捗表示
            t.set_postfix(loss=loss.get(), acc=metrics.get())

    # Test
    loss.clear()
    metrics.clear()
    for images, labels in loader_test:
        x_buf = bb.FrameBuffer.from_numpy(np.array(images).astype(np.float32))
        t_buf = bb.FrameBuffer.from_numpy(np.identity(10)[np.array(labels)].astype(np.float32))
        y_buf = net.forward(x_buf, train=False)
        loss.calculate(y_buf, t_buf)
        metrics.calculate(y_buf, t_buf)
    print('epoch[%d] : loss=%f accuracy=%f' % (epoch, loss.get(), metrics.get()))


# %%
with open('MnistLutSimple.v', 'w') as f:
    f.write('`timescale 1ns / 1ps\n\n')
    bb.dump_verilog_lut_layers(f, module_name='MnistLutSimple', net=net)


# %%
# layer を取り出す
layer0 = net[1]
connection_mat = np.array(layer0.get_connection_list())
print(connection_mat.shape)
print(connection_mat)


# %%
W = layer0.W().numpy()
print(W.shape)
print(W)


# %%
lut_mat = np.array(layer0.get_lut_table_list())
print(lut_mat.shape)
print(lut_mat)

