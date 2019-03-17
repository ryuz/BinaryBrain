import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Hyperparameters 
num_epochs = 10
batch_size = 128
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input.clamp(min=0)
    
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer0_bin  = BinActive.apply
        self.layer1_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.layer1_norm = nn.BatchNorm2d(32)
  #     self.layer1_act  = nn.ReLU()
        self.layer1_act  = BinActive.apply
        self.layer2_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer2_norm = nn.BatchNorm2d(32)
  #     self.layer2_act  = nn.ReLU()
        self.layer2_act  = BinActive.apply
        self.layer3_pol  = nn.MaxPool2d(2)
        self.layer4_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer4_norm = nn.BatchNorm2d(32)
   #     self.layer4_act  = nn.ReLU()
        self.layer4_act  = BinActive.apply
        self.layer5_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer5_norm = nn.BatchNorm2d(32)
   #     self.layer5_act  = nn.ReLU()
        self.layer5_act  = BinActive.apply
        self.layer6_pol  = nn.MaxPool2d(2)
        self.layer7_affine = nn.Linear(32 * 7 * 7, 256)
   #     self.layer7_act  = nn.ReLU()
        self.layer7_act  = BinActive.apply
#      self.layer7_norm   = nn.BatchNorm()
        self.layer8_affine = nn.Linear(256, 10)
        
    def forward(self, x):
    #    out = self.layer0_bin(x)
        out = self.layer1_conv(x)
        out = self.layer1_norm(out)
        out = self.layer1_act(out)
        out = self.layer2_conv(out)
        out = self.layer2_norm(out)
        out = self.layer2_act(out)
        out = self.layer3_pol(out)
        out = self.layer4_conv(out)
        out = self.layer4_norm(out)
        out = self.layer4_act(out)
        out = self.layer5_conv(out)
        out = self.layer5_norm(out)
        out = self.layer5_act(out)
        out = self.layer6_pol(out)
        out = out.view(out.size(0), -1)
        out = self.layer7_affine(out)
        out = self.layer7_act(out)
        out = self.layer8_affine(out)
        return out

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(train_loader):
    model.train()
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader)
    
    return train_loss


def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = outputs.max(1, keepdim=True)[1]
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(test_loader)
    val_acc = correct / total
    
    return val_loss, val_acc

loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(num_epochs):
    loss = train(train_loader)
    val_loss, val_acc = valid(test_loader)
    train_val_loss, train_val_acc = valid(train_loader)

    print('epoch %d, loss: %.4f train_val_loss: %.4f train_val_acc: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, train_val_loss, train_val_acc, val_loss, val_acc))
    
    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)


# save the trained model
np.save('loss_list.npy', np.array(loss_list))
np.save('val_loss_list.npy', np.array(val_loss_list))
np.save('val_acc_list.npy', np.array(val_acc_list))
torch.save(model.state_dict(), 'cnn.pkl')
