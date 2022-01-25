import torch
import sys
import numpy as np
import torch.nn.functional as F
from torch import optim
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision import transforms as tr
from torchvision import datasets
import torch.utils.data as t_data

BATCH_SIZE = 64
LR = 0.0007
EPOCHS = 10

class A(nn.Module):
    def __init__(self,image_size):
        super(A,self).__init__()
        self.image_size=image_size
        self.fc0=nn.Linear(image_size,100)
        self.fc1=nn.Linear(100,50)
        self.fc2=nn.Linear(50,10)

    def forward(self,x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


class E(nn.Module):
    def __init__(self, image_size):
        super(E,self).__init__()
       ##?? super()._init_()
        self.image_size=image_size
        self.fc0=nn.Linear(image_size,128)
        self.fc1=nn.Linear(128,64)
        self.fc2=nn.Linear(64,10)
        self.fc3=nn.Linear(10,10)
        self.fc4=nn.Linear(10,10)
        self.fc5=nn.Linear(10,10)

    def forward(self,x):
        x = x.view(-1,self.image_size)
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        return F.log_softmax(x, dim = 1)

class f(nn.Module):
    def __init__(self, image_size):
        super(f,self).__init__()
        super().__init__()
        self.image_size=image_size
        self.fc0=nn.Linear(image_size,128)
        self.fc1=nn.Linear(128,64)
        self.fc2=nn.Linear(64,10)
        self.fc3=nn.Linear(10,10)
        self.fc4=nn.Linear(10,10)
        self.fc5=nn.Linear(10,10)

    def forward(self,x):
        x = x.view(-1,self.image_size)
        x=torch.sigmoid(self.fc0(x))
        x=torch.sigmoid(self.fc1(x))
        x=torch.sigmoid(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))
        x=torch.sigmoid(self.fc4(x))
        x=self.fc5(x)
        return F.log_softmax(x, dim =1)



class B(nn.Module):
    def __init__(self,image_size):
        super(B,self).__init__()
        self.image_size=image_size
        self.fc0=nn.Linear(image_size,100)
        self.fc1=nn.Linear(100,50)
        self.fc2=nn.Linear(50,10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class C(nn.Module):
    def __init__(self,image_size):
        super(C,self).__init__()
        self.image_size=image_size
        self.fc0=nn.Linear(image_size,100)
        self.fc1=nn.Linear(100,50)
        self.fc2=nn.Linear(50,10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim =1)


class D(nn.Module):
    def __init__(self,image_size):
        super(D,self).__init__()
        self.image_size=image_size
        self.fc0=nn.Linear(image_size,100)
        self.bn0 = nn.BatchNorm1d(100)
        self.fc1=nn.Linear(100,50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2=nn.Linear(50,10)
        self.bn2 = nn.BatchNorm1d(10)


    def forward(self,x):
        x = x.view(-1,self.image_size)
        x = self.fc0(x)
        x=F.relu(self.bn0(x))
        x = self.fc1(x)
        x=F.relu(self.bn1(x))
        x=self.fc2(x)
        x = F.relu(self.bn2(x))
        return F.log_softmax(x, dim =1)


def normalize(train_x, test_x):
    # zscore on train_x
    mean, std = train_x.mean(), train_x.std()
    train_x = (train_x - mean) / std

    #zscore on test_x
    test_x = (test_x - mean) / std

    return train_x, test_x


# loads the data files.
def load_files(x, y, test):
    train_x = np.loadtxt(x)
    train_y = np.loadtxt(y)
    test_x = np.loadtxt(test)
    test_y = np.zeros(len(test_x))

    # normalize data with z_score
    train_x, test_x = normalize(train_x, test_x)

    # calculate the sizes of train set and validation set
    train_size = len(train_x)
    test_size = len(test_x)

    train_x_size = int(train_size * 0.8)
    validation_size = train_size - train_x_size

    # make tensors from the np arrays.
    train_x_tensor = t_data.TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).type(torch.LongTensor))
    test_x_tensor = t_data.TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).type(torch.LongTensor))

    train_set, valid_set = t_data.random_split(train_x_tensor, [train_x_size, validation_size])

    # creating the loaders.
    train_loader = t_data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = t_data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = t_data.DataLoader(test_x_tensor, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader


def train(model, optimizer, train_loader):

    train_loss = 0
    correct = 0
    for (x,y) in train_loader:
        # getting rid of the garbage values.
        optimizer.zero_grad()
        # going forward.
        output = model(x)
        # calculating loss and going backwards.
        loss = F.nll_loss(output, y)
        train_loss += F.nll_loss(output, y,reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).cpu().sum()
        loss.backward()
        #updates.
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    return train_loss, 100. * correct / len(train_loader.dataset)


def validate(model, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)
            test_loss += F.nll_loss(output, y, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).cpu().sum()
    test_loss /= len(valid_loader.dataset)
    return test_loss,100. * correct / len(valid_loader.dataset)

def test(model, test_x, test_y):
    model.eval()
    file = open(test_y, 'w+')
    for x,y in test_x:
        output = model(x)
        pred = output.max(1, keepdim=True)[1]
        for y in pred:
            file.write(str(int(y)) +'\n')
    file.close()

def graph(a,b,name):
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.plot(labels, a, label="train")
    plt.plot(labels, b, label="validation")
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title(name+' graph!')
    plt.legend()
    plt.show()

def run_model(model,optimizer,train_x,test_x):
    t_loss = []
    t_accu = []
    v_loss = []
    v_accu = []
    for ep in range(EPOCHS):
        loss,accu = train(model, optimizer, train_x)
        t_loss.append(loss)
        t_accu.append(accu)
        loss,accu=validate(model,test_x)
        v_loss.append(loss)
        v_accu.append(accu)
    #graph(t_loss,v_loss,"Loss")
    #graph(t_accu,v_accu,"Accuracy")


def main():
    x_path, y_path, test_path, test_y = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    train_x,valid_set, test_x = load_files(x_path, y_path, test_path)


    model = D(784)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    run_model(model, optimizer, train_x, valid_set)
    test(model,test_x,test_y)



if __name__ == "__main__":
    main()