import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

correct_list = []
cost_list = []
train_accurecy = []
train_loss =[]
correct_list_train = []
cost_list_train = []
num_epochs = 60
runs = 50000
nf = 1
nbs = 8
nue = 50
buff = 40

# Define GPU - CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.chdir("/home/gilmam/Desktop/study/scheduler/data8x50/gil/set_data/f1/test1")
# device = torch.device("cpu")
tstate = np.load('perfect_state_8x2_1f_20k_set0.npy')
tbuff = np.load('perfect_buffer_8x2_1f_20k_set0.npy')
tlab = np.load('perfect_labels_8x2_1f_20k_set0.npy')
for set_index in range(1,5):
    tmpstate = np.load('perfect_state_8x2_1f_20k_set{}.npy'.format(set_index))
    tmpbuff = np.load('perfect_buffer_8x2_1f_20k_set{}.npy'.format(set_index))
    tmplab = np.load('perfect_labels_8x2_1f_20k_set{}.npy'.format(set_index))
    tstate = np.concatenate((tstate, tmpstate), axis=0)
    tbuff = np.concatenate((tbuff, tmpbuff), axis=0)
    tlab = np.concatenate((tlab, tmplab), axis=0)

ttbuff = np.reshape(tbuff[0:runs], (runs,nbs*nf*2))
ttstate = np.reshape(tstate[0:runs], (runs,nbs*nue+buff*2))
ttlab = np.reshape(tlab[0:runs], (runs,nbs*nf))
ttlabX = ttlab.copy()
######YYYYYY
# inputl = [[ttstate[i][8*ttlab[i][c]+k] for c in range(16) for k in range(8)] for i in range(runs)]
inputl = [[ttstate[i][nbs*ttbuff[i][2*c]+k] for c in range(nbs*nf) for k in range(nbs)] for i in range(runs)]
##########^^^^^^
outlist = []
outputd = ttlab.copy()
for k in range(runs):
    for j in range(nbs*nf):
        outputd[k][j] = np.where(ttlabX[k] == ttbuff[k][2*j])[0][0]
        ttlabX[k][outputd[k][j]] = -2
        # outlist.append(np.int(outputd[k][j]/2))
        # outlist.append(np.mod(outputd[k][j], 2))
oneoutlist = []
for k in range(runs):
    oneout = np.zeros((nbs*nf, nbs*nf))
    for i in range(nbs*nf):
        oneout[i][outputd[k][i]] = 1
    oneoutlist.append(oneout)

inputnp = np.asarray(inputl)
outputnp = np.asarray(oneoutlist)


# Training settings
batch_size = 32

# # MNIST Data set
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=(inputnp,outputnp),
#                                            batch_size=batch_size,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


# Initializing the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 300)
        self.fc2 = nn.Linear(300, 500)
        # self.fcs = nn.Linear(1000, 5000)
        # self.fcs25 = nn.Linear(5000, 5000)
        # self.fcs26 = nn.Linear(5000, 5000)
        self.fc3 = nn.Linear(500, 300)
        self.fc4 = nn.Linear(300, 64)
        self.relu = nn.ReLU()
        # self.relu = nn.Tanh()
        self.dp = nn.Dropout(0.0)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.lsm = nn.LogSoftmax(dim=2)
        # # self.softmax2d = nn.Softmax2d()
        # self.softmax2 = F.log_softmax(dim=2)
        # self.softmax1 = (dim=1)

    def forward(self, xa):
        x = self.relu(self.fc1(xa))
        x = self.dp(self.relu(self.fc2(x)))
        # x = self.dp(self.relu(self.fcs(x)))
        # x = self.dp(self.relu(self.fcs25(x)))
        # x = self.dp(self.relu(self.fcs26(x)))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 8, 8)
        # for ind in range(0):
        #     x = self.softmax1(x.view(batch_size, 16, 16))
        #     x = self.softmax2(x)
        x = self.softmax1(x)
        # x = self.softmax2(x)
        x = self.lsm(x)

        return x

model = Net()
model.to(device)                                  # convert net parameters and buffers to CUDA tensors
# Optimizers definitions
# criterion = nn.NLLLoss()
criterion = nn.MSELoss(reduction='mean')#
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00000)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)#, cooldown=2)

# Training Loop
def train(epoch):
    model.train()
    # for i, x in enumerate(train_loader):
    trainloss=0.
    trainaccurecy = 0
    suffle=np.random.permutation(np.int(runs/32*0.9)*32)
    inputd = torch.from_numpy(inputnp)
    outputd =  torch.from_numpy(outputnp)
    count=0
    for i in range(0, np.int(runs/32*0.9)*32, 32):
        count += 1
        data, target = inputd[suffle[i: i + batch_size]].to(device, dtype=torch.float16), outputd[suffle[i: i + batch_size]].to(device, dtype=torch.float16)
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        targetdig = torch.max(target, 2)[1]
        output = model(data)
        loss = criterion(output, target)
        trainloss += float(loss)
        loss.backward()
        optimizer.step()
        pred = torch.max(output, 2)[1]
        trainaccurecy += pred.eq(targetdig).cpu().sum()
        # if batch_idx % 50 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data))
    trainloss /= count
    train_accurecy.append(trainaccurecy.cpu().numpy()/(count*32*nbs*nf))
    train_loss.append(trainloss)
# Test Loop


def test():
    model.eval()
    test_loss = 0
    correct = 0
    temp = 0
    inputd = torch.from_numpy(inputnp)
    outputd =  torch.from_numpy(outputnp)
    count=0
    with torch.no_grad():
        for i in range(np.int(runs/32*0.9)*32, runs, 32):
            count+=1
            data, target = inputd[i: i + batch_size].to(device, dtype=torch.float16), outputd[i: i + batch_size].to(device, dtype=torch.float16)
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            # sum up batch loss
            targetdig = torch.max(target,2)[1]
            test_loss += criterion(output, target)
            # get the index of the max log-probability
            pred = torch.max(output,2)[1]
            # for a32 in range(32):
            #     for ai in range(16):
            #          aaction[] = ttlab[][pred[ai]]
            correct += pred.eq(targetdig).cpu().sum()
            temp += ((pred-targetdig) == 0).cpu().sum()


    test_loss /=count #len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}%\n'.format(
        test_loss, 100*correct/(count*32*nbs*nf)))
        # 100. * correct / len(test_loader.dataset)))

    # save the result - Cost and Accuracy
    correct_list.append(correct.cpu().numpy()/(count*32*nbs*nf))
    cost_list.append(test_loss.cpu())
    return test_loss


for epoch in range(1, num_epochs+1):
    eta = 0.005
    # if np.mod(epoch, 5) == 0 and epoch > 0:
    #     eta = eta/2
    #     optimizer = optim.Adam(model.parameters(), lr=eta, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    t=time()
    train(epoch)
    print('Finished Training Epoch {} in {} sec' .format(epoch,time()-t))
    val_loss = test()
    scheduler.step(train_loss[-1])
print('Correct list:')
print(correct_list)
print('Cost list:')
print(['%.5f' % x for x in cost_list])
print('Train Accuracy:')
print(train_accurecy)
print('train_loss:')
print(['%.5f' % x for x in train_loss])

plt.figure(1)
plt.plot(np.arange(0, num_epochs), train_loss, 'r', label='train loss')
plt.plot(np.arange(0, num_epochs), cost_list, 'b', label='validation loss')
plt.title('train & validation loss')
plt.xlabel('epoch number')
plt.ylabel('MSE loss')
plt.legend()
plt.savefig('/home/gilmam/Desktop/study/scheduler/data8x50/gil/Loss_fig_perfect_mse_8x2_1f_wd0_10mt.png')

plt.figure(2)
plt.plot(np.arange(0, num_epochs), train_accurecy, 'r', label='train Accuracy')
plt.plot(np.arange(0, num_epochs), correct_list, 'b', label='validation Accuracy')
plt.title('train & validation Accuracy')
plt.xlabel('epoch number')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/home/gilmam/Desktop/study/scheduler/data8x50/gil/figs/Accuracy_fig_perfect_mse_8x2_1f_wd0_10mt.png')

torch.save(model.state_dict(), '/home/gilmam/Desktop/study/scheduler/data8x50/gil/model5m_float16')
gil=10
