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

correct_list = []
cost_list = []
train_accurecy = []
train_loss =[]
correct_list_train = []
cost_list_train = []
num_epochs = 60
runs = 20000

# Define GPU - CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
tstate = np.load('perfect_state_2x2_20k.npy')
tbuff = np.load('perfect_buffer_2x2_20k.npy')
tlab = np.load('perfect_labels_2x2_20k.npy')
ttbuff = np.reshape(tbuff, (runs,8))
ttstate = np.reshape(tstate, (runs,60))
ttlab = np.reshape(tlab, (runs,4))
ttlabX = ttlab.copy()
# inputl = [[ttstate[i][2*ttbuff[i][0]], ttstate[i][2*ttbuff[i][0]+1], ttbuff[i][1], ttstate[i][2*ttbuff[i][2]], ttstate[i][2*ttbuff[i][2]+1], ttbuff[i][3], ttstate[i][2*ttbuff[i][4]], ttstate[i][2*ttbuff[i][4]+1], ttbuff[i][5], ttstate[i][2*ttbuff[i][6]], ttstate[i][2*ttbuff[i][6]+1], ttbuff[i][7]] for i in range(2000)]
inputl = [[ttstate[i][2*ttbuff[i][0]] - ttstate[i][2*ttbuff[i][0]+1], ttstate[i][2*ttbuff[i][2]] - ttstate[i][2*ttbuff[i][2]+1], ttstate[i][2*ttbuff[i][4]] - ttstate[i][2*ttbuff[i][4]+1], ttstate[i][2*ttbuff[i][6]] - ttstate[i][2*ttbuff[i][6]+1]] for i in range(runs)]
outlist = []
outputd = ttlab.copy()
for k in range(runs):
    for j in range(4):
        outputd[k][j] = np.where(ttlabX[k] == ttbuff[k][2*j])[0][0]
        ttlabX[k][outputd[k][j]] = -2
        outlist.append(np.int(outputd[k][j]/2))
        outlist.append(np.mod(outputd[k][j], 2))
oneoutlist = []
for k in range(runs):
    oneout = np.zeros((4, 4))
    for i in range(4):
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
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 300)
        self.fcs = nn.Linear(300, 300)
        self.fcs1 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 50)
        self.fc4 = nn.Linear(50, 16)
        self.relu = nn.ReLU()
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

        # # self.softmax2d = nn.Softmax2d()
        # self.softmax2 = F.log_softmax(dim=2)
        # self.softmax1 = (dim=1)

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fcs(x))
        x = self.relu(self.fcs1(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(batch_size, 4, 4)
        for ind in range(0):
            x = self.softmax1(x)
            x = self.softmax2(x)
        x = self.softmax1(x)
        x = F.log_softmax(x, dim=2)

        return x

model = Net()
model.to(device)                                  # convert net parameters and buffers to CUDA tensors
# Optimizers definitions
# criterion = nn.MSELoss(reduction='mean')
criterion = nn.MSELoss(reduction='mean')
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00000)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)#, cooldown=2)

# Training Loop
def train(epoch):
    model.train()
    # for i, x in enumerate(train_loader):
    trainloss=0.
    trainaccurecy = 0
    suffle=np.random.permutation(18000)
    inputd = torch.from_numpy(inputnp)
    outputd =  torch.from_numpy(outputnp)
    count=0
    for i in range(0, 17983, 32):
        count += 1
        data, target = inputd[suffle[i: i + batch_size]].to(device, dtype=torch.float), outputd[suffle[i: i + batch_size]].to(device, dtype=torch.float)
        data, target = Variable(data), Variable(target)
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
    train_accurecy.append(trainaccurecy.cpu().numpy()/(count*32*4))
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
    for i in range(18000,19983,32):
        count+=1
        data, target = inputd[i: i + batch_size].to(device, dtype=torch.float), outputd[i: i + batch_size].to(device, dtype=torch.float)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        targetdig = torch.max(target,2)[1]
        test_loss += criterion(output, target)
        # get the index of the max log-probability
        pred = torch.max(output,2)[1]
        correct += pred.eq(targetdig).cpu().sum()
        temp += ((pred-targetdig) == 0).cpu().sum()

    test_loss /=count #len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}\n'.format(
        test_loss, correct, (count*32*4)))
        # 100. * correct / len(test_loader.dataset)))

    # save the result - Cost and Accuracy
    correct_list.append(correct.cpu().numpy()/(count*32*4))
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
plt.savefig('/home/gilmam/Desktop/study/scheduler/figures/NN_Supervized/multi_softmax/Loss_fig_perfect_mse_wd0_20k_8soft.png')

plt.figure(2)
plt.plot(np.arange(0, num_epochs), train_accurecy, 'r', label='train Accuracy')
plt.plot(np.arange(0, num_epochs), correct_list, 'b', label='validation Accuracy')
plt.title('train & validation Accuracy')
plt.xlabel('epoch number')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/home/gilmam/Desktop/study/scheduler/figures/NN_Supervized/multi_softmax/Accuracy_fig_perfect_mse_wd0_20k_8soft.png')
