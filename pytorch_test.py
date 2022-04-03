"""
Created on Tue Nov 12 11:41:08 2019
@author: shasha
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pulp import *


import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

correct_list = []
cost_list = []
num_epochs = 50

# Define GPU - CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training settings
batch_size = 32

# MNIST Data set
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Initializing the model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=2)
        self.dp1 = nn.Dropout2d(p=0.5, inplace=False)
        self.mp = nn.MaxPool2d(2)
        self.L1 = nn.Linear(360, 180)  # linear 320 => 10
        # self.L2 = nn.Linear(180, 90)
        self.fc = nn.Linear(180, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        # x = self.dp1(x)
        x = F.relu((self.conv3(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.L1(x)
        x = self.dp1(x)
        # x = self.L2(x)
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()
model.to(device)  # convert net parameters and buffers to CUDA tensors
# Optimizers definitions
criterion = F.nll_loss
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)


# Training Loop
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))


# Test Loop
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target, size_average=False)#.data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({}.{}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), correct / 100, torch.fmod(correct, 100)))
    # 100. * correct / len(test_loader.dataset)))

    # save the result - Cost and Accuracy
    correct_list.append(correct.cpu().numpy() / 100)
    cost_list.append(test_loss.cpu().numpy())
    return test_loss


for epoch in range(1, num_epochs + 1):
    eta = 0.0005
    if np.mod(epoch, 5) == 0:
        eta = eta / 2
        optimizer = optim.Adam(model.parameters(), lr=eta, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    train(epoch)
    print('Finished Training Epoch {}'.format(epoch))
    val_loss = test()
    scheduler.step(val_loss)
print('Correct list:')
print(correct_list)
print('Cost list:')
print(['%.5f' % x for x in cost_list])


# import torch
# from torch.autograd import Variable
# import numpy as np
#
#
# def rmse(y, y_hat):
#     """Compute root mean squared error"""
#     return torch.sqrt(torch.mean((y - y_hat).pow(2)))
#
#
# def forward(x, e):
#     """Forward pass for our fuction"""
#     return x.pow(e.repeat(x.size(0)))
#
#
# # Let's define some settings
# n = 1000  # number of examples
# learning_rate = 5e-10
#
# # Model definition
# x = Variable(torch.rand(n) * 10, requires_grad=False)
# y = forward(x, exp)
#
# # Model parameters
# exp = Variable(torch.FloatTensor([2.0]), requires_grad=False)
# exp_hat = Variable(torch.FloatTensor([4]), requires_grad=True)
#
# # Optimizer (NEW)
# opt = torch.optim.SGD([exp_hat], lr=learning_rate, momentum=0.9)
#
# loss_history = []
# exp_history = []
#
# # Training loop
# for i in range(0, 10000):
#     opt.zero_grad()
#     print("Iteration %d" % i)
#
#     # Compute current estimate
#     y_hat = forward(x, exp_hat)
#
#     # Calculate loss function
#     loss = rmse(y, y_hat)
#
#     # Do some recordings for plots
#     loss_history.append(loss.data[0])
#     exp_history.append(y_hat.data[0])
#
#     # Update model parameters
#     loss.backward()
#     opt.step()
#
#     print("loss = %s" % loss.data[0])
#     print("exp = %s" % exp_hat.data[0])


# ###################################################################################3
# class buffer:
#     totalLoss = 0
#     def __init__(self,size=20,nUE=10,rangeTTL=[8, 10, 12, 14, 16, 18]):
#         self.bufferSize = size
#         self.nUE        = nUE
#         self.rangeTTL   = rangeTTL
#         self.reset(0)
#
#     def reset(self,drop):
#         self.buff = []
#         self.rand(drop)
#         return self.status()
#
#     def rand(self,drop):
#         for k in range(self.bufferSize-len(self.buff)):
#             self.buff.append([np.random.randint(self.nUE),np.random.choice(self.rangeTTL)-drop]   )
#
#     def timeTick(self):
#         for k in self.buff:
#             k[1] -= 1
#         packetLoss = [k for k in self.buff if k[1]<0]
#         self.buff = [k for k in self.buff if k[1]>=0]
#         return len(packetLoss)
#
#     def status(self,v=0):
#         if v:
#             return self.buff[0:v]
#         else:
#             return self.buff
#
#     def sort(self,v):
#         def getKey(w):
#             return w[v]
#         return sorted(self.buff,key=getKey)
#
#     def action(self,v):
#         def sort(v):
#             def getKey(w):
#                 return w[1]
#             return sorted(v,key=getKey)
#         for p in v:
#             uu=[(n,k[1]) for n,k in enumerate(self.buff) if k[0]==p]
#             if len(uu):
#                 suu = sort(uu)
#                 self.buff.pop(suu[0][0])
#
#     def step(self,v,drop):
#         self.action(v)
#         loss=self.timeTick()
#         self.rand(drop)
#         self.totalLoss+=loss
#         return loss
#
#
# ###############################################################################
# class env:
#     def __init__(self, fileName, nf=2, trh=4):
#         self.d        = pickle.load(open(fileName+'.pckl','rb'))
#         self.nUE      = self.d['nUE']
#         self.nRecords = self.d['nRecords']
#         self.B        = buffer(nUE=self.nUE,size=nf*10)
#         self.nState   = np.shape(self.state(0))[1]
#         self.trh      = trh
#         self.nAction  = (self.nUE+1)**4
#         self.nEpisode = 0
#
#     def calcCINR(self,ueBeamFreqMatrix,uePowerMatrix,rssi):
#         nBeams=len(ueBeamFreqMatrix[0])
#         e = np.eye(nBeams).astype(int)
#         r=[]
#         for k,w in zip(ueBeamFreqMatrix,uePowerMatrix):
#             t=(rssi[np.abs(k)]+w)*(np.array(k)>=0)
#             P=np.diag(t)
#             inter = t[e==0]
#             snr = P-inter
#             r.append(snr*(np.array(k)>=0))
#         return np.array(r)
#
#     def proc(self,ueBeamFreqMatrix,uePowerMatrix,rssi,trh):
#         t=self.calcCINR(ueBeamFreqMatrix,uePowerMatrix,rssi)
# #        print(t)
#         d=np.abs(ueBeamFreqMatrix)*((t>=trh)*2-1)
#         z = [k for k in d.flatten() if k>=0]
#         return z
#
#     def state(self,n):
#         n=np.mod(n,self.nRecords)
#         self.nEpisode = n
#         rssi = self.d['rssi'][n]
#         b = np.array(self.B.status()).flatten()
#         return np.concatenate((rssi.flatten(),b)).reshape(1,-1)
#
#     def step(self,action,pAction,drop):
#         n=np.mod(self.nEpisode,self.nRecords)
#         rssi = self.d['rssi'][n]
#         a=self.proc(action,pAction,rssi,self.trh)
#         pktLoss = self.B.step(a,drop)
#         self.nEpisode += 1
#         n = np.mod(self.nEpisode, self.nRecords)
#         next_state = self.state(n)
#         pktPass = len(a)
#         return next_state, pktPass, pktLoss
# ###############################################################################
#
#
# if __name__ == "__main__":
#     def sort(v):
#         def getKey(w):
#             return w[1]
#
#         return sorted(v, key=getKey)
#     x = env('/home/gilmam/Desktop/study/scheduler/nUE10_seed1_len2000')
#     graf = np.zeros((4,9))
#     graf[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
#     for drop in range(9):
#
#         totalpass = 0
#         totalloss = 0
#         totalpower = 0
#         for run in range(2000):
#             state = x.state(run)
#             sortedbuffer = sort(x.B.buff)
#             choosen = sortedbuffer[0:4]
#
#             ####################     Exhaustive search    ######################
#
#             max_reward=0
#             for i1 in range(4):
#                 for i2 in range(4):
#                     for i3 in range(4):
#                         for i4 in range(4):
#                             if i1!=i2 and i1!=i3 and i1!=i4 and i3!=i2 and i4!=i2 and i3!=i4:
#                                 reward = 0
#                                 powerop=np.array([[0.,0.],[0.,0.]])
#                                 aop=np.array([[choosen[i1][0], choosen[i2][0]],[choosen[i3][0],choosen[i4][0]]])
#                                 A_1 = state[0][choosen[i1][0]*2] - state[0][choosen[i1][0]*2+1] + state[0][choosen[i2][0]*2+1] - state[0][choosen[i2][0]*2]
#                                 A_2 = state[0][choosen[i3][0]*2] - state[0][choosen[i3][0]*2+1] + state[0][choosen[i4][0]*2+1] - state[0][choosen[i4][0]*2]
#                                 if A_1<8:
#                                     if choosen[i1][1]<choosen[i2][1]:
#                                         if state[0][choosen[i1][0]*2]-state[0][choosen[i1][0]*2+1]>0:
#                                             aop[0][1] = -1
#                                         else:
#                                             aop[0][1] = aop[0][0]
#                                             aop[0][0] = -1
#                                     else:
#                                         if state[0][choosen[i2][0] * 2+1] - state[0][choosen[i2][0] * 2] > 0:
#                                             aop[0][0] = -1
#                                         else:
#                                             aop[0][0] = aop[0][1]
#                                             aop[0][1] = -1
#                                 else:
#                                     if state[0][choosen[i1][0]*2] - state[0][choosen[i1][0]*2+1]<4:
#                                         powerop[0][0] = -(state[0][choosen[i1][0]*2] - state[0][choosen[i1][0]*2+1]-4)
#                                     if state[0][choosen[i2][0]*2+1] - state[0][choosen[i2][0]*2]<4:
#                                         powerop[0][1] = -(state[0][choosen[i2][0]*2+1] - state[0][choosen[i2][0]*2]-4)
#
#
#                                 if A_2<8:
#                                     if choosen[i3][1] < choosen[i4][1]:
#                                         if state[0][choosen[i3][0] * 2] - state[0][choosen[i3][0] * 2 + 1] > 0:
#                                             aop[1][1] = -1
#                                         else:
#                                             aop[1][1] = aop[1][0]
#                                             aop[1][0] = -1
#                                     else:
#                                         if state[0][choosen[i4][0] * 2 + 1] - state[0][choosen[i4][0] * 2] > 0:
#                                             aop[1][0] = -1
#                                         else:
#                                             aop[1][0] = aop[1][1]
#                                             aop[1][1] = -1
#                                 else:
#                                     if state[0][choosen[i3][0] * 2] - state[0][choosen[i3][0] * 2 + 1] < 4:
#                                         powerop[1][0] = -(state[0][choosen[i3][0] * 2] - state[0][choosen[i3][0] * 2 + 1] - 4)
#                                     if state[0][choosen[i4][0] * 2 + 1] - state[0][choosen[i4][0] * 2] < 4:
#                                         powerop[1][1] = -(state[0][choosen[i4][0] * 2 + 1] - state[0][choosen[i4][0] * 2] - 4)
#                                 reward = 50*(1*(aop[0][0]>=0) + 1*(aop[0][1]>=0) + 1*(aop[1][0]>=0) + 1*(aop[1][1]>=0))-(powerop[0][0] + powerop[0][1] + powerop[1][0] + powerop[1][1])
#                                 if reward>max_reward:
#                                     max_reward = reward
#                                     action = aop
#                                     pAction = powerop
#
#             if action[0][0] == -1:
#                 for kkk in sortedbuffer[4:]:
#                     if state[0][kkk[0]*2]-state[0][kkk[0]*2+1]+state[0][action[0][1]*2+1]-state[0][action[0][1]*2]>8:
#                         action[0][0] = kkk[0]
#                         if state[0][action[0][0]*2]-state[0][action[0][0]*2+1]<4:
#                             pAction[0][0] = -(state[0][action[0][0]*2]-state[0][action[0][0]*2+1]-4)
#                         if state[0][action[0][1]*2+1]-state[0][action[0][1]*2]<4:
#                             pAction[0][1] = -(state[0][action[0][1]*2+1]-state[0][action[0][1]*2]-4)
#                         break
#             if action[0][1] == -1:
#                 for kkk in sortedbuffer[4:]:
#                     if state[0][kkk[0] * 2+1] - state[0][kkk[0] * 2] + state[0][action[0][0] * 2] - state[0][action[0][0] * 2+1] > 8:
#                         action[0][1] = kkk[0]
#                         if state[0][action[0][1] * 2+1] - state[0][action[0][1] * 2] < 4:
#                             pAction[0][1] = -(state[0][action[0][1] * 2+1] - state[0][action[0][1] * 2] - 4)
#                         if state[0][action[0][0] * 2] - state[0][action[0][0] * 2+1] < 4:
#                             pAction[0][0] = -(state[0][action[0][0] * 2] - state[0][action[0][0] * 2+1] - 4)
#                         break
#             if action[1][0] == -1:
#                 for kkk in sortedbuffer[4:]:
#                     if state[0][kkk[0] * 2] - state[0][kkk[0] * 2 + 1] + state[0][action[1][1] * 2 + 1] - state[0][action[1][1] * 2] > 8:
#                         action[1][0] = kkk[0]
#                         if state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] < 4:
#                             pAction[1][0] = -(state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] - 4)
#                         if state[0][action[1][1] * 2 + 1] - state[0][action[1][1] * 2] < 4:
#                             pAction[1][1] = -(state[0][action[1][1] * 2+1] - state[0][action[1][1] * 2 ] - 4)
#                         break
#             if action[1][1] == -1:
#                 for kkk in sortedbuffer[4:]:
#                     if state[0][kkk[0] * 2+1] - state[0][kkk[0] * 2] + state[0][action[1][0] * 2] - state[0][action[1][0] * 2+1] > 8:
#                         action[1][1] = kkk[0]
#                         if state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] < 4:
#                             pAction[1][0] = -(state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] - 4)
#                         if state[0][action[1][1] * 2 + 1] - state[0][action[1][1] * 2] < 4:
#                             pAction[1][1] = -(state[0][action[1][1] * 2+1] - state[0][action[1][1] * 2] - 4)
#                         break
#             totalpower += pAction[0][0] + pAction[0][1] + pAction[1][0] + pAction[1][1]
#             next_state, pktPass, pktLoss = x.step(action, pAction,drop)
#             totalpass += pktPass
#             totalloss += pktLoss
#          #   print(pktPass, pktLoss)
#
#         print('average TTL: ',(13-drop))
#         print('throughput: '+str(totalpass/2000))
#         graf[1][drop] = totalpass / 2000
#         print('packet loss: ',(totalloss/2000))
#         graf[2][drop] = totalloss / 2000
#         print("average power: ",(totalpower/2000))
#         graf[3][drop] = totalpower / 2000
#
#         #########################3    perfect match     ######################
#     nbs = 2
#     nf = 2
#     graf_perfect = np.zeros((4, 9))
#     graf_perfect[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
#     for drop in range(9):
#         xp = env('/home/gilmam/Desktop/study/scheduler/nUE10_seed1_len2000', nf=nf)
#         totalpass = 0
#         totalloss = 0
#         totalpower = 0
#         for run in range(2000):
#             state = xp.state(run)
#             sortedbuffer = sort(xp.B.buff)
#             choosen = sortedbuffer[0:2 * nf]
#             w = np.zeros((2 * nf, 2 * nf))
#             xx = np.zeros((2 * nf, 2 * nf))
#             action = np.array([[0, 0] for i in range(nf)])
#             pAction = np.array([[0., 0.] for i in range(nf)])
#
#             ####################     perfect match      ######################
#
#             for i in range(2 * nf):
#                 for j in range(2 * nf):
#                     if np.mod(i, 2) == 0:
#                         w[i][j] = state[0][choosen[j][0] * 2] - state[0][choosen[j][0] * 2 + 1]
#                     else:
#                         w[i][j] = state[0][choosen[j][0] * 2 + 1] - state[0][choosen[j][0] * 2]
#
#             problem = LpProblem("PerfectMatch", LpMaximize)
#
#                 ###############################################################################   LP
#
#                 # factories
#
#             x = LpVariable.dicts("x", list(range(nf * nf * 4)), 0, 1, cat="Continuous")
#
#                 # goal constraint[0,1,2]
#             for i in range(nf * 2):
#                 problem += pulp.lpSum(x[j] for j in range(2 * nf * i, 2 * nf * (i + 1))) == 1
#                 problem += pulp.lpSum(x[j] for j in range(i, nf * nf * 4, nf * 2)) == 1
#
#                 # objective function
#             problem += pulp.lpSum([x[2 * nf * i + j] * w[i][j] for i in range(2 * nf) for j in range(2 * nf)])
#
#                 # print(problem)
#
#                 # solving
#             problem.solve()
#
#             clk = np.array([[0, 0] for i in range(nf)])
#             for i1 in range(2 * nf):
#                 for i2 in range(2 * nf):
#                     if x[2 * nf * i1 + i2].varValue == 1:
#                         odd = np.mod(i1, 2)
#                         index = int((i1 - odd) / 2)
#                         action[index][odd] = choosen[i2][0]
#                         clk[index][odd] = choosen[i2][1]
#
#             toolate = []
#
#             for i in range(nf):
#                 A_i = state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] + state[0][action[i][1] * 2 + 1] - \
#                       state[0][action[i][1] * 2]
#                 if A_i < 8:
#                     if clk[i][1] < clk[i][0]:
#                         if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] > 0:
#                             action[i][0] = -1
#                         else:
#                             action[i][0] = action[i][1]
#                             action[i][1] = -1
#                     else:
#                         if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] > 0:
#                             action[i][1] = -1
#                         else:
#                             action[i][1] = action[i][0]
#                             action[i][0] = -1
#                 else:
#                     if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
#                         pAction[i][0] = -(state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
#                     if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
#                         pAction[i][1] = -(state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)
#
#                 if action[i][0] == -1:
#                     for kkk in sortedbuffer[2 * nf:]:
#                         flag = 1
#                         for j in toolate:
#                             if kkk == j:
#                                 flag = 0
#                                 break
#                         if flag:
#                             if state[0][kkk[0] * 2] - state[0][kkk[0] * 2 + 1] + state[0][action[i][1] * 2 + 1] - \
#                                     state[0][action[i][1] * 2] > 8:
#                                 action[i][0] = kkk[0]
#                                 toolate.append(kkk)
#                                 if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
#                                     pAction[i][0] = -(
#                                             state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
#                                 if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
#                                     pAction[i][1] = -(
#                                             state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)
#                                 break
#                 if action[i][1] == -1:
#                     for kkk in sortedbuffer[2 * nf:]:
#                         flag = 1
#                         for j in toolate:
#                             if kkk == j:
#                                 flag = 0
#                                 break
#                         if flag:
#                             if state[0][kkk[0] * 2 + 1] - state[0][kkk[0] * 2] + state[0][action[i][0] * 2] - \
#                                     state[0][
#                                         action[i][0] * 2 + 1] > 8:
#                                 action[i][1] = kkk[0]
#                                 toolate.append(kkk)
#                                 if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
#                                     pAction[i][0] = -(
#                                             state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
#                                 if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
#                                     pAction[i][1] = -(
#                                             state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)
#                                 break
#
#                 # for i in range(nf):
#                 #     if clk[i][1] < clk[i][0]:
#                 #         if state[0][action[i][1]*2+1]-state[0][action[i][1]*2]<4:
#                 #             pAction[i][1] = -(state[0][action[i][1]*2+1]-state[0][action[i][1]*2]-4)
#                 #         else:
#                 #             if state[0][action[i][0]*2]-state[0][action[i][0]*2+1]<4:
#                 #                 pAction[i][0] = -(state[0][action[i][0]*2]-state[0][action[i][0]*2+1]-4)
#                 #     else:
#                 #         if state[0][action[i][0]*2]-state[0][action[i][0]*2+1]<4:
#                 #             pAction[i][0] = -(state[0][action[i][0]*2]-state[0][action[i][0]*2+1]-4)
#                 #         else:
#                 #             if state[0][action[i][1]*2+1]-state[0][action[i][1]*2]<4:
#                 #                 pAction[i][1] = -(state[0][action[i][1]*2+1]-state[0][action[i][1]*2]-4)
#
#             totalpower += np.sum(pAction)
#             next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
#             totalpass += pktPass
#             totalloss += pktLoss
#             #print(run)
#
#         print('average TTL: ', (13 - drop))
#         print('throughput: ' + str(totalpass / 2000))
#         graf_perfect[1][drop] = (totalpass / 2000)
#         print('packet loss: ', (totalloss / 2000))
#         graf_perfect[2][drop] = totalloss / 2000
#         print("average power: ", (totalpower / 2000))
#         graf_perfect[3][drop] = (totalpower / 2000)
#     #########################    random algo    ######################
#
#     xr = env('/home/gilmam/Desktop/study/scheduler/nUE10_seed1_len2000')
#     graf_rand = np.zeros((4,9))
#     graf_rand[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
#     for drop in range(9):
#
#         totalpass = 0
#         totalloss = 0
#         totalpower = 0
#         for run in range(2000):
#             state = xr.state(run)
#             sortedbuffer = sort(xr.B.buff)
#             choosen = sortedbuffer[0:4]
#             action = np.array([[choosen[0][0], choosen[1][0]], [choosen[2][0], choosen[3][0]]])
#             pAction = np.array([[0., 0.], [0., 0.]])
#
#             if choosen[0][1] > choosen[1][1]:
#                 if state[0][action[0][1]*2+1]-state[0][action[0][1]*2]<4:
#                     pAction[0][1] = -(state[0][action[0][1]*2+1]-state[0][action[0][1]*2]-4)
#                 else:
#                     if state[0][action[0][0]*2]-state[0][action[0][0]*2+1]<4:
#                         pAction[0][0] = -(state[0][action[0][0]*2]-state[0][action[0][0]*2+1]-4)
#             else:
#                 if state[0][action[0][0]*2]-state[0][action[0][0]*2+1]<4:
#                     pAction[0][0] = -(state[0][action[0][0]*2]-state[0][action[0][0]*2+1]-4)
#                 else:
#                     if state[0][action[0][1]*2+1]-state[0][action[0][1]*2]<4:
#                         pAction[0][1] = -(state[0][action[0][1]*2+1]-state[0][action[0][1]*2]-4)
#
#             if choosen[2][1] > choosen[3][1]:
#                 if state[0][action[1][1]*2+1]-state[0][action[1][1]*2]<4:
#                     pAction[1][1] = -(state[0][action[1][1]*2+1]-state[0][action[1][1]*2]-4)
#                 else:
#                     if state[0][action[1][0]*2]-state[0][action[1][0]*2+1]<4:
#                         pAction[1][0] = -(state[0][action[1][0]*2]-state[0][action[1][0]*2+1]-4)
#             else:
#                 if state[0][action[1][0]*2]-state[0][action[1][0]*2+1]<4:
#                     pAction[1][0] = -(state[0][action[1][0]*2]-state[0][action[1][0]*2+1]-4)
#                 else:
#                     if state[0][action[1][1]*2+1]-state[0][action[1][1]*2]<4:
#                         pAction[1][1] = -(state[0][action[1][1]*2+1]-state[0][action[1][1]*2]-4)
#
#
#
#
#             totalpower += pAction[0][0] + pAction[0][1] + pAction[1][0] + pAction[1][1]
#             next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
#             totalpass += pktPass
#             totalloss += pktLoss
#           #   print(pktPass, pktLoss)
#
#         print('average TTL: ', (13-drop))
#         print('throughput: '+str(totalpass/2000))
#         graf_rand[1][drop] = totalpass / 2000
#         print('packet loss: ', (totalloss/2000))
#         graf_rand[2][drop] = totalloss / 2000
#         print("average power: ", (totalpower/2000))
#         graf_rand[3][drop] = totalpower / 2000
#
#         #########################3    stable match     ######################
#         nf = 2
#
#         graf_stable = np.zeros((4, 9))
#         graf_stable[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
#         for drop in range(9):
#             xp = env('/home/gilmam/Desktop/study/scheduler/nUE10_seed1_len2000', nf=nf)
#             totalpass = 0
#             totalloss = 0
#             totalpower = 0
#             for run in range(2000):
#                 state = xp.state(run)
#                 sortedbuffer = sort(xp.B.buff)
#                 choosen = sortedbuffer[0:2 * nf]
#                 xxue = (-1) * np.ones((1, 2 * nf), dtype=int)
#                 loc = np.zeros((2 * nf, 2 * nf), dtype=int)
#                 ue = np.zeros((2 * nf, 2 * nf), dtype=int)
#                 action = np.array([[0, 0] for i in range(nf)])
#                 pAction = np.array([[0., 0.] for i in range(nf)])
#
#                 ####################     data preperetion      ######################
#
#                 for i in range(2 * nf):
#                     temp_dvec = np.array([[ii, 0] for ii in range(nf * 2)])
#                     for j in range(2 * nf):
#                         if np.mod(i, 2) == 0:
#                             temp_dvec[j][1] = state[0][choosen[j][0] * 2] - state[0][choosen[j][0] * 2 + 1]
#                         else:
#                             temp_dvec[j][1] = state[0][choosen[j][0] * 2 + 1] - state[0][choosen[j][0] * 2]
#                     # print(temp_dvec)
#                     sortedd = sort(temp_dvec)
#                     # print(sortedd)
#                     sorteddd = [sortedd[iii][0] for iii in range(2 * nf)]
#                     sorteddd.reverse()
#                     loc[i] = sorteddd
#
#                 for uei in range(2 * nf):
#                     bs1 = [i for i in range(0, 2 * nf, 2)]
#                     bs2 = [i for i in range(1, 2 * nf, 2)]
#                     if state[0][choosen[uei][0] * 2] - state[0][choosen[uei][0] * 2 + 1] > 0:
#                         ue[uei] = bs1 + bs2
#                     else:
#                         ue[uei] = bs2 + bs1
#
#                 ####################     loc ask from ue    ######################
#
#                 for i in range(nf * 2):  # i is col (priorety)
#                     for j in range(nf * 2):  # j is row (loc's)
#                         if np.sum(xxue[0] == j) == 0:
#                             if xxue[0][loc[j][i]] == -1:
#                                 xxue[0][loc[j][i]] = j
#                             else:
#                                 if int(np.where(ue[loc[j][i]] == xxue[0][loc[j][i]])[0]) > int(
#                                         np.where(ue[loc[j][i]] == j)[0]):
#                                     xxue[0][loc[j][i]] = j
#                     if np.sum(xxue[0] == -1) == 0:
#                         break
#
#                 ####################     set the action matrix    ######################
#                 clk = np.array([[0, 0] for i in range(nf)])
#                 for i in range(2 * nf):
#                     odd1 = np.mod(xxue[0][i], 2)
#                     freqi = int((xxue[0][i] - odd1) / 2)
#                     action[freqi][odd1] = choosen[i][0]
#                     clk[freqi][odd1] = choosen[i][1]
#
#                 ####################     set the power matrix and change the pkt which not sent    ######################
#                 toolate = []
#
#                 for i in range(nf):
#                     A_i = state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] + state[0][action[i][1] * 2 + 1] - \
#                           state[0][action[i][1] * 2]
#                     if A_i < 8:
#                         if clk[i][1] < clk[i][0]:
#                             if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] > 0:
#                                 action[i][0] = -1
#                             else:
#                                 action[i][0] = action[i][1]
#                                 action[i][1] = -1
#                         else:
#                             if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] > 0:
#                                 action[i][1] = -1
#                             else:
#                                 action[i][1] = action[i][0]
#                                 action[i][0] = -1
#                     else:
#                         if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
#                             pAction[i][0] = -(state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
#                         if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
#                             pAction[i][1] = -(state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)
#
#                     if action[i][0] == -1:
#                         for kkk in sortedbuffer[2 * nf:]:
#                             flag = 1
#                             for j in toolate:
#                                 if kkk == j:
#                                     flag = 0
#                                     break
#                             if flag:
#                                 if state[0][kkk[0] * 2] - state[0][kkk[0] * 2 + 1] + state[0][action[i][1] * 2 + 1] - \
#                                         state[0][action[i][1] * 2] > 8:
#                                     action[i][0] = kkk[0]
#                                     toolate.append(kkk)
#                                     if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
#                                         pAction[i][0] = -(
#                                                     state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
#                                     if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
#                                         pAction[i][1] = -(
#                                                     state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)
#                                     break
#                     if action[i][1] == -1:
#                         for kkk in sortedbuffer[2 * nf:]:
#                             flag = 1
#                             for j in toolate:
#                                 if kkk == j:
#                                     flag = 0
#                                     break
#                             if flag:
#                                 if state[0][kkk[0] * 2 + 1] - state[0][kkk[0] * 2] + state[0][action[i][0] * 2] - \
#                                         state[0][
#                                             action[i][0] * 2 + 1] > 8:
#                                     action[i][1] = kkk[0]
#                                     toolate.append(kkk)
#                                     if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
#                                         pAction[i][0] = -(
#                                                     state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
#                                     if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
#                                         pAction[i][1] = -(
#                                                     state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)
#                                     break
#
#                 totalpower += np.sum(pAction)
#                 next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
#                 totalpass += pktPass
#                 totalloss += pktLoss
#                 #print(run)
#
#             print('average TTL: ', (13 - drop))
#             print('throughput: ' + str(totalpass / 2000))
#             graf_stable[1][drop] = (totalpass / 2000)
#             print('packet loss: ', (totalloss / 2000))
#             graf_stable[2][drop] = totalloss / 2000
#             print("average power: ", (totalpower / 2000))
#             graf_stable[3][drop] = (totalpower / 2000)
#
#
#
#     plt.figure(1)
#     Exhaustive_search, = plt.plot(graf[0], graf[1], 'r', label='Exhaustive search')
#     Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[1], 'b', label='Perfect match')
#     random_scheduler, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='Random')
#     Stable_match, = plt.plot(graf_stable[0], graf_stable[1], 'y', label='Stable match')
#     plt.xlabel('average TTL')
#     plt.ylabel('average throughput')
#     plt.title('throughput vs. average TTL')
#     plt.legend()
#     plt.grid()
#
#     plt.figure(2)
#     Exhaustive_search, = plt.plot(graf[0], graf[2], 'r', label='Exhaustive search')
#     Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[2], 'b', label='Perfect match')
#     random_scheduler, = plt.plot(graf_rand[0], graf_rand[2], 'g', label='Random')
#     Stable_match, = plt.plot(graf_stable[0], graf_stable[2], 'y', label='Stable match')
#     plt.xlabel('average TTL')
#     plt.ylabel('average packet loss')
#     plt.title('packet loss vs. average TTL')
#     plt.legend()
#     plt.grid()
#
#     plt.figure(3)
#     Exhaustive_search, = plt.plot(graf[0], graf[3], 'r', label='Exhaustive search')
#     Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[3], 'b', label='Perfect match')
#     random_scheduler, = plt.plot(graf_rand[0], graf_rand[3], 'g', label='Random')
#     Stable_match, = plt.plot(graf_stable[0], graf_stable[3], 'y', label='Stable match')
#     plt.xlabel('average TTL')
#     plt.ylabel('average power')
#     plt.title('power vs. average TTL')
#     plt.legend()
#     plt.grid()
#
#     plt.show
#     gil=10   # print('trhoput:' + (totalpass / 2000) + 'packet loss:' + (totalloss / 2000) + 'avarge power:' + (totalpower / 2000))