"""
Created on Tue Nov 12 11:41:08 2019
@author: shasha
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pulp import *
import copy
import os
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


###################################################################################3
class buffer:
    totalLoss = 0
    def __init__(self,size=80,nUE=50,rangeTTL=[8, 10, 12, 14, 16, 18]):
        self.bufferSize = size
        self.nUE        = nUE
        self.rangeTTL   = rangeTTL
        self.reset(0)

    def reset(self,drop):
        self.buff = []
        self.rand(drop)
        return self.status()

    def rand(self,drop):
        for k in range(self.bufferSize-len(self.buff)):
            self.buff.append([np.random.randint(self.nUE),np.random.choice(self.rangeTTL)-drop]   )

    def timeTick(self):
        for k in self.buff:
            k[1] -= 1
        packetLoss = [k for k in self.buff if k[1]<0]
        self.buff = [k for k in self.buff if k[1]>=0]
        return len(packetLoss)

    def status(self,v=0):
        if v:
            return self.buff[0:v]
        else:
            return self.buff

    def sort(self,v):
        def getKey(w):
            return w[v]
        return sorted(self.buff,key=getKey)

    def action(self,v):
        def sort(v):
            def getKey(w):
                return w[1]
            return sorted(v,key=getKey)
        for p in v:
            uu=[(n,k[1]) for n,k in enumerate(self.buff) if k[0]==p]
            if len(uu):
                suu = sort(uu)
                self.buff.pop(suu[0][0])

    def step(self,v,drop):
        self.action(v)
        loss=self.timeTick()
        self.rand(drop)
        self.totalLoss+=loss
        return loss


###############################################################################
class env:
    def __init__(self, fileName, nf=2, trh=4):
        self.d        = pickle.load(open(fileName+'.pckl','rb'))
        self.nUE      = self.d['nUE']
        self.nRecords = self.d['nRecords']
        self.B        = buffer(nUE=self.nUE,size=nf*40)
        self.nState   = np.shape(self.state(0))[1]
        self.trh      = trh
        self.nAction  = (self.nUE+1)**4
        self.nEpisode = 0

    def calcCINR(self,ueBeamFreqMatrix,uePowerMatrix,rssi):
        nBeams=len(ueBeamFreqMatrix[0])
        e = np.eye(nBeams).astype(int)
        r=[]
        for k,w in zip(ueBeamFreqMatrix,uePowerMatrix):
            t=(rssi[np.abs(k)]+w)*(np.array(k)>=0)
            P=np.diag(t)
            inter = t[e==0]
            max_inter = [np.max(inter[(i):(i+nBeams-1)]) for i in range(0,nBeams*(nBeams-1),(nBeams-1))]
            snr = P-max_inter
            r.append(snr*(np.array(k)>=0))
        return np.array(r)

    def proc(self,ueBeamFreqMatrix,uePowerMatrix,rssi,trh):
        t=self.calcCINR(ueBeamFreqMatrix,uePowerMatrix,rssi)
#        print(t)
        d=np.abs(ueBeamFreqMatrix)*((t>=trh)*2-1)
        z = [k for k in d.flatten() if k>=0]
        return z

    def state(self,n):
        n=np.mod(n,self.nRecords)
        self.nEpisode = n
        rssi = self.d['rssi'][n]
        b = np.array(self.B.status()).flatten()
        return np.concatenate((rssi.flatten(),b)).reshape(1,-1)

    def step(self,action,pAction,drop):
        n=np.mod(self.nEpisode,self.nRecords)
        rssi = self.d['rssi'][n]
        a=self.proc(action,pAction,rssi,self.trh)
        pktLoss = self.B.step(a,drop)
        self.nEpisode += 1
        n = np.mod(self.nEpisode, self.nRecords)
        next_state = self.state(n)
        pktPass = len(a)
        return next_state, pktPass, pktLoss
###############################################################################


if __name__ == "__main__":
    def sort(v):
        def getKey(w):
            return w[1]

        return sorted(v, key=getKey)
    #######################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            # x = x.view(batch_size, 16, 16)
            # for ind in range(0):
            #     x = self.softmax1(x.view(batch_size, 16, 16))
            #     x = self.softmax2(x)
            x = self.softmax1(x.view(-1, 8, 8))
            # x = self.softmax2(x)
            x = self.lsm(x)

            return x
    os.chdir("/home/gilmam/Desktop/study/scheduler/data8x50/gil")
    # graf_perfect = np.load('graf_perfect.npy')
    model = Net()
    model123 = torch.load('/home/gilmam/Desktop/study/scheduler/data8x50/gil/model_1f_10m')#/home/gilmam/Desktop/study/scheduler/data8x50/gil
    model.load_state_dict(model123)
    # data = torch.load('data.pt')
    model.to(device)

    #     #########################3    perfect match-NN     ######################
    # graf_perfect_NN = np.zeros((4, 9))
    # graf_perfect_NN[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    # runs = 1000
    # nbs = 8
    # nf = 100
    # for drop in range(9):
    #     xp = env('/home/gilmam/Desktop/study/scheduler/data8x50/gil/nUE50nRu4_seed1_len2000', nf=nf)
    #     totalpass = 0
    #     totalloss = 0
    #     totalpower = 0
    #     for run in range(runs):
    #         state = xp.state(run)
    #         sortedbuffer = sort(xp.B.buff)
    #         choosen = sortedbuffer[0:nbs]
    #         action = np.array([[-1, -1, -1, -1, -1, -1, -1, -1] for i in range(nf)])
    #         pAction = np.array([[0., 0., 0., 0., 0., 0., 0., 0.] for i in range(nf)])
    #         clk = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
    #
    #         #         #######################   predata for NN    ######################
    #         for nff in range(nf):
    #
    #             inputl = [state[0][choosen[i][0]*nbs+k] for i in range(nbs) for k in range(nbs)]
    #             inputnp = np.asarray(inputl)
    #             inputd = torch.from_numpy(inputnp)
    #             data = inputd.to(device, dtype=torch.float)
    #             with torch.no_grad():
    #                 out = model(data)
    #             pred = torch.max(out, 2)[1]
    #             no_match=[]
    #             for ru in range(nbs):
    #                 ind2 = pred.cpu().numpy()[0][ru]
    #                 if action[nff][ind2]==-1: #if the slot is aveilable
    #                     action[nff][ind2] = choosen[ru][0]
    #                     clk[nff][ind2] = choosen[ru][1]
    #                 else:
    #                     if clk[nff][ind2] > choosen[ru][1]:
    #                         no_match.append([action[nff][ind2], clk[nff][ind2]])
    #                         action[nff][ind2] = choosen[ru][0]
    #                         clk[nff][ind2] = choosen[ru][1]
    #                     else:
    #                         no_match.append(choosen[ru])
    #
    #             for ru2 in range(8):
    #                 if action[nff][ru2]==-1:
    #                     temp=no_match.pop()
    #                     action[nff][ru2]=temp[0]
    #                     clk[nff][ru2]=temp[1]
    #     #
    #     # #############    pwoer calc      #############3
    #             cng = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    #             e = np.eye(nbs).astype(int)
    #             t = np.zeros((8,8))
    #             # for fr in range(1):
    #             ind_sort = np.argsort(clk[nff])
    #             t = np.array([state[0][(action[nff][i]) * nbs:action[nff][i] * nbs + nbs] for i in range(nbs)])
    #             for kk in range(nbs):
    #                 ti = t + np.tile(pAction[nff], (nbs, 1))
    #                 inter = ti[e == 0]
    #                 max_inter = np.array([np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
    #                 difm = ti - np.reshape(max_inter, (nbs, 1))-4
    #                 if difm[ind_sort[kk]][ind_sort[kk]]<0: # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
    #                     flagw=1
    #                     for cp in range(kk):
    #                         if cng[ind_sort[cp]]==0:
    #                             if difm[ind_sort[cp]][ind_sort[cp]] - difm[ind_sort[cp]][ind_sort[kk]] + difm[ind_sort[kk]][
    #                                 ind_sort[kk]] < 4.00000001:
    #                                 flagw = 0
    #                                 cng[ind_sort[kk]] = 1
    #                                 indi = choosen.index([action[nff][ind_sort[kk]], clk[nff][ind_sort[kk]]])
    #                                 del choosen[indi]
    #                                 action[nff][ind_sort[kk]] = -1
    #                                 t[ind_sort[kk], :] = 0
    #
    #                                 break
    #                     if flagw:
    #                         pAction[nff,ind_sort[kk]] = -difm[ind_sort[kk]][ind_sort[kk]]
    #     #
    #     #         #############    cng from the buffer     #############3
    #             toolate = []
    #             for kkk in sortedbuffer[nbs:]:
    #                 if np.sum(cng):
    #                     flag = 1
    #                     for j in toolate:
    #                         if kkk == j:
    #                             flag = 0
    #                             break
    #                     if flag:   #the pkt from the buffer is avelable
    #                        # rmax = state[0][(kkk[0]) * nbs+indmax]
    #                         indmax = np.argmax(state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs] + pAction[nff])
    #                         if cng[indmax]:
    #                             ti = t + np.tile(pAction[nff], (nbs, 1))
    #                             ti[indmax] += state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
    #                             inter = ti[e == 0]
    #                             max_inter = np.array(
    #                                 [np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
    #                             difm = ti - np.reshape(max_inter, (nbs, 1)) - 4
    #                             if difm[indmax][indmax] < 0:  # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
    #                                 flagw = 1
    #                                 for cp in range(nbs):
    #                                     if cng[cp] == 0:
    #                                         if difm[cp][cp] - difm[cp][indmax] + difm[indmax][indmax] < 4.00000001:
    #                                             #not good for cng
    #                                             flagw = 0
    #                                             break
    #                                 if flagw:
    #                                     #good for cng w power
    #                                     pAction[nff, indmax] = -difm[indmax][indmax]
    #                                     cng[indmax] = 0
    #                                     action[nff][indmax] = kkk[0]
    #                                     clk[nff][indmax] = kkk[1]
    #                                     choosen.append([kkk[0],kkk[1]])
    #                                     t[indmax, :] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
    #                                     toolate.append(kkk)
    #                                     break
    #                             else:
    #                                 #good for cng w no power
    #                                 cng[indmax] = 0
    #                                 action[nff][indmax] = kkk[0]
    #                                 clk[nff][indmax] = kkk[1]
    #                                 choosen.append([kkk[0], kkk[1]])
    #                                 t[indmax, :] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
    #                                 toolate.append(kkk)
    #                                 break
    #
    #             for jkj in choosen:
    #                 sortedbuffer.remove(jkj)
    #             choosen = sortedbuffer[0:nbs]
    #             for j in range(nbs):
    #                 if action[nff][j]==-1:
    #                     action[nff][j] = 12
    #                     pAction[nff][j] = -10
    #                     # choosen.append([12, 20])
    # #
    # #
    # #
    #         # xcopy = copy.deepcopy(choosen)
    #         # buff8x50.append(xcopy)
    #         # labels8x50.append(action)
    #         totalpower += np.sum(pAction)
    #         next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
    #         totalpass += pktPass
    #         totalloss += pktLoss
    #         if np.mod(run, 10)==1:
    #             print('run:', run)
    #             print('average TTL: ', (13 - drop))
    #             print('throughput: ' + str(totalpass / run))
    #             print('packet loss: ', (totalloss / run))
    #             print("average power: ", (totalpower / run))
    #     print('average TTL: ', (13 - drop))
    #     print('throughput: ' + str(totalpass / runs))
    #     graf_perfect_NN[1][drop] = (totalpass / runs)
    #     print('packet loss: ', (totalloss / runs))
    #     graf_perfect_NN[2][drop] = totalloss / runs
    #     print("average power: ", (totalpower / runs))
    #     graf_perfect_NN[3][drop] = (totalpower / runs)
    # np.save('graf_perfect_NN_100f.npy.npy', graf_perfect_NN)

  # ###############################3 pm  ##################
    graf_perfect_det = np.zeros((4, 9))
    graf_perfect_det[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    runs = 1000
    nbs = 8
    nf = 100
    for drop in range(9):
        xp = env('/home/gilmam/Desktop/study/scheduler/data8x50/gil/nUE50nRu4_seed1_len2000', nf=nf)
        totalpass = 0
        totalloss = 0
        totalpower = 0
        for run in range(runs):
            state = xp.state(run)
            sortedbuffer = sort(xp.B.buff)
            w = np.zeros((nbs * nf, nbs * nf))
            action = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
            pAction = np.array([[0., 0., 0., 0., 0., 0., 0., 0.] for i in range(nf)])
            clk = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])

            for fr in range(nf):
                choosen = sortedbuffer[0:nbs]
                w = np.zeros((nbs, nbs))

                ####################     perfect match      ######################
                ####################     Set the W wigths      ###################### w=[bs][msg]
                for i in range(nbs):
                    for j in range(nbs):
                        w[i][j] = state[0][(choosen[j][0] * nbs) + i]

                problem = LpProblem("PerfectMatch", LpMaximize)

                ###############################################################################   LP

                # factories

                x = LpVariable.dicts("x", list(range(nbs * nbs)), 0, 1, cat="Continuous")

                # goal constraint[0,1,2]
                for i in range(nbs):
                    problem += pulp.lpSum(x[j] for j in range(nbs * i, nbs * (i + 1))) == 1
                    problem += pulp.lpSum(x[j] for j in range(i, nbs * nbs, nbs)) == 1

                    # objective function
                problem += pulp.lpSum([x[nbs * i + j] * w[i][j] for i in range(nbs) for j in range(nbs)])

                #  print(problem)

                # solving
                problem.solve()

                for i1 in range(nbs):
                    for i2 in range(nbs):
                        if x[nbs * i1 + i2].varValue == 1:
                            # if i1<nbs:
                            #     freq_ind = 0
                            # else:
                            #     freq_ind = 1
                            # bs_ind = np.mod(i1, nbs)
                            action[fr][i1] = choosen[i2][0]
                            clk[fr][i1] = choosen[i2][1]
            #     # #############    pwoer calc      #############3
                cng = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                e = np.eye(nbs).astype(int)
                t = np.zeros((8,8))
                # for fr in range(1):
                ind_sort = np.argsort(clk[fr])
                t = np.array([state[0][(action[fr][i]) * nbs:action[fr][i] * nbs + nbs] for i in range(nbs)])
                for kk in range(nbs):
                    ti = t + np.tile(pAction[fr], (nbs, 1))
                    inter = ti[e == 0]
                    max_inter = np.array([np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
                    difm = ti - np.reshape(max_inter, (nbs, 1))-4
                    if difm[ind_sort[kk]][ind_sort[kk]]<0: # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
                        flagw=1
                        for cp in range(kk):
                            if cng[ind_sort[cp]]==0:
                                if difm[ind_sort[cp]][ind_sort[cp]] - difm[ind_sort[cp]][ind_sort[kk]] + difm[ind_sort[kk]][
                                    ind_sort[kk]] < 4.00000001:
                                    flagw = 0
                                    cng[ind_sort[kk]] = 1
                                    indi = choosen.index([action[fr][ind_sort[kk]], clk[fr][ind_sort[kk]]])
                                    del choosen[indi]
                                    action[fr][ind_sort[kk]] = -1
                                    t[ind_sort[kk], :] = 0

                                    break
                        if flagw:
                            pAction[fr,ind_sort[kk]] = -difm[ind_sort[kk]][ind_sort[kk]]
        #
        #         #############    cng from the buffer     #############3
                toolate = []
                for kkk in sortedbuffer[nbs:]:
                    if np.sum(cng):
                        flag = 1
                        for j in toolate:
                            if kkk == j:
                                flag = 0
                                break
                        if flag:   #the pkt from the buffer is avelable
                           # rmax = state[0][(kkk[0]) * nbs+indmax]
                            indmax = np.argmax(state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs] + pAction[fr])
                            if cng[indmax]:
                                ti = t + np.tile(pAction[fr], (nbs, 1))
                                ti[indmax] += state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
                                inter = ti[e == 0]
                                max_inter = np.array(
                                    [np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
                                difm = ti - np.reshape(max_inter, (nbs, 1)) - 4
                                if difm[indmax][indmax] < 0:  # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
                                    flagw = 1
                                    for cp in range(nbs):
                                        if cng[cp] == 0:
                                            if difm[cp][cp] - difm[cp][indmax] + difm[indmax][indmax] < 4.00000001:
                                                #not good for cng
                                                flagw = 0
                                                break
                                    if flagw:
                                        #good for cng w power
                                        pAction[fr, indmax] = -difm[indmax][indmax]
                                        cng[indmax] = 0
                                        action[fr][indmax] = kkk[0]
                                        clk[fr][indmax] = kkk[1]
                                        choosen.append([kkk[0],kkk[1]])
                                        t[indmax, :] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
                                        toolate.append(kkk)
                                        break
                                else:
                                    #good for cng w no power
                                    cng[indmax] = 0
                                    action[fr][indmax] = kkk[0]
                                    clk[fr][indmax] = kkk[1]
                                    choosen.append([kkk[0], kkk[1]])
                                    t[indmax, :] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
                                    toolate.append(kkk)
                                    break

                for jkj in choosen:
                    sortedbuffer.remove(jkj)
                choosen = sortedbuffer[0:nbs]
                for j in range(nbs):
                    if action[fr][j]==-1:
                        action[fr][j] = 12
                        pAction[fr][j] = -10
                        # choosen.append([12, 20])
            #############    pwoer calc      #############3
            # cng = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
            # e = np.eye(nbs).astype(int)
            # t = np.zeros((8, 8, nf))
            # for fr in range(nf):
            #     ind_sort = np.argsort(clk[fr])
            #     t[:, :, fr] = np.array(
            #         [state[0][(action[fr][i]) * nbs:action[fr][i] * nbs + nbs] for i in range(nbs)])
            #     for kk in range(nbs):
            #         ti = t[:, :, fr] + np.tile(pAction[fr], (nbs, 1))
            #         inter = ti[e == 0]
            #         max_inter = np.array(
            #             [np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
            #         difm = ti - np.reshape(max_inter, (nbs, 1)) - 4
            #         if difm[ind_sort[kk]][ind_sort[
            #             kk]] < 0:  # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
            #             flagw = 1
            #             for cp in range(kk):
            #                 if cng[fr, ind_sort[cp]] == 0:
            #                     if difm[ind_sort[cp]][ind_sort[cp]] - difm[ind_sort[cp]][ind_sort[kk]] + \
            #                             difm[ind_sort[kk]][
            #                                 ind_sort[kk]] < 4.00000001:
            #                         flagw = 0
            #                         cng[fr][ind_sort[kk]] = 1
            #                         action[fr][ind_sort[kk]] = -1
            #                         t[ind_sort[kk], :, fr] = 0
            #
            #                         break
            #             if flagw:
            #                 pAction[fr, ind_sort[kk]] = -difm[ind_sort[kk]][ind_sort[kk]]
            #
            # #############    cng from the buffer     #############3
            # toolate = []
            # for kkk in sortedbuffer[nbs * nf:]:
            #     if np.sum(cng):
            #         flag = 1
            #         for j in toolate:
            #             if kkk == j:
            #                 flag = 0
            #                 break
            #         if flag:  # the pkt from the buffer is avelable
            #             indmax = np.argmax(state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs] + pAction[fr])
            #             # rmax = state[0][(kkk[0]) * nbs+indmax]
            #             for fr in range(nf):
            #                 if cng[fr][indmax]:
            #                     ti = t[:, :, fr] + np.tile(pAction[fr], (nbs, 1))
            #                     ti[indmax] += state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
            #                     inter = ti[e == 0]
            #                     max_inter = np.array(
            #                         [np.max(inter[(i):(i + nbs - 1)]) for i in
            #                          range(0, nbs * (nbs - 1), (nbs - 1))])
            #                     difm = ti - np.reshape(max_inter, (nbs, 1)) - 4
            #                     if difm[indmax][
            #                         indmax] < 0:  # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
            #                         flagw = 1
            #                         for cp in range(nbs):
            #                             if cng[fr, cp] == 0:
            #                                 if difm[cp][cp] - difm[cp][indmax] + difm[indmax][indmax] < 4.00000001:
            #                                     # not good for cng
            #                                     flagw = 0
            #                                     break
            #                         if flagw:
            #                             # good for cng w power
            #                             pAction[fr, indmax] = -difm[indmax][indmax]
            #                             cng[fr][indmax] = 0
            #                             action[fr][indmax] = kkk[0]
            #                             clk[fr][indmax] = kkk[1]
            #                             choosen.append([kkk[0], kkk[1]])
            #                             t[indmax, :, fr] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
            #                             toolate.append(kkk)
            #                             break
            #                     else:
            #                         # good for cng w no power
            #                         cng[fr][indmax] = 0
            #                         action[fr][indmax] = kkk[0]
            #                         clk[fr][indmax] = kkk[1]
            #                         choosen.append([kkk[0], kkk[1]])
            #                         t[indmax, :, fr] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
            #                         toolate.append(kkk)
            #                         break
            #
            # for i in range(nf):
            #     for j in range(nbs):
            #         if action[i][j] == -1:
            #             action[i][j] = 12
            #             pAction[i][j] = -10


            totalpower += np.sum(pAction)
            next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
            totalpass += pktPass
            totalloss += pktLoss
            if np.mod(run, 10) == 1:
                print('run:', run)
                print('average TTL: ', (13 - drop))
                print('throughput: ' + str(totalpass / run))
                print('packet loss: ', (totalloss / run))
                print("average power: ", (totalpower / run))
        print('average TTL: ', (13 - drop))
        print('throughput: ' + str(totalpass / runs))
        graf_perfect_det[1][drop] = (totalpass / runs)
        print('packet loss: ', (totalloss / runs))
        graf_perfect_det[2][drop] = totalloss / runs
        print("average power: ", (totalpower / runs))
        graf_perfect_det[3][drop] = (totalpower / runs)
    np.save('graf_PM_100f_negP.npy', graf_perfect_det)
            #########################3    random match     ######################
    # graf_rand = np.zeros((4, 9))
    # graf_rand[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    # for drop in range(9):
    #     nbs = 8
    #     nf = 2
    #     labels8x50 = []
    #     buff8x50 = []
    #     state8x50 = []
    #     xp = env('/home/gilmam/Desktop/study/scheduler/data8x50/gil/nUE50nRu4_seed1_len2000', nf=nf)
    #     totalpass = 0
    #     totalloss = 0
    #     totalpower = 0
    #     for run in range(10000):
    #         state = xp.state(run)
    #         sortedbuffer = sort(xp.B.buff)
    #         choosen = sortedbuffer[0:nbs * nf]
    #         state8x50.append(state)
    #         xcopy = copy.deepcopy(choosen)
    #         buff8x50.append(xcopy)
    #         w = np.zeros((nbs * nf, nbs * nf))
    #         xx = np.zeros((nbs * nf, nbs * nf))
    #         action = np.array([[-1, -1, -1, -1, -1, -1, -1, -1] for i in range(nf)])
    #         pAction = np.array([[0., 0., 0., 0., 0., 0., 0., 0.] for i in range(nf)])
    #         clk = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
    #
    #         #######################   rand    ######################
    #         # inputl = [state[0][choosen[i][0] * 8 + k] for i in range(16) for k in range(8)]
    #         # inputnp = np.asarray(inputl)
    #         # inputd = torch.from_numpy(inputnp)
    #         # data = inputd.to(device, dtype=torch.float)
    #         # with torch.no_grad():
    #         #     out = model(data)
    #         # pred = torch.max(out, 2)[1]
    #         # no_match = []
    #         # for ru in range(16):
    #         #     ind1 = pred.cpu().numpy()[0][ru] // 8
    #         #     ind2 = np.mod(pred.cpu().numpy()[0][ru], 8)
    #         #     if action[ind1][ind2] == -1:  # if the slot is aveilable
    #         #         action[ind1][ind2] = choosen[ru][0]
    #         #         clk[ind1][ind2] = choosen[ru][1]
    #         #     else:
    #         #         if clk[ind1][ind2] > choosen[ru][1]:
    #         #             no_match.append([action[ind1][ind2], clk[ind1][ind2]])
    #         #             action[ind1][ind2] = choosen[ru][0]
    #         #             clk[ind1][ind2] = choosen[ru][1]
    #         #         else:
    #         #             no_match.append(choosen[ru])
    #         for ru in range(2):
    #             for ru2 in range(8):
    #                 action[ru][ru2] = choosen[ru*8+ru2][0]
    #                 clk[ru][ru2] = choosen[ru*8+ru2][1]
    #
    #
    #         # #############    pwoer calc      #############3
    #         cng = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
    #         # e = np.eye(nbs).astype(int)
    #         # t = np.zeros((8, 8, 2))
    #         # for fr in range(nf):
    #         #     ind_sort = np.argsort(clk[fr])
    #         #     t[:, :, fr] = np.array(
    #         #         [state[0][(action[fr][i]) * nbs:action[fr][i] * nbs + nbs] for i in range(nbs)])
    #         #     for kk in range(nbs):
    #         #         ti = t[:, :, fr] + np.tile(pAction[fr], (nbs, 1))
    #         #         inter = ti[e == 0]
    #         #         max_inter = np.array(
    #         #             [np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
    #         #         difm = ti - np.reshape(max_inter, (nbs, 1)) - 4
    #         #         if difm[ind_sort[kk]][ind_sort[
    #         #             kk]] < 0:  # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
    #         #             flagw = 1
    #         #             for cp in range(kk):
    #         #                 if cng[fr, ind_sort[cp]] == 0:
    #         #                     if difm[ind_sort[cp]][ind_sort[cp]] - difm[ind_sort[cp]][ind_sort[kk]] + \
    #         #                             difm[ind_sort[kk]][
    #         #                                 ind_sort[kk]] < 4.00000001:
    #         #                         flagw = 0
    #         #                         cng[fr][ind_sort[kk]] = 1
    #         #                         indi = choosen.index([action[fr][ind_sort[kk]], clk[fr][ind_sort[kk]]])
    #         #                         del choosen[indi]
    #         #                         action[fr][ind_sort[kk]] = -1
    #         #                         t[ind_sort[kk], :, fr] = 0
    #         #
    #         #                         break
    #         #             if flagw:
    #         #                 pAction[fr, ind_sort[kk]] = -difm[ind_sort[kk]][ind_sort[kk]]
    #
    #         # #############    cng from the buffer     #############3
    #         toolate = []
    #         for kkk in sortedbuffer[nbs * nf:]:
    #             if np.sum(cng):
    #                 flag = 1
    #                 for j in toolate:
    #                     if kkk == j:
    #                         flag = 0
    #                         break
    #                 if flag:  # the pkt from the buffer is avelable
    #                     indmax = np.argmax(state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs] + pAction[fr])
    #                     # rmax = state[0][(kkk[0]) * nbs+indmax]
    #                     for fr in range(nf):
    #                         if cng[fr][indmax]:
    #                             ti = t[:, :, fr] + np.tile(pAction[fr], (nbs, 1))
    #                             ti[indmax] += state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
    #                             inter = ti[e == 0]
    #                             max_inter = np.array(
    #                                 [np.max(inter[(i):(i + nbs - 1)]) for i in
    #                                  range(0, nbs * (nbs - 1), (nbs - 1))])
    #                             difm = ti - np.reshape(max_inter, (nbs, 1)) - 4
    #                             if difm[indmax][
    #                                 indmax] < 0:  # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
    #                                 flagw = 1
    #                                 for cp in range(nbs):
    #                                     if cng[fr, cp] == 0:
    #                                         if difm[cp][cp] - difm[cp][indmax] + difm[indmax][
    #                                             indmax] < 4.00000001:
    #                                             # not good for cng
    #                                             flagw = 0
    #                                             break
    #                                 if flagw:
    #                                     # good for cng w power
    #                                     pAction[fr, indmax] = -difm[indmax][indmax]
    #                                     cng[fr][indmax] = 0
    #                                     action[fr][indmax] = kkk[0]
    #                                     clk[fr][indmax] = kkk[1]
    #                                     choosen.append([kkk[0], kkk[1]])
    #                                     t[indmax, :, fr] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
    #                                     toolate.append(kkk)
    #                                     break
    #                             else:
    #                                 # good for cng w no power
    #                                 cng[fr][indmax] = 0
    #                                 action[fr][indmax] = kkk[0]
    #                                 clk[fr][indmax] = kkk[1]
    #                                 choosen.append([kkk[0], kkk[1]])
    #                                 t[indmax, :, fr] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
    #                                 toolate.append(kkk)
    #                                 break
    #         #
    #         for i in range(nf):
    #             for j in range(nbs):
    #                 if action[i][j] == -1:
    #                     action[i][j] = 12
    #                     choosen.append([12, 20])
    #
    #         xcopy = copy.deepcopy(choosen)
    #         buff8x50.append(xcopy)
    #         labels8x50.append(action)
    #         totalpower += np.sum(pAction)
    #         next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
    #         totalpass += pktPass
    #         totalloss += pktLoss
    #         if np.mod(run, 1000) == 1:
    #             print('run:', run)
    #             print('average TTL: ', (13 - drop))
    #             print('throughput: ' + str(totalpass / run))
    # +             print('packet loss: ', (totalloss / run))
    #             print("average power: ", (totalpower / run))
    #     print('average TTL: ', (13 - drop))
    #     print('throughput: ' + str(totalpass / 10000))
    #     graf_rand[1][drop] = (totalpass / 10000)
    #     print('packet loss: ', (totalloss / 10000))
    #     graf_rand[2][drop] = totalloss / 10000
    #     print("average power: ", (totalpower / 10000))
    #     graf_rand[3][drop] = (totalpower / 10000)
    #     # print('set_index:', set_index)
    #     # os.chdir("/home/gilmam/Desktop/study/scheduler/data8x50/gil/set_data/100k")
    #     # np.save('perfect_state_8x2_10k_set{}.npy'.format(set_index), state8x50)
    #     # np.save('perfect_buffer_8x2_10k_set{}.npy'.format(set_index), buff8x50)
    #     # np.save('perfect_labels_8x2_10k_set{}.npy'.format(set_index), labels8x50)
    # np.save('graf_rand_n_power_w_s.npy', graf_rand)
    # gil = 10

    # graf_perfect_NN = np.load('graf_perfect_NN.npy')
    # graf_perfect_det = np.load('graf_perfect_det.npy')
    # graf_rand = np.load('graf_rand.npy')
    # graf_rand_wn_power = np.load('graf_rand_wn_power.npy')
    # graf_rand_w_power_n_s = np.load('graf_rand_w_power_n_s.npy')
    # graf_rand_w_power_w_s = np.load('graf_rand_w_power_w_s.npy')
    # graf_perfect = np.load('graf_NN_1f.npy')


    plt.figure(1)
    # Exhaustive_search, = plt.plot(graf[0], graf[1], 'r', label='Exhaustive search')
    Perfect_match_NN_100f, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[1], 'c', label='Perfect match NN 100F')
    # Perfect_match_NN, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[1], 'b', label='Perfect match NN')
    Perfect_match_det, = plt.plot(graf_perfect_det[0], graf_perfect_det[1], 'r', label='Perfect match deterministic')
    # # random, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='all wrong')
    # random_npns, = plt.plot(graf_rand_wn_power[0], graf_rand_wn_power[1], 'y', label='random match no Power and swich')
    # random_w_power_n_s, = plt.plot(graf_rand_w_power_n_s[0], graf_rand_w_power_n_s[1], 'k',
    #                                label='random match w Power and swich')
    # random_scheduler, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='Random')
    plt.xlabel('average TTL')
    plt.ylabel('average throughput')
    plt.title('throughput vs. average TTL')
    plt.legend()
    plt.grid()
    plt.savefig('/home/gilmam/Desktop/study/scheduler/data8x50/gil/throughput_8x2_NN_pm_1k_100f.png')
    plt.figure(2)
    Perfect_match_NN_100f, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[2], 'c', label='Perfect match NN 100F')
    # Perfect_match_NN, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[1], 'b', label='Perfect match NN')
    Perfect_match_det, = plt.plot(graf_perfect_det[0], graf_perfect_det[2], 'r', label='Perfect match deterministic')
    # Perfect_match_det, = plt.plot(graf_perfect_det[0], graf_perfect_det[2], 'r', label='Perfect match deterministic')
    # # random, = plt.plot(graf_rand[0], graf_rand[2], 'g', label='all wrong')
    # random_npns, = plt.plot(graf_rand_wn_power[0], graf_rand_wn_power[2], 'y', label='random match no Power and swich')
    # random_w_power_n_s, = plt.plot(graf_rand_w_power_n_s[0], graf_rand_w_power_n_s[2], 'k',
    #                                label='random match w Power and swich')
    plt.xlabel('average TTL')
    plt.ylabel('average packet loss')
    plt.title('packet loss vs. average TTL')
    plt.legend()
    plt.grid()
    plt.savefig('/home/gilmam/Desktop/study/scheduler/data8x50/gil/packet_loss_8x2_NN_pm_1k_100f.png')
    plt.figure(3)
    Perfect_match_NN_100f, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[3], 'c', label='Perfect match NN 100F')
    # Perfect_match_NN, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[1], 'b', label='Perfect match NN')
    Perfect_match_det, = plt.plot(graf_perfect_det[0], graf_perfect_det[3], 'r', label='Perfect match deterministic')
    # # random, = plt.plot(graf_rand[0], graf_rand[3], 'g', label='all wrong')
    # random_npns, = plt.plot(graf_rand_wn_power[0], graf_rand_wn_power[3], 'y', label='random match no Power and swich')
    # random_w_power_n_s, = plt.plot(graf_rand_w_power_n_s[0], graf_rand_w_power_n_s[3], 'k',
    #                                label='random match w Power and swich')
    # random_w_power_w_s, = plt.plot(graf_rand_w_power_w_s[0], graf_rand_w_power_w_s[3], 'm', label='random match_w_p_w_s')
    plt.xlabel('average TTL')
    plt.ylabel('average power')
    plt.title('power vs. average TTL')
    plt.legend()
    plt.grid()
    plt.savefig('/home/gilmam/Desktop/study/scheduler/data8x50/gil/power_8x2_NN_pm_1k_100f.png')
    plt.show
    gil = 10