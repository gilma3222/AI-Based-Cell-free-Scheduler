"""
Created on Tue Nov 12 11:41:08 2019
@author: shasha
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pulp import *
###############################################################################

class buffer:
    totalLoss = 0
    def __init__(self,size=20,nUE=10,rangeTTL=[8, 10, 12, 14, 16, 18]):
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
    def __init__(self, fileName,trh=4):
        self.d        = pickle.load(open(fileName+'.pckl','rb'))
        self.nUE      = self.d['nUE']
        self.nRecords = self.d['nRecords']
        self.B        = buffer(nUE=self.nUE)
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
            snr = P-inter
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
    x = env('/home/gilmam/Desktop/study/scheduler/nUE10_seed1_len2000')
    graf_perfect = np.zeros((4,9))
    graf_perfect[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    for drop in range(9):

        totalpass = 0
        totalloss = 0
        totalpower = 0
        for run in range(2000):
            state = x.state(run)
            sortedbuffer = sort(x.B.buff)
            choosen = sortedbuffer[0:4]
            w = np.zeros((4, 4))
            xx = np.zeros((4, 4))
            action = np.array([[0, 0], [0, 0]])
            pAction = np.array([[0., 0.], [0., 0.]])

            ####################     perfect match      ######################

            for i in range(4):
                for j in range(4):
                    if np.mod(i, 2) == 0:
                        w[i][j] = state[0][choosen[j][0]*2] - state[0][choosen[j][0]*2+1]
                    else:
                        w[i][j] = state[0][choosen[j][0]*2+1] - state[0][choosen[j][0]*2]

            problem = LpProblem("PerfectMatch", LpMaximize)

            ###############################################################################   LP

            # factories

            xxx = LpVariable.dict("xxx", ((i, j) for i in range(4) for j in range(4)), 0, 1, cat="Continuous")

            # goal constraint
            c1 = np.sum(xxx, 0) == np.array([1, 1, 1, 1])  # production constraints
            c2 = np.sum(xxx, 1) == np.array([1, 1, 1, 1])
            c3 = xxx[0][0] <= 1
            c4 = xxx[0][1] <= 1
            c5 = xxx[0][2] <= 1
            c6 = xxx[0][3] <= 1
            c7 = xxx[1][0] <= 1
            c8 = xxx[1][1] <= 1
            c9 = xxx[1][2] <= 1
            c10 = xxx[1][3] <= 1
            c11 = xxx[2][0] <= 1
            c12 = xxx[2][1] <= 1
            c13 = xxx[2][2] <= 1
            c14 = xxx[2][3] <= 1
            c15 = xxx[3][0] <= 1
            c16 = xxx[3][1] <= 1
            c17 = xxx[3][2] <= 1
            c18 = xxx[3][3] <= 1
            c19 = xxx[0][0] >= 0
            c20 = xxx[0][1] >= 0
            c21 = xxx[0][2] >= 0
            c22 = xxx[0][3] >= 0
            c23 = xxx[1][0] >= 0
            c24 = xxx[1][1] >= 0
            c25 = xxx[1][2] >= 0
            c26 = xxx[1][3] >= 0
            c27 = xxx[2][0] >= 0
            c28 = xxx[2][1] >= 0
            c29 = xxx[2][2] >= 0
            c30 = xxx[2][3] >= 0
            c31 = xxx[3][0] >= 0
            c32 = xxx[3][1] >= 0
            c33 = xxx[3][2] >= 0
            c34 = xxx[3][3] >= 0

            problem += c1
            problem += c2
            problem += c3
            problem += c4
            problem += c5
            problem += c6
            problem += c7
            problem += c8
            problem += c9
            problem += c10
            problem += c11
            problem += c12
            problem += c13
            problem += c14
            problem += c15
            problem += c16
            problem += c17
            problem += c18
            problem += c19
            problem += c20
            problem += c21
            problem += c22
            problem += c23
            problem += c24
            problem += c25
            problem += c26
            problem += c27
            problem += c28
            problem += c29
            problem += c30
            problem += c31
            problem += c32
            problem += c33
            problem += c34

            # objective function
            problem += np.sum(np.multiply(xxx, w))

            print(problem)

            # solving
            problem.solve()

            # for i in range(3):
            #     print(f"Factory {i}: {factory_days[i].varValue}")





            max_reward = -100
            for i1 in range(4):
                for i2 in range(4):
                    for i3 in range(4):
                        for i4 in range(4):
                            if i1 != i2 and i1 != i3 and i1 != i4 and i3 != i2 and i4 != i2 and i3 != i4:
                                xx = np.zeros((4, 4))
                                xx[0][i1] = 1
                                xx[1][i2] = 1
                                xx[2][i3] = 1
                                xx[3][i4] = 1
                                reward = np.sum(np.multiply(xx, w))
                                if reward > max_reward:
                                    pAction = np.array([[0., 0.], [0., 0.]])
                                    max_reward = reward
                                    action = np.array([[choosen[i1][0], choosen[i2][0]], [choosen[i3][0], choosen[i4][0]]])

                                    if choosen[i1][1]>choosen[i2][1]:
                                        if state[0][action[0][1]*2+1]-state[0][action[0][1]*2]<4:
                                            pAction[0][1] = -(state[0][action[0][1]*2+1]-state[0][action[0][1]*2]-4)
                                        else:
                                            if state[0][action[0][0]*2]-state[0][action[0][0]*2+1]<4:
                                                pAction[0][0] = -(state[0][action[0][0]*2]-state[0][action[0][0]*2+1]-4)
                                    else:
                                        if state[0][action[0][0]*2]-state[0][action[0][0]*2+1]<4:
                                            pAction[0][0] = -(state[0][action[0][0]*2]-state[0][action[0][0]*2+1]-4)
                                        else:
                                            if state[0][action[0][1]*2+1]-state[0][action[0][1]*2]<4:
                                                pAction[0][1] = -(state[0][action[0][1]*2+1]-state[0][action[0][1]*2]-4)

                                    if choosen[i3][1]>choosen[i4][1]:
                                        if state[0][action[1][1]*2+1]-state[0][action[1][1]*2]<4:
                                            pAction[1][1] = -(state[0][action[1][1]*2+1]-state[0][action[1][1]*2]-4)
                                        else:
                                            if state[0][action[1][0]*2]-state[0][action[1][0]*2+1]<4:
                                                pAction[1][0] = -(state[0][action[1][0]*2]-state[0][action[1][0]*2+1]-4)
                                    else:
                                        if state[0][action[1][0]*2]-state[0][action[1][0]*2+1]<4:
                                            pAction[1][0] = -(state[0][action[1][0]*2]-state[0][action[1][0]*2+1]-4)
                                        else:
                                            if state[0][action[1][1]*2+1]-state[0][action[1][1]*2]<4:
                                                pAction[1][1] = -(state[0][action[1][1]*2+1]-state[0][action[1][1]*2]-4)


            totalpower += pAction[0][0] + pAction[0][1] + pAction[1][0] + pAction[1][1]
            next_state, pktPass, pktLoss = x.step(action, pAction, drop)
            totalpass += pktPass
            totalloss += pktLoss
          #   print(pktPass, pktLoss)

        print('avarge TTL: ', (13-drop))
        print('throughput: '+str(totalpass/2000))
        graf_perfect[1][drop] = totalpass / 2000
        print('packet loss: ', (totalloss/2000))
        graf_perfect[2][drop] = totalloss / 2000
        print("avarge power: ", (totalpower/2000))
        graf_perfect[3][drop] = totalpower / 2000

    # plt.figure(1)
    # plt.plot(graf[0], graf[1])
    # plt.xlabel('avareg TTL')
    # plt.ylabel('avarge throughput')
    # plt.title('throughput vs. avarge TTL')
    # plt.grid()
    #
    # plt.figure(2)
    # plt.plot(graf[0], graf[2])
    # plt.xlabel('avareg TTL')
    # plt.ylabel('avarge packet loss')
    # plt.title('packet loss vs. avarge TTL')
    # plt.grid()
    #
    # plt.figure(3)
    # plt.plot(graf[0], graf[3])
    # plt.xlabel('avareg TTL')
    # plt.ylabel('avarge power')
    # plt.title('power vs. avarge TTL')
    # plt.grid()
    #
    # plt.show
    gil=10   # print('trhoput:' + (totalpass / 2000) + 'packet loss:' + (totalloss / 2000) + 'avarge power:' + (totalpower / 2000))