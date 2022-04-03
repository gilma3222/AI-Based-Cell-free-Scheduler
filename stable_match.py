"""
Created on Tue Nov 12 11:41:08 2019
@author: shasha
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pulp import *



###################################################################################3
class buffer:
    totalLoss = 0
    def __init__(self,size,nUE=10,rangeTTL=[8, 10, 12, 14, 16, 18]):
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
    def __init__(self, fileName, nf, trh=4):
        self.d        = pickle.load(open(fileName+'.pckl','rb'))
        self.nUE      = self.d['nUE']
        self.nRecords = self.d['nRecords']
        self.B        = buffer(nUE=self.nUE,size=nf*10)
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

    #########################3    stable match     ######################
    nf=150

    graf_stable = np.zeros((4,9))
    graf_stable[0] = [13,12,11,10,9,8,7,6,5]
    for drop in range(9):
        xp = env('/home/gilmam/Desktop/study/scheduler/nUE10_seed1_len2000', nf=nf)
        totalpass = 0
        totalloss = 0
        totalpower = 0
        for run in range(2000):
            state = xp.state(run)
            sortedbuffer = sort(xp.B.buff)
            choosen = sortedbuffer[0:2*nf]
            xxue = (-1)*np.ones((1, 2*nf), dtype=int)
            loc = np.zeros((2*nf, 2*nf), dtype=int)
            ue = np.zeros((2*nf, 2*nf), dtype=int)
            action = np.array([[0, 0] for i in range(nf)])
            pAction = np.array([[0., 0.] for i in range(nf)])

            ####################     data preperetion      ######################

            for i in range(2*nf):
                temp_dvec = np.array([[ii, 0] for ii in range(nf*2)])
                for j in range(2*nf):
                    if np.mod(i, 2) == 0:
                        temp_dvec[j][1] = state[0][choosen[j][0]*2] - state[0][choosen[j][0]*2+1]
                    else:
                        temp_dvec[j][1] = state[0][choosen[j][0]*2+1] - state[0][choosen[j][0]*2]
                #print(temp_dvec)
                sortedd = sort(temp_dvec)
                #print(sortedd)
                sorteddd = [sortedd[iii][0] for iii in range(2*nf)]
                sorteddd.reverse()
                loc[i] = sorteddd

            for uei in range(2*nf):
                bs1 = [i for i in range(0, 2*nf, 2)]
                bs2 = [i for i in range(1, 2*nf, 2)]
                if state[0][choosen[uei][0]*2] - state[0][choosen[uei][0]*2+1] > 0:
                    ue[uei] = bs1 + bs2
                else:
                    ue[uei] = bs2 + bs1


            ####################     loc ask from ue    ######################

            for i in range(nf*2): # i is col (priorety)
                for j in range(nf*2):  # j is row (loc's)
                    if np.sum(xxue[0] == j) == 0:
                        if xxue[0][loc[j][i]] == -1:
                            xxue[0][loc[j][i]] = j
                        else:
                            if int(np.where(ue[loc[j][i]] == xxue[0][loc[j][i]])[0]) > int(np.where(ue[loc[j][i]] == j)[0]):
                                xxue[0][loc[j][i]] = j
                if np.sum(xxue[0] == -1) == 0:
                    break

            ####################     set the action matrix    ######################
            clk = np.array([[0, 0] for i in range(nf)])
            for i in range(2*nf):
                odd1 = np.mod(xxue[0][i], 2)
                freqi = int((xxue[0][i] - odd1)/2)
                action[freqi][odd1] = choosen[i][0]
                clk[freqi][odd1] = choosen[i][1]

            ####################     set the power matrix and change the pkt which not sent    ######################
            toolate = []

            for i in range(nf):
                A_i = state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] + state[0][action[i][1] * 2 + 1] - \
                      state[0][action[i][1] * 2]
                if A_i < 8:
                    if clk[i][1] < clk[i][0]:
                        if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] > 0:
                            action[i][0] = -1
                        else:
                            action[i][0] = action[i][1]
                            action[i][1] = -1
                    else:
                        if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] > 0:
                            action[i][1] = -1
                        else:
                            action[i][1] = action[i][0]
                            action[i][0] = -1
                else:
                    if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
                        pAction[i][0] = -(state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
                    if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
                        pAction[i][1] = -(state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)

                if action[i][0] == -1:
                    for kkk in sortedbuffer[2 * nf:]:
                        flag = 1
                        for j in toolate:
                            if kkk == j:
                                flag = 0
                                break
                        if flag:
                            if state[0][kkk[0] * 2] - state[0][kkk[0] * 2 + 1] + state[0][action[i][1] * 2 + 1] - \
                                    state[0][action[i][1] * 2] > 8:
                                action[i][0] = kkk[0]
                                toolate.append(kkk)
                                if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
                                    pAction[i][0] = -(state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
                                if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
                                    pAction[i][1] = -(state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)
                                break
                if action[i][1] == -1:
                    for kkk in sortedbuffer[2 * nf:]:
                        flag = 1
                        for j in toolate:
                            if kkk == j:
                                flag = 0
                                break
                        if flag:
                            if state[0][kkk[0] * 2 + 1] - state[0][kkk[0] * 2] + state[0][action[i][0] * 2] - state[0][
                                action[i][0] * 2 + 1] > 8:
                                action[i][1] = kkk[0]
                                toolate.append(kkk)
                                if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
                                    pAction[i][0] = -(state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
                                if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
                                    pAction[i][1] = -(state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)
                                break



            totalpower += np.sum(pAction)
            next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
            totalpass += pktPass
            totalloss += pktLoss
            print(run)

        print('average TTL: ', (13-drop))
        print('throughput: '+str(totalpass/2000))
        graf_stable[1][drop] = (totalpass / 2000)
        print('packet loss: ', (totalloss/2000))
        graf_stable[2][drop] = totalloss / 2000
        print("average power: ", (totalpower/2000))
        graf_stable[3][drop] = (totalpower / 2000)

    #
    #
    plt.figure(1)
    #Exhaustive_search, = plt.plot(graf[0], graf[1], 'r', label='Exhaustive search')
    Stable_match, = plt.plot(graf_stable[0], graf_stable[1], 'b', label='Stable match')
    #random_scheduler, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='Random')
    plt.xlabel('average TTL')
    plt.ylabel('average throughput')
    plt.title('throughput vs. average TTL')
    plt.legend()
    plt.grid()

    plt.figure(2)
    #Exhaustive_search, = plt.plot(graf[0], graf[2], 'r', label='Exhaustive search')
    Stable_match, = plt.plot(graf_stable[0], graf_stable[2], 'b', label='Stable match')
    #random_scheduler, = plt.plot(graf_rand[0], graf_rand[2], 'g', label='Random')
    plt.xlabel('average TTL')
    plt.ylabel('average packet loss')
    plt.title('packet loss vs. average TTL')
    plt.legend()
    plt.grid()

    plt.figure(3)
    #Exhaustive_search, = plt.plot(graf[0], graf[3], 'r', label='Exhaustive search')
    Stable_match, = plt.plot(graf_stable[0], graf_stable[3], 'b', label='Stable match')
    #random_scheduler, = plt.plot(graf_rand[0], graf_rand[3], 'g', label='Random')
    plt.xlabel('average TTL')
    plt.ylabel('average power')
    plt.title('power vs. average TTL')
    plt.legend()
    plt.grid()
    #
    # plt.show
    gil=10   # print('trhoput:' + (totalpass / 2000) + 'packet loss:' + (totalloss / 2000) + 'avarge power:' + (totalpower / 2000))