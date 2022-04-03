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
    # x = env('/home/gilmam/Desktop/study/scheduler/nUE10_seed1_len2000')
    # graf = np.zeros((4, 9))
    # graf[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    # for drop in range(9):
    #     totalpass = 0
    #     totalloss = 0
    #     totalpower = 0
    #     for run in range(2000):
    #         state = x.state(run)
    #         sortedbuffer = sort(x.B.buff)
    #         choosen = sortedbuffer[0:6]
    #
    #     ####################     Exhaustive search    ######################
    #
    #         max_reward=0
    #         for i1 in range(6):
    #             for i2 in range(6):
    #                 for i3 in range(6):
    #                     for i4 in range(6):
    #                         for i5 in range(6):
    #                             for i6 in range(6):
    #
    #                                         if i1!=i2 and i1!=i3 and i1!=i4 and i1!=i5 and i1!=i6 and i2!=i3 and i2!=i4 and i2!=i5 and i2!=i6 and i3!=i4 and i3!=i5 and i3!=i6 and i4!=i5 and i4!=i6 and i5!=i6:
    #                                             reward = 0
    #                                             powerop=np.array([[0.,0.],[0.,0.],[0., 0.]])
    #                                             aop=np.array([[choosen[i1][0], choosen[i2][0]],[choosen[i3][0],choosen[i4][0]],[choosen[i5][0],choosen[i6][0]]])
    #                                             A_1 = state[0][choosen[i1][0]*2] - state[0][choosen[i1][0]*2+1] + state[0][choosen[i2][0]*2+1] - state[0][choosen[i2][0]*2]
    #                                             A_2 = state[0][choosen[i3][0]*2] - state[0][choosen[i3][0]*2+1] + state[0][choosen[i4][0]*2+1] - state[0][choosen[i4][0]*2]
    #                                             A_3 = state[0][choosen[i5][0] * 2] - state[0][choosen[i5][0] * 2 + 1] + state[0][choosen[i6][0] * 2 + 1] - state[0][choosen[i6][0] * 2]
    #                                             if A_1<8:
    #                                                 if choosen[i1][1]<choosen[i2][1]:
    #                                                     if state[0][choosen[i1][0]*2]-state[0][choosen[i1][0]*2+1]>0:
    #                                                         aop[0][1] = -1
    #                                                     else:
    #                                                         aop[0][1] = aop[0][0]
    #                                                         aop[0][0] = -1
    #                                                 else:
    #                                                     if state[0][choosen[i2][0] * 2+1] - state[0][choosen[i2][0] * 2] > 0:
    #                                                         aop[0][0] = -1
    #                                                     else:
    #                                                         aop[0][0] = aop[0][1]
    #                                                         aop[0][1] = -1
    #                                             else:
    #                                                 if state[0][choosen[i1][0]*2] - state[0][choosen[i1][0]*2+1]<4:
    #                                                     powerop[0][0] = -(state[0][choosen[i1][0]*2] - state[0][choosen[i1][0]*2+1]-4)
    #                                                 if state[0][choosen[i2][0]*2+1] - state[0][choosen[i2][0]*2]<4:
    #                                                     powerop[0][1] = -(state[0][choosen[i2][0]*2+1] - state[0][choosen[i2][0]*2]-4)
    #
    #
    #                                             if A_2<8:
    #                                                 if choosen[i3][1] < choosen[i4][1]:
    #                                                     if state[0][choosen[i3][0] * 2] - state[0][choosen[i3][0] * 2 + 1] > 0:
    #                                                         aop[1][1] = -1
    #                                                     else:
    #                                                         aop[1][1] = aop[1][0]
    #                                                         aop[1][0] = -1
    #                                                 else:
    #                                                     if state[0][choosen[i4][0] * 2 + 1] - state[0][choosen[i4][0] * 2] > 0:
    #                                                         aop[1][0] = -1
    #                                                     else:
    #                                                         aop[1][0] = aop[1][1]
    #                                                         aop[1][1] = -1
    #                                             else:
    #                                                 if state[0][choosen[i3][0] * 2] - state[0][choosen[i3][0] * 2 + 1] < 4:
    #                                                     powerop[1][0] = -(state[0][choosen[i3][0] * 2] - state[0][choosen[i3][0] * 2 + 1] - 4)
    #                                                 if state[0][choosen[i4][0] * 2 + 1] - state[0][choosen[i4][0] * 2] < 4:
    #                                                     powerop[1][1] = -(state[0][choosen[i4][0] * 2 + 1] - state[0][choosen[i4][0] * 2] - 4)
    #
    #                                             if A_3<8:
    #                                                 if choosen[i5][1] < choosen[i6][1]:
    #                                                     if state[0][choosen[i5][0] * 2] - state[0][choosen[i5][0] * 2 + 1] > 0:
    #                                                         aop[2][1] = -1
    #                                                     else:
    #                                                         aop[2][1] = aop[2][0]
    #                                                         aop[2][0] = -1
    #                                                 else:
    #                                                     if state[0][choosen[i6][0] * 2 + 1] - state[0][choosen[i6][0] * 2] > 0:
    #                                                         aop[2][0] = -1
    #                                                     else:
    #                                                         aop[2][0] = aop[2][1]
    #                                                         aop[2][1] = -1
    #                                             else:
    #                                                 if state[0][choosen[i5][0] * 2] - state[0][choosen[i5][0] * 2 + 1] < 4:
    #                                                     powerop[2][0] = -(state[0][choosen[i5][0] * 2] - state[0][choosen[i5][0] * 2 + 1] - 4)
    #                                                 if state[0][choosen[i6][0] * 2 + 1] - state[0][choosen[i6][0] * 2] < 4:
    #                                                     powerop[2][1] = -(state[0][choosen[i6][0] * 2 + 1] - state[0][choosen[i6][0] * 2] - 4)
    #                                             reward = 50*(1*(aop[0][0]>=0) + 1*(aop[0][1]>=0) + 1*(aop[1][0]>=0) + 1*(aop[1][1]>=0) + 1*(aop[2][0]>=0) + 1*(aop[2][1]>=0))-(powerop[0][0] + powerop[0][1] + powerop[1][0] + powerop[1][1] + powerop[2][0] + powerop[2][1])
    #                                             if reward>max_reward:
    #                                                 max_reward = reward
    #                                                 action = aop
    #                                                 pAction = powerop
    #
    #         if action[0][0] == -1:
    #             for kkk in sortedbuffer[6:]:
    #                 if state[0][kkk[0]*2]-state[0][kkk[0]*2+1]+state[0][action[0][1]*2+1]-state[0][action[0][1]*2]>8:
    #                     action[0][0] = kkk[0]
    #                     if state[0][action[0][0]*2]-state[0][action[0][0]*2+1]<4:
    #                         pAction[0][0] = -(state[0][action[0][0]*2]-state[0][action[0][0]*2+1]-4)
    #                     if state[0][action[0][1]*2+1]-state[0][action[0][1]*2]<4:
    #                         pAction[0][1] = -(state[0][action[0][1]*2+1]-state[0][action[0][1]*2]-4)
    #                     break
    #         if action[0][1] == -1:
    #             for kkk in sortedbuffer[6:]:
    #                 if state[0][kkk[0] * 2+1] - state[0][kkk[0] * 2] + state[0][action[0][0] * 2] - state[0][action[0][0] * 2+1] > 8:
    #                     action[0][1] = kkk[0]
    #                     if state[0][action[0][1] * 2+1] - state[0][action[0][1] * 2] < 4:
    #                         pAction[0][1] = -(state[0][action[0][1] * 2+1] - state[0][action[0][1] * 2] - 4)
    #                     if state[0][action[0][0] * 2] - state[0][action[0][0] * 2+1] < 4:
    #                         pAction[0][0] = -(state[0][action[0][0] * 2] - state[0][action[0][0] * 2+1] - 4)
    #                     break
    #         if action[1][0] == -1:
    #             for kkk in sortedbuffer[6:]:
    #                 if state[0][kkk[0] * 2] - state[0][kkk[0] * 2 + 1] + state[0][action[1][1] * 2 + 1] - state[0][action[1][1] * 2] > 8:
    #                     action[1][0] = kkk[0]
    #                     if state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] < 4:
    #                         pAction[1][0] = -(state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] - 4)
    #                     if state[0][action[1][1] * 2 + 1] - state[0][action[1][1] * 2] < 4:
    #                         pAction[1][1] = -(state[0][action[1][1] * 2+1] - state[0][action[1][1] * 2 ] - 4)
    #                     break
    #         if action[1][1] == -1:
    #             for kkk in sortedbuffer[6:]:
    #                 if state[0][kkk[0] * 2+1] - state[0][kkk[0] * 2] + state[0][action[1][0] * 2] - state[0][action[1][0] * 2+1] > 8:
    #                     action[1][1] = kkk[0]
    #                     if state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] < 4:
    #                         pAction[1][0] = -(state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] - 4)
    #                     if state[0][action[1][1] * 2 + 1] - state[0][action[1][1] * 2] < 4:
    #                         pAction[1][1] = -(state[0][action[1][1] * 2+1] - state[0][action[1][1] * 2] - 4)
    #                     break
    #         if action[2][0] == -1:
    #             for kkk in sortedbuffer[6:]:
    #                 if state[0][kkk[0] * 2] - state[0][kkk[0] * 2 + 1] + state[0][action[2][1] * 2 + 1] - state[0][action[2][1] * 2] > 8:
    #                     action[2][0] = kkk[0]
    #                     if state[0][action[2][0] * 2] - state[0][action[2][0] * 2 + 1] < 4:
    #                         pAction[2][0] = -(state[0][action[2][0] * 2] - state[0][action[2][0] * 2 + 1] - 4)
    #                     if state[0][action[2][1] * 2 + 1] - state[0][action[2][1] * 2] < 4:
    #                         pAction[2][1] = -(state[0][action[2][1] * 2+1] - state[0][action[2][1] * 2 ] - 4)
    #                     break
    #         if action[2][1] == -1:
    #             for kkk in sortedbuffer[6:]:
    #                 if state[0][kkk[0] * 2+1] - state[0][kkk[0] * 2] + state[0][action[2][0] * 2] - state[0][action[2][0] * 2+1] > 8:
    #                     action[2][1] = kkk[0]
    #                     if state[0][action[2][0] * 2] - state[0][action[2][0] * 2 + 1] < 4:
    #                         pAction[2][0] = -(state[0][action[2][0] * 2] - state[0][action[2][0] * 2 + 1] - 4)
    #                     if state[0][action[2][1] * 2 + 1] - state[0][action[2][1] * 2] < 4:
    #                         pAction[2][1] = -(state[0][action[2][1] * 2+1] - state[0][action[2][1] * 2] - 4)
    #                     break
    #         totalpower += pAction[0][0] + pAction[0][1] + pAction[1][0] + pAction[1][1] + pAction[2][0] + pAction[2][1]
    #         next_state, pktPass, pktLoss = x.step(action, pAction,drop)
    #         totalpass += pktPass
    #         totalloss += pktLoss
    #    # print(run)
    #
    # print('average TTL: ',(13-drop))
    # print('throughput: '+str(totalpass/2000))
    # graf[1][drop] = totalpass / 2000
    # print('packet loss: ',(totalloss/2000))
    # graf[2][drop] = totalloss / 2000
    # print("average power: ",(totalpower/2000))
    # graf[3][drop] = totalpower / 2000

    #########################3    perfect match     ######################
    nf=70

    graf_perfect = np.zeros((4,9))
    graf_perfect[0] = [13,12,11,10,9,8,7,6,5]
    for drop in range(9):
        xp = env('/home/gilmam/Desktop/study/scheduler/nUE10_seed1_len2000', nf=nf)
        totalpass = 0
        totalloss = 0
        totalpower = 0
        for run in range(2000):
            state = xp.state(run)
            sortedbuffer = sort(xp.B.buff)
            choosen = sortedbuffer[0:2*nf]
            w = np.zeros((2*nf, 2*nf))
            xx = np.zeros((2*nf, 2*nf))
            action = np.array([[0, 0] for i in range(nf)])
            pAction = np.array([[0., 0.] for i in range(nf)])

            ####################     perfect match      ######################

            for i in range(2*nf):
                for j in range(2*nf):
                    if np.mod(i, 2) == 0:
                        w[i][j] = state[0][choosen[j][0]*2] - state[0][choosen[j][0]*2+1]
                    else:
                        w[i][j] = state[0][choosen[j][0]*2+1] - state[0][choosen[j][0]*2]

            problem = LpProblem("PerfectMatch", LpMaximize)

            ###############################################################################   LP

            # factories

            x = LpVariable.dicts("x", list(range(nf*nf*4)), 0, 1, cat="Continuous")

            # goal constraint[0,1,2]
            for i in range(nf*2):
                problem += pulp.lpSum(x[j] for j in range(2*nf*i, 2*nf*(i+1))) == 1
                problem += pulp.lpSum(x[j] for j in range(i, nf*nf*4, nf*2)) == 1


            # objective function
            problem += pulp.lpSum([x[2*nf*i+j]*w[i][j] for i in range(2*nf) for j in range(2*nf)])

            #print(problem)

            # solving
            problem.solve()


            clk = np.array([[0, 0] for i in range(nf)])
            for i1 in range(2*nf):
                for i2 in range(2*nf):
                    if x[2*nf*i1+i2].varValue==1:
                        odd = np.mod(i1,2)
                        index = int((i1-odd)/2)
                        action[index][odd] = choosen[i2][0]
                        clk[index][odd]=choosen[i2][1]




            pAction = np.array([[0., 0.] for i in range(nf)])


            for i in range(nf):
                if clk[i][1] < clk[i][0]:
                    if state[0][action[i][1]*2+1]-state[0][action[i][1]*2]<4:
                        pAction[i][1] = -(state[0][action[i][1]*2+1]-state[0][action[i][1]*2]-4)
                    else:
                        if state[0][action[i][0]*2]-state[0][action[i][0]*2+1]<4:
                            pAction[i][0] = -(state[0][action[i][0]*2]-state[0][action[i][0]*2+1]-4)
                else:
                    if state[0][action[i][0]*2]-state[0][action[i][0]*2+1]<4:
                        pAction[i][0] = -(state[0][action[i][0]*2]-state[0][action[i][0]*2+1]-4)
                    else:
                        if state[0][action[i][1]*2+1]-state[0][action[i][1]*2]<4:
                            pAction[i][1] = -(state[0][action[i][1]*2+1]-state[0][action[i][1]*2]-4)



            totalpower += np.sum(pAction)
            next_state, pktPass, pktLoss = xp.step(action, pAction,drop)
            totalpass += pktPass
            totalloss += pktLoss
            print(run)

        print('average TTL: ', (13-drop))
        print('throughput: '+str(totalpass/2000))
        graf_perfect[1][drop] = (totalpass / 2000)
        print('packet loss: ', (totalloss/2000))
        graf_perfect[2][drop] = totalloss / 2000
        print("average power: ", (totalpower/2000))
        graf_perfect[3][drop] = (totalpower / 2000)
    #
    #
    #
    # #########################    random algo    ######################
    #
    # xr = env('/home/gilmam/Desktop/study/scheduler/nUE10_seed1_len2000')
    # graf_rand = np.zeros((4,9))
    # graf_rand[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    # for drop in range(9):
    #
    #     totalpass = 0
    #     totalloss = 0
    #     totalpower = 0
    #     for run in range(2000):
    #         state = xr.state(run)
    #         sortedbuffer = sort(xr.B.buff)
    #         choosen = sortedbuffer[0:4]
    #         action = np.array([[choosen[0][0], choosen[1][0]], [choosen[2][0], choosen[3][0]]])
    #         pAction = np.array([[0., 0.], [0., 0.]])
    #         #
    #         # if choosen[0][1] > choosen[1][1]:
    #         #     if state[0][action[0][1]*2+1]-state[0][action[0][1]*2]<4:
    #         #         pAction[0][1] = -(state[0][action[0][1]*2+1]-state[0][action[0][1]*2]-4)
    #         #     else:
    #         #         if state[0][action[0][0]*2]-state[0][action[0][0]*2+1]<4:
    #         #             pAction[0][0] = -(state[0][action[0][0]*2]-state[0][action[0][0]*2+1]-4)
    #         # else:
    #         #     if state[0][action[0][0]*2]-state[0][action[0][0]*2+1]<4:
    #         #         pAction[0][0] = -(state[0][action[0][0]*2]-state[0][action[0][0]*2+1]-4)
    #         #     else:
    #         #         if state[0][action[0][1]*2+1]-state[0][action[0][1]*2]<4:
    #         #             pAction[0][1] = -(state[0][action[0][1]*2+1]-state[0][action[0][1]*2]-4)
    #         #
    #         # if choosen[2][1] > choosen[3][1]:
    #         #     if state[0][action[1][1]*2+1]-state[0][action[1][1]*2]<4:
    #         #         pAction[1][1] = -(state[0][action[1][1]*2+1]-state[0][action[1][1]*2]-4)
    #         #     else:
    #         #         if state[0][action[1][0]*2]-state[0][action[1][0]*2+1]<4:
    #         #             pAction[1][0] = -(state[0][action[1][0]*2]-state[0][action[1][0]*2+1]-4)
    #         # else:
    #         #     if state[0][action[1][0]*2]-state[0][action[1][0]*2+1]<4:
    #         #         pAction[1][0] = -(state[0][action[1][0]*2]-state[0][action[1][0]*2+1]-4)
    #         #     else:
    #         #         if state[0][action[1][1]*2+1]-state[0][action[1][1]*2]<4:
    #         #             pAction[1][1] = -(state[0][action[1][1]*2+1]-state[0][action[1][1]*2]-4)
    #         #
    #
    #
    #
    #         totalpower += pAction[0][0] + pAction[0][1] + pAction[1][0] + pAction[1][1]
    #         next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
    #         totalpass += pktPass
    #         totalloss += pktLoss
    #       #   print(pktPass, pktLoss)
    #
    #     print('average TTL: ', (13-drop))
    #     print('throughput: '+str(totalpass/2000))
    #     graf_rand[1][drop] = totalpass / 2000
    #     print('packet loss: ', (totalloss/2000))
    #     graf_rand[2][drop] = totalloss / 2000
    #     print("average power: ", (totalpower/2000))
    #     graf_rand[3][drop] = totalpower / 2000
    #
    #
    #
    #
    #
    plt.figure(1)
    #Exhaustive_search, = plt.plot(graf[0], graf[1], 'r', label='Exhaustive search')
    Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[1], 'b', label='Perfect match')
    #random_scheduler, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='Random')
    plt.xlabel('average TTL')
    plt.ylabel('average throughput')
    plt.title('throughput vs. average TTL')
    plt.legend()
    plt.grid()

    plt.figure(2)
    #Exhaustive_search, = plt.plot(graf[0], graf[2], 'r', label='Exhaustive search')
    Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[2], 'b', label='Perfect match')
    #random_scheduler, = plt.plot(graf_rand[0], graf_rand[2], 'g', label='Random')
    plt.xlabel('average TTL')
    plt.ylabel('average packet loss')
    plt.title('packet loss vs. average TTL')
    plt.legend()
    plt.grid()

    plt.figure(3)
    #Exhaustive_search, = plt.plot(graf[0], graf[3], 'r', label='Exhaustive search')
    Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[3], 'b', label='Perfect match')
    #random_scheduler, = plt.plot(graf_rand[0], graf_rand[3], 'g', label='Random')
    plt.xlabel('average TTL')
    plt.ylabel('average power')
    plt.title('power vs. average TTL')
    plt.legend()
    plt.grid()
    #
    # plt.show
    gil=10   # print('trhoput:' + (totalpass / 2000) + 'packet loss:' + (totalloss / 2000) + 'avarge power:' + (totalpower / 2000))