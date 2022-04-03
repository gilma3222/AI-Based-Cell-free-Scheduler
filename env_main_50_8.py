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
    # x = env('/home/gilmam/Desktop/study/scheduler/data8x50/gil/nUE50nRu4_seed1_len2000')
    # graf = np.zeros((4,9))
    # graf[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    # for drop in range(9):
    #
    #     totalpass = 0
    #     totalloss = 0
    #     totalpower = 0
    #     for run in range(2000):
    #         state = x.state(run)
    #         sortedbuffer = sort(x.B.buff)
    #         choosen = sortedbuffer[0:16]

        #     ####################     Exhaustive search    ######################
        #
        #     max_reward=0
        #     for i1 in range(4):
        #         for i2 in range(4):
        #             for i3 in range(4):
        #                 for i4 in range(4):
        #                     if i1!=i2 and i1!=i3 and i1!=i4 and i3!=i2 and i4!=i2 and i3!=i4:
        #                         reward = 0
        #                         powerop=np.array([[0.,0.],[0.,0.]])
        #                         aop=np.array([[choosen[i1][0], choosen[i2][0]],[choosen[i3][0],choosen[i4][0]]])
        #                         A_1 = state[0][choosen[i1][0]*2] - state[0][choosen[i1][0]*2+1] + state[0][choosen[i2][0]*2+1] - state[0][choosen[i2][0]*2]
        #                         A_2 = state[0][choosen[i3][0]*2] - state[0][choosen[i3][0]*2+1] + state[0][choosen[i4][0]*2+1] - state[0][choosen[i4][0]*2]
        #                         if A_1<8:
        #                             if choosen[i1][1]<choosen[i2][1]:
        #                                 if state[0][choosen[i1][0]*2]-state[0][choosen[i1][0]*2+1]>0:
        #                                     aop[0][1] = -1
        #                                 else:
        #                                     aop[0][1] = aop[0][0]
        #                                     aop[0][0] = -1
        #                             else:
        #                                 if state[0][choosen[i2][0] * 2+1] - state[0][choosen[i2][0] * 2] > 0:
        #                                     aop[0][0] = -1
        #                                 else:
        #                                     aop[0][0] = aop[0][1]
        #                                     aop[0][1] = -1
        #                         else:
        #                             if state[0][choosen[i1][0]*2] - state[0][choosen[i1][0]*2+1]<4:
        #                                 powerop[0][0] = -(state[0][choosen[i1][0]*2] - state[0][choosen[i1][0]*2+1]-4)
        #                             if state[0][choosen[i2][0]*2+1] - state[0][choosen[i2][0]*2]<4:
        #                                 powerop[0][1] = -(state[0][choosen[i2][0]*2+1] - state[0][choosen[i2][0]*2]-4)
        #
        #
        #                         if A_2<8:
        #                             if choosen[i3][1] < choosen[i4][1]:
        #                                 if state[0][choosen[i3][0] * 2] - state[0][choosen[i3][0] * 2 + 1] > 0:
        #                                     aop[1][1] = -1
        #                                 else:
        #                                     aop[1][1] = aop[1][0]
        #                                     aop[1][0] = -1
        #                             else:
        #                                 if state[0][choosen[i4][0] * 2 + 1] - state[0][choosen[i4][0] * 2] > 0:
        #                                     aop[1][0] = -1
        #                                 else:
        #                                     aop[1][0] = aop[1][1]
        #                                     aop[1][1] = -1
        #                         else:
        #                             if state[0][choosen[i3][0] * 2] - state[0][choosen[i3][0] * 2 + 1] < 4:
        #                                 powerop[1][0] = -(state[0][choosen[i3][0] * 2] - state[0][choosen[i3][0] * 2 + 1] - 4)
        #                             if state[0][choosen[i4][0] * 2 + 1] - state[0][choosen[i4][0] * 2] < 4:
        #                                 powerop[1][1] = -(state[0][choosen[i4][0] * 2 + 1] - state[0][choosen[i4][0] * 2] - 4)
        #                         reward = 50*(1*(aop[0][0]>=0) + 1*(aop[0][1]>=0) + 1*(aop[1][0]>=0) + 1*(aop[1][1]>=0))-(powerop[0][0] + powerop[0][1] + powerop[1][0] + powerop[1][1])
        #                         if reward>max_reward:
        #                             max_reward = reward
        #                             action = aop
        #                             pAction = powerop
        #
        #     if action[0][0] == -1:
        #         for kkk in sortedbuffer[4:]:
        #             if state[0][kkk[0]*2]-state[0][kkk[0]*2+1]+state[0][action[0][1]*2+1]-state[0][action[0][1]*2]>8:
        #                 action[0][0] = kkk[0]
        #                 if state[0][action[0][0]*2]-state[0][action[0][0]*2+1]<4:
        #                     pAction[0][0] = -(state[0][action[0][0]*2]-state[0][action[0][0]*2+1]-4)
        #                 if state[0][action[0][1]*2+1]-state[0][action[0][1]*2]<4:
        #                     pAction[0][1] = -(state[0][action[0][1]*2+1]-state[0][action[0][1]*2]-4)
        #                 break
        #     if action[0][1] == -1:
        #         for kkk in sortedbuffer[4:]:
        #             if state[0][kkk[0] * 2+1] - state[0][kkk[0] * 2] + state[0][action[0][0] * 2] - state[0][action[0][0] * 2+1] > 8:
        #                 action[0][1] = kkk[0]
        #                 if state[0][action[0][1] * 2+1] - state[0][action[0][1] * 2] < 4:
        #                     pAction[0][1] = -(state[0][action[0][1] * 2+1] - state[0][action[0][1] * 2] - 4)
        #                 if state[0][action[0][0] * 2] - state[0][action[0][0] * 2+1] < 4:
        #                     pAction[0][0] = -(state[0][action[0][0] * 2] - state[0][action[0][0] * 2+1] - 4)
        #                 break
        #     if action[1][0] == -1:
        #         for kkk in sortedbuffer[4:]:
        #             if state[0][kkk[0] * 2] - state[0][kkk[0] * 2 + 1] + state[0][action[1][1] * 2 + 1] - state[0][action[1][1] * 2] > 8:
        #                 action[1][0] = kkk[0]
        #                 if state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] < 4:
        #                     pAction[1][0] = -(state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] - 4)
        #                 if state[0][action[1][1] * 2 + 1] - state[0][action[1][1] * 2] < 4:
        #                     pAction[1][1] = -(state[0][action[1][1] * 2+1] - state[0][action[1][1] * 2 ] - 4)
        #                 break
        #     if action[1][1] == -1:
        #         for kkk in sortedbuffer[4:]:
        #             if state[0][kkk[0] * 2+1] - state[0][kkk[0] * 2] + state[0][action[1][0] * 2] - state[0][action[1][0] * 2+1] > 8:
        #                 action[1][1] = kkk[0]
        #                 if state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] < 4:
        #                     pAction[1][0] = -(state[0][action[1][0] * 2] - state[0][action[1][0] * 2 + 1] - 4)
        #                 if state[0][action[1][1] * 2 + 1] - state[0][action[1][1] * 2] < 4:
        #                     pAction[1][1] = -(state[0][action[1][1] * 2+1] - state[0][action[1][1] * 2] - 4)
        #                 break
        #     totalpower += pAction[0][0] + pAction[0][1] + pAction[1][0] + pAction[1][1]
        #     next_state, pktPass, pktLoss = x.step(action, pAction,drop)
        #     totalpass += pktPass
        #     totalloss += pktLoss
        #  #   print(pktPass, pktLoss)
        #
        # print('average TTL: ',(13-drop))
        # print('throughput: '+str(totalpass/2000))
        # graf[1][drop] = totalpass / 2000
        # print('packet loss: ',(totalloss/2000))
        # graf[2][drop] = totalloss / 2000
        # print("average power: ",(totalpower/2000))
        # graf[3][drop] = totalpower / 2000

        #########################3    perfect match     ######################
    nbs = 8
    nf = 2
    graf_perfect = np.zeros((4, 9))
    graf_perfect[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    labels8x50 = []
    buff8x50 = []
    state8x50 = []
    for drop in range(9):
        xp = env('/home/gilmam/Desktop/study/scheduler/data8x50/gil/nUE50nRu4_seed1_len2000', nf=nf)
        totalpass = 0
        totalloss = 0
        totalpower = 0
        for run in range(2000):
            state = xp.state(run)
            sortedbuffer = sort(xp.B.buff)
            choosen = sortedbuffer[0:nbs * nf]
            w = np.zeros((nbs * nf, nbs * nf))
            xx = np.zeros((nbs * nf, nbs * nf))
            action = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
            pAction = np.array([[0., 0., 0., 0., 0., 0., 0., 0.] for i in range(nf)])

            ####################     perfect match      ######################

            for i in range(nbs * nf):
                for j in range(nbs * nf):
                    ii = np.mod(i, nbs)
                    w[i][j] = state[0][(choosen[j][0] * nbs) + ii] # - state[0][choosen[j][0] * 2 + 1]
                    # else:
                    #     w[i][j] = state[0][choosen[j][0] * 2 + 1]# - state[0][choosen[j][0] * 2]

            problem = LpProblem("PerfectMatch", LpMaximize)

                ###############################################################################   LP

                # factories

            x = LpVariable.dicts("x", list(range(nf*nf*nbs*nbs)), 0, 1, cat="Continuous")

                # goal constraint[0,1,2]
            for i in range(nf * nbs):
                problem += pulp.lpSum(x[j] for j in range(nbs * nf * i, nbs * nf * (i + 1))) == 1
                problem += pulp.lpSum(x[j] for j in range(i, nf * nf * nbs * nbs, nf * nbs)) == 1

                # objective function
            problem += pulp.lpSum([x[nbs * nf * i + j] * w[i][j] for i in range(nbs * nf) for j in range(nbs * nf)])

          #  print(problem)

                # solving
            problem.solve()

            clk = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
            for i1 in range(nbs * nf):
                for i2 in range(nbs * nf):
                    if x[nbs * nf * i1 + i2].varValue == 1:
                        if i1<nbs:
                            freq_ind = 0
                        else:
                            freq_ind = 1
                        bs_ind = np.mod(i1, nbs)
                        action[freq_ind][bs_ind] = choosen[i2][0]
                        clk[freq_ind][bs_ind] = choosen[i2][1]

#############    pwoer calc      #############3
            e = np.eye(nbs).astype(int)
            for fr in range(nf):
                t = np.array([state[0][(action[fr][i])*nbs:action[fr][i]*nbs+nbs] for i in range(nbs)])#(rssi[np.abs(k)] + w) * (np.array(k) >= 0)
                P = np.diag(t)
                ind_sort = np.argsort(clk[fr])
                inter = t[e == 0]
                max_inter = np.array([np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
                difm = t - np.reshape(max_inter, (nbs, 1))
                difm=difm-4
                for kk in range(nbs):
                    if difm[ind_sort[kk]][ind_sort[kk]]<0: # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
                        pAction[fr,ind_sort[kk]] = -difm[ind_sort[kk]][ind_sort[kk]]
                        # tmp = np.transpose(t)
                        # tmp[ind_sort[kk]] += pAction[fr,ind_sort[kk]]
                        # t = np.transpose(tmp)
                        # P = np.diag(t)
                        # inter = t[e == 0]
                        # max_inter = np.array([np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
                        # difm = t - np.reshape(max_inter, (nbs, 1))
                        # difm = difm - 4

            labels8x50.append(action)
            totalpower += np.sum(pAction)
            next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
            totalpass += pktPass
            totalloss += pktLoss
            if np.mod(run, 100)==0:
                print(run)

        print('average TTL: ', (13 - drop))
        print('throughput: ' + str(totalpass / 2000))
        graf_perfect[1][drop] = (totalpass / 2000)
        print('packet loss: ', (totalloss / 2000))
        graf_perfect[2][drop] = totalloss / 2000
        print("average power: ", (totalpower / 2000))
        graf_perfect[3][drop] = (totalpower / 2000)

        # np.save('perfect_state_2x2_200k.npy', perfect_state)
        # np.save('perfect_buffer_2x2_200k.npy', perfect_buff)
        # np.save('perfect_labels_2x2_200k.npy', perfect_labels)
        #########################3    perfect match with power    ######################
    nbs = 8
    nf = 2
    graf_stable = np.zeros((4, 9))
    graf_stable[0] = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    labels8x50 = []
    buff8x50 = []
    state8x50 = []
    for drop in range(9):
        xp = env('/home/gilmam/Desktop/study/scheduler/data8x50/gil/nUE50nRu4_seed1_len2000', nf=nf)
        totalpass = 0
        totalloss = 0
        totalpower = 0
        for run in range(2000):
            state = xp.state(run)
            sortedbuffer = sort(xp.B.buff)
            choosen = sortedbuffer[0:nbs * nf]
            w = np.zeros((nbs * nf, nbs * nf))
            xx = np.zeros((nbs * nf, nbs * nf))
            action = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
            pAction = np.array([[0., 0., 0., 0., 0., 0., 0., 0.] for i in range(nf)])

            ####################     perfect match      ######################

            for i in range(nbs * nf):
                for j in range(nbs * nf):
                    ii = np.mod(i, nbs)
                    w[i][j] = state[0][(choosen[j][0] * nbs) + ii]  # - state[0][choosen[j][0] * 2 + 1]
                    # else:
                    #     w[i][j] = state[0][choosen[j][0] * 2 + 1]# - state[0][choosen[j][0] * 2]

            problem = LpProblem("PerfectMatch", LpMaximize)

            ###############################################################################   LP

            # factories

            x = LpVariable.dicts("x", list(range(nf * nf * nbs * nbs)), 0, 1, cat="Continuous")

            # goal constraint[0,1,2]
            for i in range(nf * nbs):
                problem += pulp.lpSum(x[j] for j in range(nbs * nf * i, nbs * nf * (i + 1))) == 1
                problem += pulp.lpSum(x[j] for j in range(i, nf * nf * nbs * nbs, nf * nbs)) == 1

                # objective function
            problem += pulp.lpSum([x[nbs * nf * i + j] * w[i][j] for i in range(nbs * nf) for j in range(nbs * nf)])

            #  print(problem)

            # solving
            problem.solve()

            clk = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
            for i1 in range(nbs * nf):
                for i2 in range(nbs * nf):
                    if x[nbs * nf * i1 + i2].varValue == 1:
                        if i1 < nbs:
                            freq_ind = 0
                        else:
                            freq_ind = 1
                        bs_ind = np.mod(i1, nbs)
                        action[freq_ind][bs_ind] = choosen[i2][0]
                        clk[freq_ind][bs_ind] = choosen[i2][1]

            #############    pwoer calc      #############3
            e = np.eye(nbs).astype(int)
            for fr in range(nf):
                t = np.array([state[0][(action[fr][i])*nbs:action[fr][i]*nbs+nbs] for i in range(nbs)])#(rssi[np.abs(k)] + w) * (np.array(k) >= 0)
                P = np.diag(t)
                ind_sort = np.argsort(clk[fr])
                inter = t[e == 0]
                max_inter = np.array([np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
                difm = t - np.reshape(max_inter, (nbs, 1))
                difm=difm-4
                for kk in range(nbs):
                    if difm[ind_sort[kk]][ind_sort[kk]]<0: # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
                        pAction[fr,ind_sort[kk]] = -difm[ind_sort[kk]][ind_sort[kk]]
                        # tmp = np.transpose(t)
                        # tmp[ind_sort[kk]] += pAction[fr,ind_sort[kk]]
                        # t = np.transpose(tmp)
                        # P = np.diag(t)
                        # inter = t[e == 0]
                        # max_inter = np.array([np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
                        # difm = t - np.reshape(max_inter, (nbs, 1))
                        # difm = difm - 4

            # labels8x50.append(action)
            totalpower += np.sum(pAction)
            next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
            totalpass += pktPass
            totalloss += pktLoss
            if np.mod(run, 100) == 0:
                print(run)

        print('average TTL: ', (13 - drop))
        print('throughput: ' + str(totalpass / 2000))
        graf_stable[1][drop] = (totalpass / 2000)
        print('packet loss: ', (totalloss / 2000))
        graf_stable[2][drop] = totalloss / 2000
        print("average power: ", (totalpower / 2000))
        graf_stable[3][drop] = (totalpower / 2000)


    plt.figure(1)
   # Exhaustive_search, = plt.plot(graf[0], graf[1], 'r', label='Exhaustive search')
    Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[1], 'b', label='Perfect match')
   # random_scheduler, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='Random')
    Stable_match, = plt.plot(graf_stable[0], graf_stable[1], 'y', label='Perfect match with Power')
    plt.xlabel('average TTL')
    plt.ylabel('average throughput')
    plt.title('throughput vs. average TTL')
    plt.legend()
    plt.grid()

    plt.figure(2)
  #  Exhaustive_search, = plt.plot(graf[0], graf[2], 'r', label='Exhaustive search')
    Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[2], 'b', label='Perfect match')
   # random_scheduler, = plt.plot(graf_rand[0], graf_rand[2], 'g', label='Random')
    Stable_match, = plt.plot(graf_stable[0], graf_stable[2], 'y', label='Perfect match with Power')
    plt.xlabel('average TTL')
    plt.ylabel('average packet loss')
    plt.title('packet loss vs. average TTL')
    plt.legend()
    plt.grid()

    plt.figure(3)
 #   Exhaustive_search, = plt.plot(graf[0], graf[3], 'r', label='Exhaustive search')
    Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[3], 'b', label='Perfect match')
   # random_scheduler, = plt.plot(graf_rand[0], graf_rand[3], 'g', label='Random')
    Stable_match, = plt.plot(graf_stable[0], graf_stable[3], 'y', label='Perfect match with Power')
    plt.xlabel('average TTL')
    plt.ylabel('average power')
    plt.title('power vs. average TTL')
    plt.legend()
    plt.grid()

    plt.show
    gil=10   # print('trhoput:' + (totalpass / 2000) + 'packet loss:' + (totalloss / 2000) + 'avarge power:' + (totalpower / 2000))