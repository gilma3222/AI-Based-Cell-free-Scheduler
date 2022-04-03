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



###################################################################################3
class buffer:
    totalLoss = 0
    def __init__(self,size=40,nUE=50,rangeTTL=[8, 10, 12, 14, 16, 18]):
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

        #########################3    perfect match     ######################
    nbs = 8
    nf = 1
    drop = 0
    num_set = 500
    num_runs = 20000
    for set_index in range(num_set):
        labels8x50 = []
        buff8x50 = []
        state8x50 = []
        xp = env('/home/gilmam/Desktop/study/scheduler/data8x50/gil/nUE50nRu4_seed1_len2000', nf=nf)
        totalpass = 0
        totalloss = 0
        totalpower = 0
        for run in range(num_runs):
            state = xp.state(run)
            state8x50.append(state)
            sortedbuffer = sort(xp.B.buff)
            action = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
            pAction = np.array([[0., 0., 0., 0., 0., 0., 0., 0.] for i in range(nf)])
            clk = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
            for fr in range(nf):
                choosen = sortedbuffer[0:nbs]
                # xcopy = copy.deepcopy(choosen)
                # buff8x50.append(xcopy)
                w = np.zeros((nbs, nbs))
                # xx = np.zeros((nbs, nbs))
                ####################     perfect match      ######################
                ####################     Set the W wigths      ###################### w=[bs][msg]
                for i in range(nbs):
                    for j in range(nbs):
                        w[i][j] = state[0][(choosen[j][0] * nbs) + i]

                problem = LpProblem("PerfectMatch", LpMaximize)

                    ###############################################################################   LP

                    # factories

                x = LpVariable.dicts("x", list(range(nbs*nbs)), 0, 1, cat="Continuous")

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

    # #############    pwoer calc      #############3
    #         cng = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(nf)])
    #         e = np.eye(nbs).astype(int)
    #         t = np.zeros((8,8,2))
    #         for fr in range(nf):
    #             ind_sort = np.argsort(clk[fr])
    #             t[:,:,fr] = np.array([state[0][(action[fr][i]) * nbs:action[fr][i] * nbs + nbs] for i in range(nbs)])
    #             for kk in range(nbs):
    #                 ti = t[:,:,fr] + np.tile(pAction[fr], (nbs, 1))
    #                 inter = ti[e == 0]
    #                 max_inter = np.array([np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
    #                 difm = ti - np.reshape(max_inter, (nbs, 1))-4
    #                 if difm[ind_sort[kk]][ind_sort[kk]]<0: # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
    #                     flagw=1
    #                     for cp in range(kk):
    #                         if cng[fr,ind_sort[cp]]==0:
    #                             if difm[ind_sort[cp]][ind_sort[cp]] - difm[ind_sort[cp]][ind_sort[kk]] + difm[ind_sort[kk]][
    #                                 ind_sort[kk]] < 4.00000001:
    #                                 flagw = 0
    #                                 cng[fr][ind_sort[kk]] = 1
    #                                 indi = choosen.index([action[fr][ind_sort[kk]], clk[fr][ind_sort[kk]]])
    #                                 del choosen[indi]
    #                                 action[fr][ind_sort[kk]] = -1
    #                                 t[ind_sort[kk], :, fr] = 0
    #
    #                                 break
    #                     if flagw:
    #                         pAction[fr,ind_sort[kk]] = -difm[ind_sort[kk]][ind_sort[kk]]
    #
    #         #############    cng from the buffer     #############3
    #         toolate = []
    #         for kkk in sortedbuffer[nbs * nf:]:
    #             if np.sum(cng):
    #                 flag = 1
    #                 for j in toolate:
    #                     if kkk == j:
    #                         flag = 0
    #                         break
    #                 if flag:   #the pkt from the buffer is avelable
    #                     indmax = np.argmax(state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]+pAction[fr])
    #                    # rmax = state[0][(kkk[0]) * nbs+indmax]
    #                     for fr in range(nf):
    #                         if cng[fr][indmax]:
    #                             ti = t[:,:,fr] + np.tile(pAction[fr], (nbs, 1))
    #                             ti[indmax] += state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
    #                             inter = ti[e == 0]
    #                             max_inter = np.array(
    #                                 [np.max(inter[(i):(i + nbs - 1)]) for i in range(0, nbs * (nbs - 1), (nbs - 1))])
    #                             difm = ti - np.reshape(max_inter, (nbs, 1)) - 4
    #                             if difm[indmax][indmax] < 0:  # P[ind_sort[kk]]+pAction[0][ind_sort[kk]]-4<max_inter[ind_sort[kk]]+pAction[0][ind_sort[kk]]:
    #                                 flagw = 1
    #                                 for cp in range(nbs):
    #                                     if cng[fr, cp] == 0:
    #                                         if difm[cp][cp] - difm[cp][indmax] + difm[indmax][indmax] < 4.00000001:
    #                                             #not good for cng
    #                                             flagw = 0
    #                                             break
    #                                 if flagw:
    #                                     #good for cng w power
    #                                     pAction[fr, indmax] = -difm[indmax][indmax]
    #                                     cng[fr][indmax] = 0
    #                                     action[fr][indmax] = kkk[0]
    #                                     clk[fr][indmax] = kkk[1]
    #                                     choosen.append([kkk[0],kkk[1]])
    #                                     t[indmax, :, fr] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
    #                                     toolate.append(kkk)
    #                                     break
    #                             else:
    #                                 #good for cng w no power
    #                                 cng[fr][indmax] = 0
    #                                 action[fr][indmax] = kkk[0]
    #                                 clk[fr][indmax] = kkk[1]
    #                                 choosen.append([kkk[0], kkk[1]])
    #                                 t[indmax, :, fr] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
    #                                 toolate.append(kkk)
    #                                 break
    #
    #         for i in range(nf):
    #             for j in range(nbs):
    #                 if action[i][j]==-1:
    #                     action[i][j] = 12
    #                     choosen.append([12, 20])

                            # replace
                            # if cng[fr][bs]:
                            #     t = np.array([state[0][(action[fr][i]) * nbs:action[fr][i] * nbs + nbs] for i in range(nbs)])
                            #     ti = t + np.tile(pAction[fr], (nbs, 1))
                            #     ti[bs] = state[0][(kkk[0]) * nbs:kkk[0] * nbs + nbs]
                            #     if ti[][]

            #         for kkk in sortedbuffer[2 * nf:]:
            #             flag = 1
            #             for j in toolate:
            #                 if kkk == j:
            #                     flag = 0
            #                     break
            #             if flag:
            #                 if state[0][kkk[0] * 2] - state[0][kkk[0] * 2 + 1] + state[0][action[i][1] * 2 + 1] - \
            #                         state[0][action[i][1] * 2] > 8:
            #                     action[i][0] = kkk[0]
            #                     toolate.append(kkk)
            #                     if state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] < 4:
            #                         pAction[i][0] = -(state[0][action[i][0] * 2] - state[0][action[i][0] * 2 + 1] - 4)
            #                     if state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] < 4:
            #                         pAction[i][1] = -(state[0][action[i][1] * 2 + 1] - state[0][action[i][1] * 2] - 4)
            #                     break

            xcopy = copy.deepcopy(choosen)
            buff8x50.append(xcopy)
            labels8x50.append(action)
            totalpower += np.sum(pAction)
            next_state, pktPass, pktLoss = xp.step(action, pAction, drop)
            totalpass += pktPass
            totalloss += pktLoss
            if np.mod(run, 1000)==1:
                print('run:', run)
                print('average TTL: ', (13 - drop))
                print('throughput: ' + str(totalpass / run))
                print('packet loss: ', (totalloss / run))
                print("average power: ", (totalpower / run))
        # print('average TTL: ', (13 - drop))
        # print('throughput: ' + str(totalpass / 2000))
        # graf_perfect[1][drop] = (totalpass / 2000)
        # print('packet loss: ', (totalloss / 2000))
        # graf_perfect[2][drop] = totalloss / 2000
        # print("average power: ", (totalpower / 2000))
        # graf_perfect[3][drop] = (totalpower / 2000)
        print('set_index:', set_index)
        os.chdir("/home/gilmam/Desktop/study/scheduler/data8x50/gil/set_data/f1/test1")
        np.save('perfect_state_8x2_1f_20k_set{}.npy'.format(set_index), state8x50)
        np.save('perfect_buffer_8x2_1f_20k_set{}.npy'.format(set_index), buff8x50)
        np.save('perfect_labels_8x2_1f_20k_set{}.npy'.format(set_index), labels8x50)

 #
 #
 #    plt.figure(1)
 #   # Exhaustive_search, = plt.plot(graf[0], graf[1], 'r', label='Exhaustive search')
 #    Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[1], 'b', label='Perfect match')
 #   # random_scheduler, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='Random')
 #    Stable_match, = plt.plot(graf_stable[0], graf_stable[1], 'y', label='Perfect match with Power')
 #    plt.xlabel('average TTL')
 #    plt.ylabel('average throughput')
 #    plt.title('throughput vs. average TTL')
 #    plt.legend()
 #    plt.grid()
 #
 #    plt.figure(2)
 #  #  Exhaustive_search, = plt.plot(graf[0], graf[2], 'r', label='Exhaustive search')
 #    Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[2], 'b', label='Perfect match')
 #   # random_scheduler, = plt.plot(graf_rand[0], graf_rand[2], 'g', label='Random')
 #    Stable_match, = plt.plot(graf_stable[0], graf_stable[2], 'y', label='Perfect match with Power')
 #    plt.xlabel('average TTL')
 #    plt.ylabel('average packet loss')
 #    plt.title('packet loss vs. average TTL')
 #    plt.legend()
 #    plt.grid()
 #
 #    plt.figure(3)
 # #   Exhaustive_search, = plt.plot(graf[0], graf[3], 'r', label='Exhaustive search')
 #    Perfect_match, = plt.plot(graf_perfect[0], graf_perfect[3], 'b', label='Perfect match')
 #   # random_scheduler, = plt.plot(graf_rand[0], graf_rand[3], 'g', label='Random')
 #    Stable_match, = plt.plot(graf_stable[0], graf_stable[3], 'y', label='Perfect match with Power')
 #    plt.xlabel('average TTL')
 #    plt.ylabel('average power')
 #    plt.title('power vs. average TTL')
 #    plt.legend()
 #    plt.grid()
 #
 #    plt.show
 #    gil=10   # print('trhoput:' + (totalpass / 2000) + 'packet loss:' + (totalloss / 2000) + 'avarge power:' + (totalpower / 2000))