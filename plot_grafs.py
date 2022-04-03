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


# graf_perfect_NN = np.load('graf_perfect_NN.npy')
graf_perfect_det = np.load('graf_perfect_det.npy')
# graf_rand = np.load('graf_rand.npy')
graf_rand_wn_power = np.load('graf_rand_wn_power.npy')
# graf_rand_w_power_n_s = np.load('graf_rand_w_power_n_s.npy')
# graf_rand_w_power_w_s = np.load('graf_rand_w_power_w_s.npy')
graf_perfect = np.load('graf_NN_1f.npy')
# graf_NN_1f_NegP = np.load('graf_negP_10k_NN.npy')
# graf_perfect_NN_100f = np.load('graf_perfect_NN_100f.npy.npy')
# graf_PM_100f_negP = np.load('graf_PM_100f_negP.npy')

plt.figure(1)
# # Exhaustive_search, = plt.plot(graf[0], graf[1], 'r', label='Exhaustive search')
# Perfect_match_NN_100f, = plt.plot(graf_perfect_NN_100f[0], graf_perfect_NN_100f[1], 'm', label='NN-freq reuse factor')
Perfect_match_det_100f, = plt.plot(graf_perfect_det[0], graf_perfect_det[1], 'r', label='Perfect match')
Perfect_match_NN_1f, = plt.plot(graf_perfect[0], graf_perfect[1], 'c', label='NN')
# Perfect_match_NN, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[1], 'b', label='NN')
# # # # random, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='all wrong')
random_npns, = plt.plot(graf_rand_wn_power[0], graf_rand_wn_power[1], 'y', label='Round-robin')
# # random_w_power_n_s, = plt.plot(graf_rand_w_power_n_s[0], graf_rand_w_power_n_s[1], 'k', label='random match w Power and swich')
# # random_scheduler, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='Random')
plt.xlabel('average TTL')
plt.ylabel('average throughput')
plt.title('throughput vs. average TTL')
plt.legend()
plt.grid()
plt.savefig('/home/gilmam/Desktop/study/scheduler/data8x50/gil/throughput_8BS_2f_PM_vs_NN1.png')
plt.figure(2)
# Perfect_match_NN_100f, = plt.plot(graf_perfect_NN_100f[0], graf_perfect_NN_100f[2], 'm', label='NN-freq reuse factor')
# Perfect_match_det_100f, = plt.plot(graf_PM_100f_negP[0], graf_PM_100f_negP[2], 'r', label='Perfect match deterministic')
# Perfect_match_NN_1f, = plt.plot(graf_perfect[0], graf_perfect[2], 'm', label='NN 1F')
# # Perfect_match_NN_1f_negP, = plt.plot(graf_NN_1f_NegP[0], graf_NN_1f_NegP[2], 'c', label='Perfect match NN 1F W NegP')
# # Perfect_match_NN, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[2], 'b', label='Perfect match NN')
# Perfect_match_det, = plt.plot(graf_perfect_det[0], graf_perfect_det[2], 'r', label='Perfect match')
# # random, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='Round Robin')
# random_npns, = plt.plot(graf_rand_wn_power[0], graf_rand_wn_power[2], 'y', label='Round Robin')
# random_w_power_n_s, = plt.plot(graf_rand_w_power_n_s[0], graf_rand_w_power_n_s[2], 'k', label='random match w Power and swich')
Perfect_match_det_100f, = plt.plot(graf_perfect_det[0], graf_perfect_det[2], 'r', label='Perfect match')
Perfect_match_NN_1f, = plt.plot(graf_perfect[0], graf_perfect[2], 'c', label='NN')
# Perfect_match_NN, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[2], 'b', label='NN')
random_npns, = plt.plot(graf_rand_wn_power[0], graf_rand_wn_power[2], 'y', label='Round-robin')
plt.xlabel('average TTL')
plt.ylabel('average packet loss')
plt.title('packet loss vs. average TTL')
plt.legend()
plt.grid()
plt.savefig('/home/gilmam/Desktop/study/scheduler/data8x50/gil/packet_loss_8BS_2f_PM_vs_NN1.png')
plt.figure(3)
# Perfect_match_NN_100f, = plt.plot(graf_perfect_NN_100f[0], graf_perfect_NN_100f[3], 'm', label='NN-freq reuse factor')
# Perfect_match_det_100f, = plt.plot(graf_PM_100f_negP[0], graf_PM_100f_negP[3], 'r', label='Perfect match deterministic')
# Perfect_match_NN_1f, = plt.plot(graf_perfect[0], graf_perfect[3], 'm', label='Perfect match NN 1F')
# Perfect_match_NN_1f_negP, = plt.plot(graf_NN_1f_NegP[0], graf_NN_1f_NegP[3], 'c', label='Perfect match NN 1F W NegP')
# Perfect_match_NN, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[3], 'b', label='Perfect match NN')
# Perfect_match_det, = plt.plot(graf_perfect_det[0], graf_perfect_det[3], 'r', label='Perfect match deterministic')
# # # random, = plt.plot(graf_rand[0], graf_rand[1], 'g', label='all wrong')
# random_npns, = plt.plot(graf_rand_wn_power[0], graf_rand_wn_power[3], 'y', label='random match no Power and swich')
# random_w_power_n_s, = plt.plot(graf_rand_w_power_n_s[0], graf_rand_w_power_n_s[3], 'k', label='random match w Power and swich')
Perfect_match_det_100f, = plt.plot(graf_perfect_det[0], graf_perfect_det[3], 'r', label='Perfect match')
Perfect_match_NN_1f, = plt.plot(graf_perfect[0], graf_perfect[3], 'c', label='NN')
# Perfect_match_NN, = plt.plot(graf_perfect_NN[0], graf_perfect_NN[3], 'b', label='NN')
random_npns, = plt.plot(graf_rand_wn_power[0], graf_rand_wn_power[3], 'y', label='Round-robin')
plt.xlabel('average TTL')
plt.ylabel('average power')
plt.title('power vs. average TTL')
plt.legend()
plt.grid()
plt.savefig('/home/gilmam/Desktop/study/scheduler/data8x50/gil/power_8BS_2f_PM_vs_NN1.png')
plt.show
gil = 10