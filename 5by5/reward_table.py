import networkx as nx
import os
import numpy as np
import json

ep = [0.2]
# [.1,.1,.1,.15,.15,.15,.2,.2,.2]
prdes = [0.8]
# [0.6,0.7,0.8,0.6,0.7,0.8,0.6,0.7,0.8]

rew_list = []
for idx,i in enumerate(ep):
    j = prdes[idx]
    fname = './{}_{}/ep_rewards.npy'.format(j,i)
    rew_data = np.load(fname)
    rew_list.append(rew_data)

for r,i in enumerate(rew_list):
    print("\n","    epsilon: ",ep[r],"     pr_des :",prdes[r],"\n")
    y=[3000,20000,80000,200000, 396000]
    # [3000,10000,20000,30000,40000,80000,120000,160000,196000,230000,260000,296000]
    d=3000
    for k in y:
        print(k,'        {:<7}'.format(sum(i[k-d:k+d])/(d+d)))
sat_list = []
for idx,i in enumerate(ep):
    j = prdes[idx]
    fname = './{}_{}/training_sat_count.txt'.format(j,i)
    with open(fname) as fp:
        sat_data=json.load(fp)
        sat_list.append(sum(sat_data)/len(sat_data))
print(sat_list)

