# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:35:25 2018

@author: fan
"""


import numpy as np
import pandas as pd
import os

path_root = '../data'
path_turns = path_root + '/VAD_turns/'
filename = os.listdir(path_turns)

## pause time
ID = []
pause_time = []
for i in range(len(filename)):
    file = pd.read_csv(path_turns+filename[i], sep=';',header=None)
    
    duration = []
    for j in range(len(file)-1):
        duration.append(file.iloc[j+1,0]-file.iloc[j,1])

    pause_time.append((np.min(duration), np.max(duration), np.mean(duration), np.std(duration), np.sum(duration)))
    ID.append(filename[i][:-4])
pause_time=np.array(pause_time)
features = pd.DataFrame(pause_time)
features['ID'] = ID
features.to_csv('../features/pause_time.csv',header=None,index=None)

## length
length = []
for i in range(len(filename)):
    file = pd.read_csv(path_turns+filename[i], sep=';',header=None)
    
    duration = []
    for j in range(len(file)):
        duration.append(file.iloc[j,1]-file.iloc[j,0])

    length.append((np.min(duration), np.max(duration), np.mean(duration), np.std(duration), np.sum(duration), file.iloc[j,1]))
length=np.array(length)
features = pd.DataFrame(length)
features['ID'] = ID
features.to_csv('../features/duration.csv',header=None,index=None)