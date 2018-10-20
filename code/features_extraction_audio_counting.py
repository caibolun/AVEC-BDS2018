# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:26:30 2018

@author: fan
"""


import os
import numpy as np
import pandas as pd


# Hyperparameters
time = np.float64(pd.read_csv('../data/time_3topics.csv',header=None)) 
time = np.array([time[t] for t in range(len(time)) if t % 3 == 1])  # the counting part

path_root = '../data' 
path_egemaps = path_root + '/LLDs_audio_eGeMAPS/'
path_mfcc = path_root + '/LLDs_audio_opensmile_MFCCs/'
filename = os.listdir(path_egemaps)

features = []
ID = []
for i in range(len(filename)):
    print(i)
    temp_egemaps = np.float64(pd.read_csv(path_egemaps+filename[i],sep=';').iloc[:,2:])
    temp_mfcc = np.float64(pd.read_csv(path_mfcc+filename[i],sep=';').iloc[:,2:])	     

    feat_topics = []
    index_start = int(100*time[i, 0])
    index_end = int(100*time[i, 1])
    
    if index_start == index_end:
        feat = list(-1 * np.ones((62,3)))
        feat_topics = feat_topics + feat
        continue
    
    feat_egemaps = temp_egemaps[index_start:index_end, :]
    feat_mfcc = temp_mfcc[index_start:index_end, :]
    
    egemaps_max = np.max(feat_egemaps,axis=0)
    egemaps_min = np.min(feat_egemaps,axis=0)
    egemaps_mean = np.mean(feat_egemaps,axis=0)  
    
    mfcc_max = np.max(feat_mfcc,axis=0)
    mfcc_min = np.min(feat_mfcc,axis=0)
    mfcc_mean = np.mean(feat_mfcc,axis=0)
    
    # 62*3
    max_all = np.concatenate((egemaps_max,mfcc_max))[:,np.newaxis]
    min_all = np.concatenate((egemaps_min,mfcc_min))[:,np.newaxis]
    mean_all = np.concatenate((egemaps_mean,mfcc_mean))[:,np.newaxis]
    feat = list(np.concatenate((max_all,min_all,mean_all),axis=1))
    feat_topics = feat_topics + feat
        
    ID.append(filename[i][0:3])
    features.append(np.array(feat_topics).reshape(-1))      

features = pd.DataFrame(features)
features['ID'] = ID
features.to_csv('../features/egemaps_mfcc_counting.csv',header=None,index=None)