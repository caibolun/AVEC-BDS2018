# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:23:52 2018

@author: fan
"""


import numpy as np
import pandas as pd
import os

TOPICS_SIZE = 7

time = pd.read_csv('../data/time_7topics.csv').replace(float('nan'), 0)
come_start = np.array(time['why_come_starttime(s)'])[:,np.newaxis]
come_end = np.array(time['why_come_endtime(s)'])[:,np.newaxis]
count1_start = np.array(time['count1_starttime(s)'])[:,np.newaxis]
count1_end = np.array(time['count1_endtime(s)'])[:,np.newaxis]
count2_start = np.array(time['count2_starttime(s)'])[:,np.newaxis]
count2_end = np.array(time['count2_endtime(s)'])[:,np.newaxis]
pic1_start = np.array(time['man_pic_starttime(s)'])[:,np.newaxis]
pic1_end = np.array(time['man_pic_endtime(s)'])[:,np.newaxis]
pic2_start = np.array(time['family_pic_starttime(s)'])[:,np.newaxis]
pic2_end = np.array(time['family_pic_endtime(s)'])[:,np.newaxis]
memory1_start = np.array(time['best_memory_starttime(s)'])[:,np.newaxis]
memory1_end = np.array(time['best_memory_endtime(s)'])[:,np.newaxis]
memory2_start = np.array(time['worst_memory_starttime(s)'])[:,np.newaxis]
memory2_end = np.array(time['worst_memory_endtime(s)'])[:,np.newaxis]

start_time = np.concatenate((come_start, count1_start, count2_start, pic1_start, pic2_start, memory1_start,memory2_start), axis=0)
end_time = np.concatenate((come_end, count1_end, count2_end, pic1_end, pic2_end, memory1_end,memory2_end), axis=0)
time = np.concatenate((start_time, end_time), axis=1)

path_root = '../data'  
#path_lld = path_root + '/LLDs_audio_eGeMAPS/'
path_lld = path_root + '/LLDs_audio_opensmile_MFCCs/'
filename = os.listdir(path_lld)

features = []
ID = []
for i in range(len(filename)):
    print(i)
    temp_lld = np.float64(pd.read_csv(path_lld+filename[i],sep=';').iloc[:,2:])

    feat_topics = []
    for j in range(TOPICS_SIZE):
        index_start = int(100*time[len(filename)*j+i, 0])
        index_end = int(100*time[len(filename)*j+i, 1])        
        
        if index_start == index_end:
            feat = list(-1 * np.zeros((temp_lld.shape[1],4)))
            feat_topics = feat_topics + feat
            continue
        
        feat_lld = temp_lld[index_start:index_end, :]
        
        lld_max = np.max(feat_lld,axis=0)[:,np.newaxis]
        lld_min = np.min(feat_lld,axis=0)[:,np.newaxis]
        lld_mean = np.mean(feat_lld,axis=0)[:,np.newaxis]
        lld_std = np.std(feat_lld,axis=0)[:,np.newaxis]  
        
        feat = list(np.concatenate((lld_max,lld_min,lld_mean,lld_std),axis=1))
        feat_topics = feat_topics + feat
        
    ID.append(filename[i])
    features.append(np.array(feat_topics).reshape(-1))      

features = pd.DataFrame(features)
features['ID'] = ID
features.to_csv('../features/mfcc_7topics.csv',header=None,index=None)
#features.to_csv('../features/egemaps_7topics.csv',header=None,index=None)