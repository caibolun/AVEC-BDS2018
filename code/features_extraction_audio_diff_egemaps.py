# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:41:38 2018

@author: fan
"""

import numpy as np
import pandas as pd
import os

TOPICS_SIZE = 2

time = pd.read_csv('../data/time_7topics.csv').replace(float('nan'), 0)
count1_start = np.array(time['count1_starttime(s)'])[:,np.newaxis]
count1_end = np.array(time['count1_endtime(s)'])[:,np.newaxis]
count2_start = np.array(time['count2_starttime(s)'])[:,np.newaxis]
count2_end = np.array(time['count2_endtime(s)'])[:,np.newaxis]

start_time = np.concatenate((count1_start, count2_start), axis=0)
end_time = np.concatenate((count1_end, count2_end), axis=0)
time = np.concatenate((start_time, end_time), axis=1)

path_root = '../data'  
path_lld = path_root + '/LLDs_audio_eGeMAPS/'
#path_lld = path_root + '/LLDs_audio_opensmile_MFCCs/'
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
            if j==0:
                feat_topics = feat_topics + feat
            continue
        
        feat_lld = temp_lld[index_start:index_end, :]
        
        lld_max = np.max(feat_lld,axis=0)[:,np.newaxis]
        lld_min = np.min(feat_lld,axis=0)[:,np.newaxis]
        lld_mean = np.mean(feat_lld,axis=0)[:,np.newaxis]
        lld_std = np.std(feat_lld,axis=0)[:,np.newaxis]  
        
        feat = list(np.concatenate((lld_max,lld_min,lld_mean,lld_std),axis=1))
        feat_topics = feat_topics + feat
        
        if j == 0:
            temp_diff_1 = []
            index_mid = index_end - (index_end-index_start)//2            
            for j in range(2):
                if j == 0:
                    index_start2 = index_start
                    index_end2 = index_mid
                if j == 1:
                    index_start2 = index_mid
                    index_end2 = index_end 
                    
                feat_lld = temp_lld[index_start2:index_end2, :]
                
                lld_max = np.max(feat_lld,axis=0)[:,np.newaxis]
                lld_min = np.min(feat_lld,axis=0)[:,np.newaxis]
                lld_mean = np.mean(feat_lld,axis=0)[:,np.newaxis]
                lld_std = np.std(feat_lld,axis=0)[:,np.newaxis]
                
                feat = list(np.concatenate((lld_max,lld_min,lld_mean,lld_std),axis=1))

                temp_diff_1 = temp_diff_1 + feat
            temp_diff = list(np.array(temp_diff_1)[:len(temp_diff_1)//2,:] - np.array(temp_diff_1)[len(temp_diff_1)//2:,:])
            feat_topics = feat_topics + temp_diff  
            
    diff = list(np.array(feat_topics)[:len(feat),:] - np.array(feat_topics)[len(feat):2*len(feat),:])
    feat_topics = feat_topics + diff  
        
    ID.append(filename[i])
    features.append(np.array(feat_topics).reshape(-1))      

features = pd.DataFrame(features)
features['ID'] = ID
features.to_csv('../features/egemaps_counting_twice_diff12_diff1.csv',header=None,index=None)
#features.to_csv('../features/mfcc_counting_twice_diff12_diff1.csv',header=None,index=None)