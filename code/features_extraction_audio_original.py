# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:23:52 2018

@author: fan
"""


import os
import numpy as np
import pandas as pd


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
    
    egemaps_max = np.max(temp_egemaps,axis=0)	
    egemaps_min = np.min(temp_egemaps,axis=0)	 
    egemaps_mean = np.mean(temp_egemaps,axis=0)  	
    mfcc_max = np.max(temp_mfcc,axis=0)
    mfcc_min = np.min(temp_mfcc,axis=0)
    mfcc_mean = np.mean(temp_mfcc,axis=0)
    
    max_all = np.concatenate((egemaps_max,mfcc_max))[:,np.newaxis]
    min_all = np.concatenate((egemaps_min,mfcc_min))[:,np.newaxis]
    mean_all = np.concatenate((egemaps_mean,mfcc_mean))[:,np.newaxis]
    feat = np.concatenate((max_all,min_all,mean_all),axis=1)
        
    ID.append(filename[i])
    features.append(feat.reshape(-1))        

features = pd.DataFrame(features)
features['ID'] = ID
features.to_csv('../features/egemaps_mfcc.csv',header=None,index=None)