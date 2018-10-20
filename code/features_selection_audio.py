# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:18:06 2018

@author: fan
"""


import numpy as np
import pandas as pd


## import labels
info = pd.read_csv('../data/labels_metadata.csv')
ID = info['Instance_name']
SubjectID = info['SubjectID']
gender = info['Gender']
YMRS = info['Total_YMRS']
level = info['ManiaLevel']
partition = info['Partition']
    
train_level = np.array(level[partition == 'train'])
dev_level = np.array(level[partition == 'dev'])

## import features
features_orig = pd.read_csv('../features/egemaps_mfcc.csv', header=None)
features_ID = features_orig.iloc[:,-1]
features_orig = np.float64(features_orig.iloc[:,0:-1])

features_counting = pd.read_csv('../features/egemaps_mfcc_counting.csv', header=None)
features_counting = np.float64(features_counting.iloc[:,0:-1])

features_pausetime = pd.read_csv('../features/pause_time.csv', header=None)
features_pausetime = np.float64(features_pausetime.iloc[:,0:-1])

features_length = pd.read_csv('../features/duration.csv', header=None)
features_length = np.float64(features_length.iloc[:,0:-1])

features_egemaps_7topics = pd.read_csv('../features/egemaps_7topics.csv', header=None)
features_egemaps_7topics = np.float64(features_egemaps_7topics.iloc[:,0:-1])

features_mfcc_7topics = pd.read_csv('../features/mfcc_7topics.csv', header=None)
features_mfcc_7topics = np.float64(features_mfcc_7topics.iloc[:,0:-1])

features_egemaps_diff = pd.read_csv('../features/egemaps_counting_twice_diff12_diff1.csv', header=None)
features_egemaps_diff1 = np.float64(features_egemaps_diff.iloc[:,92:184])
features_egemaps_diff12 = np.float64(features_egemaps_diff.iloc[:,276:-1])

features_mfcc_diff = pd.read_csv('../features/mfcc_counting_twice_diff12_diff1.csv', header=None)
features_mfcc_diff1 = np.float64(features_mfcc_diff.iloc[:,92:184])
features_mfcc_diff12 = np.float64(features_mfcc_diff.iloc[:,276:-1])

features = np.concatenate((features_orig, features_counting, features_egemaps_7topics, features_mfcc_7topics, features_egemaps_7topics, features_mfcc_7topics, features_egemaps_diff1, features_egemaps_diff12, features_mfcc_diff1, features_mfcc_diff12, features_pausetime, features_length), axis=1)


train_features = features[114:,:]
dev_features = features[:60,:]


from sklearn import feature_selection
fs = feature_selection.SelectKBest(feature_selection.f_classif, k=100)
fs.fit(np.concatenate((dev_features, train_features), axis=0), np.concatenate((dev_level, train_level), axis=0))

selected_features = fs.transform(features)

feature_important = pd.DataFrame(selected_features)
feature_important.to_csv('../features/audio_features.csv', header=None, index=None)
