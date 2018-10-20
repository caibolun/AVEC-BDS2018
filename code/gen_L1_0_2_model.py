# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:41:10 2018

@author: fan
"""


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn import metrics
from func_params import tuning_xgb_params
from sklearn.externals import joblib

info = pd.read_csv('../data/labels_metadata.csv')
level = info['ManiaLevel'] - 1
partition = info['Partition']

train_level = np.array(level[partition == 'train'])
dev_level = np.array(level[partition == 'dev'])

modality = 'audio'
features = np.array(pd.read_csv('../features/'+modality+'_features.csv',header=None))
features = preprocessing.scale(features)

train_features = features[114:,:]
dev_features = features[:60,:]
test_features = features[60:114,:]

train_features = train_features[train_level != 1]
train_level = train_level[train_level != 1]
train_level[train_level==2] = 1
dev_features = dev_features[dev_level != 1]
dev_level = dev_level[dev_level != 1]
dev_level[dev_level==2] = 1


dev_level = np.array(dev_level)
dev1_level, dev2_level = dev_level[:len(dev_level)//2], dev_level[len(dev_level)//2:]
dev1_features, dev2_features = dev_features[:len(dev_features)//2], dev_features[len(dev_features)//2:]

# balance the data
train_features = np.concatenate((train_features, train_features[train_level==0]), axis=0)
train_level = np.concatenate((train_level, train_level[train_level==0]), axis=0)
    
xgb_train = xgb.DMatrix(train_features, label=train_level)
xgb_dev1 = xgb.DMatrix(dev1_features, label=dev1_level)
xgb_dev2 = xgb.DMatrix(dev2_features)    
xgb_test = xgb.DMatrix(test_features)   

params={
        'booster':'gbtree',
        'objective': 'multi:softmax',  #loss function
        'num_class':2,   # the number of class
        
        'max_depth':2,  #the depth of the tree, the bigger value, model tend to local
        'min_child_weight':1, #sum of the weight of sample in min leaf
        'gamma':0.6,   # bigger value,  the model more conservative
        'subsample':0.7,  #the ratio of random sample in each tree
        'colsample_bytree':0.7,  #the ratio of random sample in features matrix
        'lambda':0.2,   #L2 normalize's weight parameter
        
        'silent':0 ,  #silent is 1, means that keep close on model ,which will not output any message
        'eta': 0.2,     #learning rate
        'seed':200,   #the seed of random data
        'nthread':7, #the thread , set the number of cores on CPU
}

watchlist = [(xgb_train, 'train'),(xgb_dev1, 'val')]    
params = tuning_xgb_params(params, train_features, train_level, dev1_features, dev1_level)
params['objective'] = 'multi:softprob'
xgm = xgb.train(params, xgb_train, num_boost_round=500, evals=watchlist, early_stopping_rounds=50)  


joblib.dump(xgm, '../model/div_1_3_'+modality+'.model')