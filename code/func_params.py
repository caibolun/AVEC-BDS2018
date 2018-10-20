# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:19:21 2018

@author: fan
"""

import numpy as np
import xgboost as xgb
from sklearn import metrics
                                                                                                                                                                                
def tuning_xgb_params(params, train_features, train_level, dev_features, dev_level):
    xgb_train = xgb.DMatrix(train_features, label=train_level)
    xgb_dev = xgb.DMatrix(dev_features, label=dev_level)
    watchlist = [(xgb_train, 'train'),(xgb_dev, 'val')]
    
    params_tuning=['max_depth', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'lambda', 'max_depth']
    
    for param_index in range(len(params_tuning)):
        span = np.arange(1,7,1)
        if param_index>=2 and param_index<=5:
            span = np.arange(0.1,1,0.1)
    
        uar_all_index, uar_all = [], []
        for i in span:
            uar_all_index.append(i)
            
            params[params_tuning[param_index]] = i
            
            model = xgb.train(params, xgb_train, num_boost_round=1000, evals=watchlist, early_stopping_rounds=100)
            predictions = model.predict(xgb_dev, ntree_limit=model.best_ntree_limit)
        
            UAR = sum(metrics.recall_score(dev_level,predictions,average=None))/float(3)
            uar_all.append(UAR)
        
        max_index = uar_all.index(max(uar_all))
        index = uar_all_index[max_index]
        params[params_tuning[param_index]] = index
    
    return params