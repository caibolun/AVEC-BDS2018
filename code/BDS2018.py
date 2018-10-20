# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:29:54 2018

@author: fan
"""


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn import metrics
from sklearn.externals import joblib

# Hypermeters
thre = 0.85
thre02 = thre
thre01 = thre - 0.1
thre12 = thre - 0.1


# import features
features_audio = np.array(pd.read_csv('../features/audio_features.csv',header=None))
features_video = np.array(pd.read_csv('../features/video_features.csv',header=None))
features_semantic = np.array(pd.read_csv('../features/semantic_features.csv',header=None))

features_audio = preprocessing.scale(features_audio)
features_video = preprocessing.scale(features_video)

train_features = features_audio[114:,:]
dev_features = features_audio[:60,:]
test_features = features_audio[60:114,:]

train_features_video = features_video[114:,:]
dev_features_video = features_video[:60,:]
test_features_video = features_video[60:114,:]

train_features_semantic = features_semantic[114:,:]
dev_features_semantic = features_semantic[:60,:]
test_features_semantic = features_semantic[60:114,:]


# import labels
info = pd.read_csv('../data/labels_metadata.csv')
level = info['ManiaLevel'] - 1
partition = info['Partition']

train_level = np.array(level[partition == 'train'])
dev_level = np.array(level[partition == 'dev'])


# import models
model_0_2 = joblib.load('../model/div_1_3_audio.model')
model_0_1 = joblib.load('../model/div_1_2_audio.model')
model_1_2 = joblib.load('../model/div_2_3_audio.model')
model_v_0_1 = joblib.load('../model/div_1_2_video.model')
model_v_1_2 = joblib.load('../model/div_2_3_video.model')



## test on Dev set
xgb_dev = xgb.DMatrix(dev_features)

pred_dev_0_2 = model_0_2.predict(xgb_dev,ntree_limit=model_0_2.best_ntree_limit)
pred_dev_0_1 = model_0_1.predict(xgb_dev,ntree_limit=model_0_1.best_ntree_limit)
pred_dev_1_2 = model_1_2.predict(xgb_dev,ntree_limit=model_1_2.best_ntree_limit)

pred_dev_0_1_class = np.argmax(pred_dev_0_1, axis=1)
pred_dev_1_2_class = np.argmax(pred_dev_1_2, axis=1)

# initialize the results in dev
pred_dev_class = 10 * np.ones(len(pred_dev_0_2))

# recall the class0 and class2 from 0-2 the model in Layer1
pred_dev_class[pred_dev_0_2[:,0] > thre02] = 0
pred_dev_class[pred_dev_0_2[:,1] > thre02] = 2

# the index for the layer2
index_L2_0_1 = np.c_[pred_dev_class == 10,pred_dev_0_2[:,0]>=pred_dev_0_2[:,1]].all(axis=1)
index_L2_1_2 = np.c_[pred_dev_class == 10,pred_dev_0_2[:,0]< pred_dev_0_2[:,1]].all(axis=1)

# recall the class0 and class1 from the 0-1 model in Layer2
class_L2_0_1_temp = pred_dev_class[index_L2_0_1]
class_L2_0_1_temp[pred_dev_0_1[index_L2_0_1][:,0] > thre01] = 0
class_L2_0_1_temp[pred_dev_0_1[index_L2_0_1][:,1] > thre01] = 1
pred_dev_class[index_L2_0_1] = class_L2_0_1_temp

# recall the class1 and class2 from the 1-2 model in Layer2
class_L2_1_2_temp = pred_dev_class[index_L2_1_2]
class_L2_1_2_temp[pred_dev_1_2[index_L2_1_2][:,0] > thre12] = 1
class_L2_1_2_temp[pred_dev_1_2[index_L2_1_2][:,1] > thre12] = 2
pred_dev_class[index_L2_1_2] = class_L2_1_2_temp



# recall the uncertain samples using video and text modality
# the index for the layer3
index_L3_all = pred_dev_class==10
index_L3_0_1 = np.argmax(pred_dev_0_2[index_L3_all], axis=1) == 0
index_L3_1_2 = np.argmax(pred_dev_0_2[index_L3_all], axis=1) == 1

fetures_uncertain_video = dev_features_video[index_L3_all]
fetures_uncertain_semantic = dev_features_semantic[index_L3_all]

# recall the rest class1 and class2 from the 0-1 model in Layer3
features_L3_0_1_video = fetures_uncertain_video[index_L3_0_1]
features_L3_0_1_semantic = fetures_uncertain_semantic[index_L3_0_1]

pred_L3_0_1_v_dev = np.argmax(model_v_0_1.predict(xgb.DMatrix(features_L3_0_1_video),ntree_limit=model_v_0_1.best_ntree_limit), axis=1)
pred_L3_0_1_s_dev = np.sign(features_L3_0_1_semantic).flatten()
pred_dev_temp = pred_dev_class[index_L3_all]
pred_dev_temp[index_L3_0_1] = pred_L3_0_1_v_dev + pred_L3_0_1_s_dev
pred_dev_temp = np.array([x if x < 3 else x-1 for x in pred_dev_temp])	# restrict the results for exceeding class3
pred_dev_class[index_L3_all] = pred_dev_temp

# recall the rest class1 and class2 from the 1-2 model in Layer3
features_L3_1_2_video = fetures_uncertain_video[index_L3_1_2]
features_L3_1_2_semantic = fetures_uncertain_semantic[index_L3_1_2]

pred_L3_1_2_v_dev = np.argmax(model_v_1_2.predict(xgb.DMatrix(features_L3_1_2_video),ntree_limit=model_v_1_2.best_ntree_limit), axis=1) + 1
pred_L3_1_2_s_dev = np.sign(features_L3_1_2_semantic).flatten()
pred_dev_temp = pred_dev_class[index_L3_all]
pred_dev_temp[index_L3_1_2] = pred_L3_1_2_v_dev + pred_L3_1_2_s_dev
pred_dev_temp = np.array([x if x < 3 else x-1 for x in pred_dev_temp])	# restrict the results for exceeding class3
pred_dev_class[index_L3_all] = pred_dev_temp

# print the results in Dev set
print('Accuracy(Dev): ', np.around(sum((pred_dev_class == dev_level)/float(len(dev_level))), 4))
print('UAR(Dev): ', np.around(sum(metrics.recall_score(dev_level,pred_dev_class,average=None))/float(3), 4))

predictions_dev = pred_dev_class + 1	# transform class0 1 2 to class1 2 3
predictions_dev = pd.DataFrame(predictions_dev)
predictions_dev.to_csv('../result/predictions_dev.csv', index=None, header=None)



## test on Test set
xgb_test = xgb.DMatrix(test_features)

pred_test_0_2 = model_0_2.predict(xgb_test,ntree_limit=model_0_2.best_ntree_limit)
pred_test_0_1 = model_0_1.predict(xgb_test,ntree_limit=model_0_1.best_ntree_limit)
pred_test_1_2 = model_1_2.predict(xgb_test,ntree_limit=model_1_2.best_ntree_limit)

pred_test_0_1_class = np.argmax(pred_test_0_1, axis=1)
pred_test_1_2_class = np.argmax(pred_test_1_2, axis=1)

# initialize the results in Test
pred_test_class = 10 * np.ones(len(pred_test_0_2))

# recall the class0 and class2 from 0-2 the model in Layer1
pred_test_class[pred_test_0_2[:,0] > thre02] = 0
pred_test_class[pred_test_0_2[:,1] > thre02] = 2


# the index for the layer2
index_L2_0_1 = np.c_[pred_test_class == 10,pred_test_0_2[:,0]>=pred_test_0_2[:,1]].all(axis=1)
index_L2_1_2 = np.c_[pred_test_class == 10,pred_test_0_2[:,0]< pred_test_0_2[:,1]].all(axis=1)

# recall the class0 and class1 from the 0-1 model in Layer2
class_L2_0_1_temp = pred_test_class[index_L2_0_1]
class_L2_0_1_temp[pred_test_0_1[index_L2_0_1][:,0] > thre01] = 0
class_L2_0_1_temp[pred_test_0_1[index_L2_0_1][:,1] > thre01] = 1
pred_test_class[index_L2_0_1] = class_L2_0_1_temp

# recall the class1 and class2 from the 1-2 model in Layer2
class_L2_1_2_temp = pred_test_class[index_L2_1_2]
class_L2_1_2_temp[pred_test_1_2[index_L2_1_2][:,0] > thre12] = 1
class_L2_1_2_temp[pred_test_1_2[index_L2_1_2][:,1] > thre12] = 2
pred_test_class[index_L2_1_2] = class_L2_1_2_temp



# recall the uncertain samples using video and text modality
# the index for the layer3
index_L3_all = pred_test_class==10
index_L3_0_1 = np.argmax(pred_test_0_2[index_L3_all], axis=1) == 0
index_L3_1_2 = np.argmax(pred_test_0_2[index_L3_all], axis=1) == 1

fetures_uncertain_video = test_features_video[index_L3_all]
fetures_uncertain_semantic = test_features_semantic[index_L3_all]

# recall the rest class1 and class2 from the 0-1 model in Layer3
features_L3_0_1_video = fetures_uncertain_video[index_L3_0_1]
features_L3_0_1_semantic = fetures_uncertain_semantic[index_L3_0_1]

pred_L3_0_1_v_test_prob = model_v_0_1.predict(xgb.DMatrix(features_L3_0_1_video),ntree_limit=model_v_0_1.best_ntree_limit)
pred_L3_0_1_v_test = np.argmax(model_v_0_1.predict(xgb.DMatrix(features_L3_0_1_video),ntree_limit=model_v_0_1.best_ntree_limit), axis=1)
pred_L3_0_1_s_test = np.sign(features_L3_0_1_semantic).flatten()
pred_test_temp = pred_test_class[index_L3_all]
pred_test_temp[index_L3_0_1] = pred_L3_0_1_v_test + pred_L3_0_1_s_test
pred_test_temp = np.array([x if x < 3 else x-1 for x in pred_test_temp])	# restrict the results for exceeding class3
pred_test_class[index_L3_all] = pred_test_temp

# recall the rest class1 and class2 from the 1-2 model in Layer3
features_L3_1_2_video = fetures_uncertain_video[index_L3_1_2]
features_L3_1_2_semantic = fetures_uncertain_semantic[index_L3_1_2]
pred_L3_1_2_v_test = np.argmax(model_v_1_2.predict(xgb.DMatrix(features_L3_1_2_video),ntree_limit=model_v_1_2.best_ntree_limit), axis=1) + 1
pred_L3_1_2_s_test = np.sign(features_L3_1_2_semantic).flatten()
pred_test_temp = pred_test_class[index_L3_all]
pred_test_temp[index_L3_1_2] = pred_L3_1_2_v_test + pred_L3_1_2_s_test
pred_test_temp = np.array([x if x < 3 else x-1 for x in pred_test_temp])	# restrict the results for exceeding class3
pred_test_class[index_L3_all] = pred_test_temp

predictions_test = pred_test_class + 1
predictions_test = pd.DataFrame(predictions_test)
predictions_test.to_csv('../result/predictions_test.csv', index=None, header=None)
