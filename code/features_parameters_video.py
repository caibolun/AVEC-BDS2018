#-----------------------------------------------------------------------------------------------------------------------
#-----data: 2018.06.17
#-----feature: AU, emotion, VA, head, eyecontact, AUstatistic
#-----part: part1, part2, part3
#-----function: prepare all kinds of feature parameters, feature file path and label path, whose files includes dev+train+test
#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os,sys
import string


path_code = '../'
feature_path = path_code + 'features/'

#the path of  body_action_features
feature_idt_path =  feature_path + 'body_action_features.csv'

#the path of part1 features
feature_part1_path = [
    feature_path + 'AU_part1.csv',
    feature_path + 'emotion_part1.csv',
    feature_path + 'VA_process_part1.csv',
    feature_path + 'gaze_head_part1.csv',
    feature_path + 'AU_staticstic_part1.csv'

                ]
#the path of part2 features
feature_part2_path = [
    feature_path + 'AU_part2.csv',
    feature_path + 'emotion_part2.csv',
    feature_path + 'VA_process_part2.csv',
    feature_path + 'gaze_head_part2.csv',
    feature_path + 'AU_staticstic_part2.csv'
                ]
#the path of part3 features
feature_part3_path = [
    feature_path + 'AU_part3.csv',
    feature_path + 'emotion_part3.csv',
    feature_path + 'VA_process_part3.csv',
    feature_path + 'gaze_head_part3.csv',
    feature_path + 'AU_staticstic_part3.csv'
                ]
#the path of label
label_path = path_code + 'data/labels_metadata.csv'





#---------------------------------------------------------read all the features files-----------------
feature_idt = pd.read_csv(feature_idt_path, header=None)

feature_part1_AU = pd.read_csv(feature_part1_path[0], header=None)
feature_part2_AU = pd.read_csv(feature_part2_path[0], header=None)
feature_part3_AU = pd.read_csv(feature_part3_path[0], header=None)


feature_part1_emotion = pd.read_csv(feature_part1_path[1], header=None)
feature_part2_emotion = pd.read_csv(feature_part2_path[1], header=None)
feature_part3_emotion = pd.read_csv(feature_part3_path[1], header=None)

feature_part1_VA = pd.read_csv(feature_part1_path[2], header=None)
feature_part2_VA = pd.read_csv(feature_part2_path[2], header=None)
feature_part3_VA = pd.read_csv(feature_part3_path[2], header=None)

feature_part1_eye = pd.read_csv(feature_part1_path[3], header=None)
feature_part2_eye = pd.read_csv(feature_part2_path[3], header=None)
feature_part3_eye = pd.read_csv(feature_part3_path[3], header=None)

feature_part1_AU_statistic = pd.read_csv(feature_part1_path[4], header=None)
feature_part2_AU_statistic = pd.read_csv(feature_part2_path[4], header=None)
feature_part3_AU_statistic = pd.read_csv(feature_part3_path[4], header=None)

label = pd.read_csv(label_path)


#---------------------------------------------------------change format of the features to array -------------------------------------------

feature_part1_AU = np.array(feature_part1_AU.ix[:, 1:])
feature_part2_AU = np.array(feature_part2_AU.ix[:, 1:])
feature_part3_AU = np.array(feature_part3_AU.ix[:, 1:])


feature_part1_emotion = np.array(feature_part1_emotion.ix[:, 1:])
feature_part2_emotion = np.array(feature_part2_emotion.ix[:, 1:])
feature_part3_emotion = np.array(feature_part3_emotion.ix[:, 1:])

feature_part1_VA = np.array(feature_part1_VA.ix[:, 1:])
feature_part2_VA = np.array(feature_part2_VA.ix[:, 1:])
feature_part3_VA = np.array(feature_part3_VA.ix[:, 1:])


feature_part1_eye = np.array(feature_part1_eye.ix[:, 1:])
feature_part2_eye = np.array(feature_part2_eye.ix[:, 1:])
feature_part3_eye = np.array(feature_part3_eye.ix[:, 1:])

feature_part1_AU_statistic = np.array(feature_part1_AU_statistic.ix[:, 1:])
feature_part2_AU_statistic = np.array(feature_part2_AU_statistic.ix[:, 1:])
feature_part3_AU_statistic = np.array(feature_part3_AU_statistic.ix[:, 1:])

feature_idt = np.array(feature_idt.ix[:, 1:])

label_all = label.ix[:, 'ManiaLevel']




