#-----------------------------------------------------------------------------------------------------
#-----data: 2018.06.10
#-----function: select 50 most important freatures of all prepared video features
#-----input: all features file from video
#-----output: 50-dimensional video features of each subject
#-----------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn import feature_selection
import features_parameters_video

np.set_printoptions(suppress=True)

feature_selection_save_path =  '../features/video_features.csv'

#prepare all features from video
feature_all = np.concatenate((
    features_parameters_video.feature_part1_AU,
    features_parameters_video.feature_part1_AU_statistic,
    features_parameters_video.feature_part1_emotion,
    features_parameters_video.feature_part1_VA,
    features_parameters_video.feature_part1_eye,

    features_parameters_video.feature_part2_AU,
    features_parameters_video.feature_part2_AU_statistic,
    features_parameters_video.feature_part2_emotion,
    features_parameters_video.feature_part2_VA,
    features_parameters_video.feature_part2_eye,

    features_parameters_video.feature_part3_AU,
    features_parameters_video.feature_part3_AU_statistic,
    features_parameters_video.feature_part3_emotion,
    features_parameters_video.feature_part3_VA,
    features_parameters_video.feature_part3_eye,
    features_parameters_video.feature_idt
                                ), axis=1)


feature  = np.concatenate((feature_all[:60,:], feature_all[114:218,:]), axis=0)
label = np.array(features_parameters_video.label_all)

#use sklearn f_classif function to select 50-dimentional features from all video features
fs = feature_selection.SelectKBest(feature_selection.f_classif, k=50)
fs.fit(feature, label)
selected_features = fs.transform(feature_all)

#save the selected video features as file
selected_features = pd.DataFrame(selected_features)
selected_features.to_csv(feature_selection_save_path,header=None,index=None)

