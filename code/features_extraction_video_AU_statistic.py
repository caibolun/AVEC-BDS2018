#-----------------------------------------------------------------------------------------------------
#-----data: 2018.06.16
#-----feature:  17 AU
#-----part:  part1, part2, part3
#-----function: process the AU staticstic feature on 3 topics
#-----input: AU values in each frame with timestamp
#-----output: AU statistics feature and histogram feature
#-----------------------------------------------------------------------------------------------------


import os,sys
import pandas as pd
import xlrd
import string
from _tkinter import _flatten
import numpy as np


np.set_printoptions(suppress=True)

path_code = ''
path_csv = path_code + '../data/AU/'

save_csv = [path_code + '../features/AU_staticstic_part1.csv',
            path_code + '../features/AU_staticstic_part2.csv',
            path_code + '../features/AU_staticstic_part3.csv']
time_csv_path = path_code + '../data/times_3topics_video.csv'


feature_au_final = []
stactics_feature_data=[]


AU_number_ten = (435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451)


#get AU first-order and second-order statistic functions
def AU_histogram(files, time_csv_path, video_number, count_part):

    time_csv = pd.read_csv(time_csv_path, header=None)
    time_start = time_csv.at[video_number*3+count_part,0]
    time_end = time_csv.at[video_number*3+count_part,1]
    # print video_number
    # print time_start
    # print time_end

    # os.chdir(path_csv)
    AU_csv = pd.read_csv(files, header=None)
    row_number = AU_csv.shape[0]

    timestamp_start = 0
    timestamp_end = 0
    for i in range(1,row_number):
        if isinstance(AU_csv.ix[i,2], str) == 1:
            time_last = float(AU_csv.ix[row_number-1, 2])
        else :
            time_last = AU_csv.ix[row_number-1, 2]

        if time_end >= time_last:
            time_end = time_last
        else:
            pass

        if isinstance(AU_csv.ix[i,2], str) == 1:
            temp = float(AU_csv.ix[i,2])
        else :
            temp = AU_csv.ix[i, 2]


        if  temp == time_start:
            timestamp_start = i
            # print str(timestamp_start) + 's'
        elif temp == time_end :
            timestamp_end = i
            # print str(timestamp_end) + 't'
        else:
            continue


    statistics_array = []
    if time_start == time_end :
        statistics_array = np.zeros(16*17)

    else:
        for h in range(435,452):
            AU_raw=[]
            for i in range(timestamp_start, timestamp_end):
                AU_raw.append(float(AU_csv.ix[i, h]))

            AU_raw_array = np.array(AU_raw)
            AU_raw_pd = pd.DataFrame(AU_raw)
            difference_statistics = AU_raw_pd.diff()
            difference_statistics = difference_statistics.drop(0)
            difference_statistics_pd = pd.DataFrame(difference_statistics)
            difference_statistics_array = np.array(difference_statistics_pd)

            statistics_array.append(np.array(AU_raw_pd.skew())[0])
            statistics_array.append(np.array(AU_raw_pd.kurt())[0])
            statistics_array.append(AU_raw_array.max())
            statistics_array.append(AU_raw_array.min())
            statistics_array.append(AU_raw_array.mean())
            statistics_array.append(AU_raw_array.ptp())
            statistics_array.append(AU_raw_array.var())
            statistics_array.append(AU_raw_array.std())

            statistics_array.append(np.array(difference_statistics_pd.skew())[0])
            statistics_array.append(np.array(difference_statistics_pd.kurt())[0])
            statistics_array.append(difference_statistics_array.max())
            statistics_array.append(difference_statistics_array.min())
            statistics_array.append(difference_statistics_array.mean())
            statistics_array.append(difference_statistics_array.ptp())
            statistics_array.append(difference_statistics_array.var())
            statistics_array.append(difference_statistics_array.std())

        statistics_array = np.array(statistics_array)
        #print feature_result.shape, statistics_array.shape
    return statistics_array


#go through all AU csv file including the dev/test/train AU csv file, and get AU first-order and second-order statistic functions features for each subject
def batch_feacture_csv(path_csv):
    global files
    files = []
    for count_part in range(3):
        feature_au_final = []

        for file_path, sub_dirs, files in os.walk(path_csv):
            files = sorted(files)
            print(file_path, sub_dirs, files)
    #        files_name = files[0]
     #       print files_name
        video_number = 0
        for file_name in files:


            print(file_name)
            temp_list = []
            stactics_feature_data = AU_histogram(file_path + file_name,time_csv_path, video_number, count_part)


            temp_list = stactics_feature_data.tolist()
            temp_list.insert(0, file_name)
            temp_list = _flatten(temp_list)

            feature_au_final.append(temp_list)
            video_number = video_number + 1
            #print feature_au_final


        csv_result = pd.DataFrame(feature_au_final)
        csv_result.to_csv(save_csv[count_part], header=None, index=None)


batch_feacture_csv(path_csv)