#-----------------------------------------------------------------------------------------------------
#-----data: 2018.06.04
#-----feature:  17 AU
#-----part:  part1, part2, part3
#-----function: process the AU feature on 3 topics
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
path_csv = path_code +  '../data/AU/'
save_csv = [path_code + '../features/AU_part1.csv',
            path_code + '../features/AU_part2.csv',
            path_code + '../features/AU_part3.csv']
time_csv_path = path_code + '../data/times_3topics_video.csv'

feature_result_row = 17
feature_reuslt_col = 50

feature_au_final = []
stactics_feature_data=[]


time_mk = (10, 20, 30, 40, 50)
AU_number_ten = (435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451)




#get the histogram feature of 17 key Au
def AU_histogram(files, time_csv_path, video_number, count_part):

    feature_result = np.zeros((17, 50))
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

    global statistics_array
    statistics_array = []
    if time_start == time_end :
        feature_result = np.zeros((17, 50))
        statistics_array = np.zeros(34)

    else:
        for h in range(435,452):
            AU_raw=[]
            for i in range(timestamp_start, timestamp_end):
                AU_raw.append(float(AU_csv.ix[i, h]))
            AU_raw_array = np.array(AU_raw)
            statistics_array.append(AU_raw_array.max())
            statistics_array.append(AU_raw_array.min())
#        statistics_array_temp = AU_csv.loc[timestamp_start:timestamp_end, 435:452]
#        statistics_array_temp = np.array(statistics_array_temp)
#        #statistics_mean = statistics_array_temp.mean(axis=0)
#        statistics_max = statistics_array_temp.max(axis=0)
#        statistics_min = statistics_array_temp.min(axis=0)
#        statistics_array = []
#        for p in range(17):
#           # statistics_array.append(statistics_mean[p])
#            statistics_array.append(statistics_max[p])
#            statistics_array.append(statistics_min[p])

        flag = 0
        for AU_number in AU_number_ten:
            cols_temp = 0

            for mk in time_mk:
                histogram_bin = [0,0,0,0,0,0,0,0,0,0]
                if (timestamp_end - timestamp_start) - mk <= 0:
                    flag = 1
                else:
                    flag = 0

                for i in range(timestamp_start,timestamp_end):

                    if (i + mk > timestamp_end -1):
                       break

                    AU_csv_temp1 = float(AU_csv.ix[i+mk, AU_number])
                    AU_csv_temp2 = float(AU_csv.ix[i, AU_number])
                    histogram_bin_temp = AU_csv_temp1 - AU_csv_temp2

                    if  (histogram_bin_temp >= -5) and (histogram_bin_temp < -4):
                        histogram_bin[0] = histogram_bin[0]  + 1
                    elif (histogram_bin_temp >= -4) and (histogram_bin_temp < -3):
                        histogram_bin[1] = histogram_bin[1]  + 1
                    elif (histogram_bin_temp >= -3) and (histogram_bin_temp < -2):
                        histogram_bin[2] = histogram_bin[2]  + 1
                    elif (histogram_bin_temp >= -2) and (histogram_bin_temp < -1):
                        histogram_bin[3] = histogram_bin[3]  + 1
                    elif (histogram_bin_temp >= -1) and (histogram_bin_temp < 0):
                        histogram_bin[4] = histogram_bin[4]  + 1
                    elif (histogram_bin_temp >= 0) and (histogram_bin_temp < 1):
                        histogram_bin[5] = histogram_bin[5]  + 1
                    elif (histogram_bin_temp >= 1) and (histogram_bin_temp < 2):
                        histogram_bin[6] = histogram_bin[6]  + 1
                    elif (histogram_bin_temp >= 2) and (histogram_bin_temp < 3):
                        histogram_bin[7] = histogram_bin[7]  + 1
                    elif (histogram_bin_temp >= 3) and (histogram_bin_temp < 4):
                        histogram_bin[8] = histogram_bin[8]  + 1
                    else:
                        histogram_bin[9] = histogram_bin[9]  + 1

                for j in range(10):
                    if flag == 1:
                        feature_result[AU_number - AU_number_ten[0], cols_temp + j] = 0
                    else:
                        feature_result[AU_number - AU_number_ten[0], cols_temp + j] = float(histogram_bin[j]) / ((timestamp_end - timestamp_start) - mk )
                flag = 0
                cols_temp = cols_temp + 10
        statistics_array = np.array(statistics_array)
    return feature_result,statistics_array


#go through all AU csv file including the dev/test/train AU csv file, and get the AU histogram features for each subject
def batch_feacture_csv(path_csv):
    global files
    files = []
    for count_part in range(3):
        feature_au_final = []

        for file_path, sub_dirs, files_ in os.walk(path_csv):
            files = sorted(files_)
            print(file_path, sub_dirs, files)
        video_number = 0
        for file_name in files:

            print(file_name)
            temp_list = []
            feature_result = AU_histogram( file_path + file_name,time_csv_path, video_number, count_part)[0]
            stactics_feature_data = AU_histogram(file_path + file_name,time_csv_path, video_number,count_part)[1]

            temp_list = feature_result.tolist()
            temp_list = temp_list + stactics_feature_data.tolist()
            temp_list.insert(0, file_name)
            temp_list = _flatten(temp_list)

            feature_au_final.append(temp_list)
            video_number = video_number + 1

        csv_result = pd.DataFrame(feature_au_final)
        csv_result.to_csv(save_csv[count_part], header=None, index=None)


batch_feacture_csv(path_csv)