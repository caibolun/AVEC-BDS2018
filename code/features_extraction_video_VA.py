#-----------------------------------------------------------------------------------------------------
#-----data: 2018.06.06
#-----feature:  VA
#-----part:  part1, part2, part3
#-----function: process the VA feature on 3 topics
#-----input: VA values in each frame with timestamp
#-----output: VA statistics feature and Educlidean feature and its statistics feature on VA
#-----50 features in total in each sample
#-----------------------------------------------------------------------------------------------------

import numpy as np
import xlrd
import xlwt
import os,sys
from scipy.spatial.distance import pdist
import pandas as pd
from _tkinter import _flatten
import math
from numpy import nan as NaN
# import XGB_combination_parameters

np.set_printoptions(suppress=True)

# path_code = XGB_combination_parameters.path_code
path_code = ''
VA_path = path_code + '../features/VA_withtimestamp/'
save_path_statistics = [path_code + '../features/VA_process_part1.csv',
                        path_code + '../features/VA_process_part2.csv',
                        path_code + '../features/VA_process_part3.csv']

time_csv_path =  path_code + '../data/times_3topics_video.csv'

# get the Euclidean distance of two consecutive frame in Valence-Arousal coordinate space
def Euclidean_Distance_process(Excel_path, time_csv_path, video_number,count_part):
    time_csv = pd.read_csv(time_csv_path, header=None)
    time_start = time_csv.at[video_number * 3 + count_part, 0]
    time_end = time_csv.at[video_number * 3 + count_part, 1]
    # print video_number
    # print time_start
    # print time_end

    flag = 0
    if time_start == time_end:
        flag = 1
        Euclidean_Distance, valence, arousal, difference_Euclidean_Distance, difference_valence, difference_arousal = [0],[0],[0],[0],[0],[0]
    else:
        flag = 0

        ExcelFile = xlrd.open_workbook(Excel_path)
        Excel = ExcelFile.sheet_by_index(0)

        rows = Excel.nrows
        Euclidean_Distance = []
        valence, arousal = [], []

        datapoint_start = time_start - math.modf(time_start)[1]
        if (datapoint_start >= 0.0) & (datapoint_start <= 0.2):
            time_start = math.modf(time_start)[1] + 0.0
        elif (datapoint_start > 0.2) & (datapoint_start <= 0.5):
            time_start = math.modf(time_start)[1] + 0.3
        elif (datapoint_start > 0.5) & (datapoint_start <= 0.8):
            time_start = math.modf(time_start)[1] + 0.7
        elif (datapoint_start > 0.8) & (datapoint_start <= 1.0):
            time_start = math.modf(time_start)[1] + 1
        else:
            pass

        datapoint_end = time_end - math.modf(time_end)[1]
        if (datapoint_end >= 0.0) & (datapoint_end <= 0.2):
            time_end = math.modf(time_end)[1] + 0.0
        elif (datapoint_end > 0.2) & (datapoint_end <= 0.5):
            time_end = math.modf(time_end)[1] + 0.3
        elif (datapoint_end > 0.5) & (datapoint_end <= 0.8):
            time_end = math.modf(time_end)[1] + 0.7
        elif (datapoint_end > 0.8) & (datapoint_end <= 1.0):
            time_end = math.modf(time_end)[1] + 1
        else:
            pass

        timestamp_start = 0.0
        timestamp_end = 0.0

        for i in range(1, rows):

            time_last = round((Excel.cell(rows-1, 0).value)/float(1000),1)
            if time_end >= time_last:
                time_end = time_last
            else:
                pass

            time_first = 0.7
            if time_start <= time_first:
                time_start = time_first
            else:
                pass


            temp = round((Excel.cell(i, 0).value) / float(1000), 1)

            if temp == time_start:
                timestamp_start = i
                # print str(timestamp_start) + 's'
            elif temp == time_end:
                timestamp_end = i
                # print str(timestamp_end) + 't'
            else:
                pass

        while timestamp_start == 0:
            time_start = time_start + 1
            for n in range(1, rows):
                temp1 = round((Excel.cell(n, 0).value) / float(1000), 1)
                if temp1 == time_start:
                    timestamp_start = n
                    # print str(timestamp_start) + 's'
                else:
                    pass

        while timestamp_end == 0:
            time_end = time_end + 1
            for j in range(1, rows):
                temp2 = round((Excel.cell(j, 0).value) / float(1000), 1)
                if temp2 == time_end:
                    timestamp_end = j
                    # print str(timestamp_end) + 't'
                else:
                    pass

        if timestamp_end <= timestamp_start:
            flag =1
            Euclidean_Distance, valence, arousal, difference_Euclidean_Distance, difference_valence, difference_arousal = [0,0,0,0,0,0]
        else:

            for p in range(timestamp_start+1,timestamp_end):
                X1 = Excel.cell(p-1,1).value
                Y1 = Excel.cell(p-1,2).value
                valence.append(X1)
                arousal.append(Y1)
                Vector1 = [X1, Y1]
                X2 = Excel.cell(p,1).value
                Y2 = Excel.cell(p,2).value
                Vector2 = [X2, Y2]
                Vector1 = np.array(Vector1)
                Vector2 = np.array(Vector2)
                Euclidean_Distance.append(np.sqrt(np.sum(np.square(Vector1 - Vector2))))
            valence.append(X2)
            arousal.append(Y2)

            pd_Euclidean_Distance = pd.DataFrame(Euclidean_Distance)
            pd_valence = pd.DataFrame(valence)
            pd_arousal = pd.DataFrame(arousal)

            difference_Euclidean_Distance = pd_Euclidean_Distance.diff()
            difference_valence = pd_valence.diff()
            difference_arousal = pd_arousal.diff()
            difference_Euclidean_Distance.drop(0,axis=0,inplace=True)
            difference_valence.drop(0,axis=0,inplace=True)
            difference_arousal.drop(0,axis=0,inplace=True)


    #return Euclidean_Distance, valence, arousal
    return Euclidean_Distance, valence, arousal, difference_Euclidean_Distance, difference_valence, difference_arousal, flag

#respectively get the statistic functions features of valence values , arousal values  and the Euclidean distance of the consecutive frame.
def stactics_process(Euclidean_Distance,valence,arousal,difference_Euclidean_Distance,difference_valence,difference_arousal):
    statistics_Euclidean_Distance = []
    Euclidean_Distance = np.array(Euclidean_Distance)
    valence = np.array(valence)
    arousal = np.array(arousal)
    difference_Euclidean_Distance = np.array(difference_Euclidean_Distance)
    difference_valence = np.array(difference_valence)
    difference_arousal = np.array(difference_arousal)

    pd_Euclidean_Distance = pd.DataFrame(Euclidean_Distance)
    pd_valence = pd.DataFrame(valence)
    pd_arousal = pd.DataFrame(arousal)
    # print pd_Euclidean_Distance

    statistics_Euclidean_Distance.extend(pd_Euclidean_Distance.skew())
    statistics_Euclidean_Distance.extend(pd_Euclidean_Distance.kurt())

    statistics_Euclidean_Distance.append(Euclidean_Distance.max())
    statistics_Euclidean_Distance.append(Euclidean_Distance.min())
    statistics_Euclidean_Distance.append(Euclidean_Distance.mean())
    # statistics_Euclidean_Distance.append(Euclidean_Distance.mode())
    statistics_Euclidean_Distance.append(Euclidean_Distance.ptp())
    statistics_Euclidean_Distance.append(Euclidean_Distance.var())
    statistics_Euclidean_Distance.append(Euclidean_Distance.std())
    # statistics_Euclidean_Distance.append(Euclidean_Distance.cov())

    statistics_Euclidean_Distance.append(Euclidean_Distance.max() - Euclidean_Distance.min())
    histogram_Euclidean_Distance = np.histogram(Euclidean_Distance, bins=10, range=(0, 6.3))[0]
    histogram_Euclidean_Distance = histogram_Euclidean_Distance / float(np.sum(histogram_Euclidean_Distance))
    statistics_Euclidean_Distance.extend(histogram_Euclidean_Distance.tolist())

    statistics_Euclidean_Distance.append(difference_Euclidean_Distance.ptp())
    statistics_Euclidean_Distance.append(difference_Euclidean_Distance.var())
    statistics_Euclidean_Distance.append(difference_Euclidean_Distance.std())
    statistics_Euclidean_Distance.extend((pd.DataFrame(difference_Euclidean_Distance)).skew())
    statistics_Euclidean_Distance.extend((pd.DataFrame(difference_Euclidean_Distance)).kurt())

    # --------------valence------------------
    statistics_Euclidean_Distance.extend(pd_valence.skew())
    statistics_Euclidean_Distance.extend(pd_valence.kurt())

    statistics_Euclidean_Distance.append(valence.max())
    statistics_Euclidean_Distance.append(valence.min())
    statistics_Euclidean_Distance.append(valence.mean())
    # statistics_Euclidean_Distance.append(valence.mode())
    statistics_Euclidean_Distance.append(valence.ptp())
    statistics_Euclidean_Distance.append(valence.var())
    statistics_Euclidean_Distance.append(valence.std())
    # statistics_Euclidean_Distance.append(valence.cov())

    statistics_Euclidean_Distance.append(difference_valence.ptp())
    statistics_Euclidean_Distance.append(difference_valence.var())
    statistics_Euclidean_Distance.append(difference_valence.std())
    statistics_Euclidean_Distance.extend((pd.DataFrame(difference_valence)).skew())
    statistics_Euclidean_Distance.extend((pd.DataFrame(difference_valence)).kurt())

    # --------------arousal------------------
    statistics_Euclidean_Distance.extend(pd_arousal.skew())
    statistics_Euclidean_Distance.extend(pd_arousal.kurt())

    statistics_Euclidean_Distance.append(arousal.max())
    statistics_Euclidean_Distance.append(arousal.min())
    statistics_Euclidean_Distance.append(arousal.mean())
    # statistics_Euclidean_Distance.append(arousal.mode())
    statistics_Euclidean_Distance.append(arousal.ptp())
    statistics_Euclidean_Distance.append(arousal.var())
    statistics_Euclidean_Distance.append(arousal.std())
    # statistics_Euclidean_Distance.append(arousal.cov())

    statistics_Euclidean_Distance.append(difference_arousal.ptp())
    statistics_Euclidean_Distance.append(difference_arousal.var())
    statistics_Euclidean_Distance.append(difference_arousal.std())
    statistics_Euclidean_Distance.extend((pd.DataFrame(difference_arousal)).skew())
    statistics_Euclidean_Distance.extend((pd.DataFrame(difference_arousal)).kurt())

    return statistics_Euclidean_Distance




#go through all VA file including the dev/test/train VA file, and get VA statistic functions features for each subject
def batch_process(file_path_):
    global files
    files = []
    for count_part in range(3):

        for file_path, sub_dirs, files_ in os.walk(file_path_):
            files = sorted(files_)
            print (file_path, sub_dirs, files)

        video_number = 0
        statistics = []
        temp_array = []
        for t in range(50):
            temp_array.append(0)


        for file_name in files:
            print(file_name)
            flag_global = 0
            statistics_Euclidean_Distance_all = []
            statistics_Euclidean_Distance_all.append(file_name)

            Euclidean_Distance, valence, arousal, difference_Euclidean_Distance, difference_valence, difference_arousal, flag_global = Euclidean_Distance_process(file_path  +file_name, time_csv_path, video_number,count_part)
            #Euclidean_Distance, valence, arousal, difference_Euclidean_Distance, difference_valence, difference_arousal, flag_global = Euclidean_Distance_process(file_path + 'test_010.xls', time_csv_path, 69)



            if flag_global == 1:
                statistics_Euclidean_Distance_all.extend(temp_array)
                statistics.append(statistics_Euclidean_Distance_all)
                #print statistics_Euclidean_Distance_all
            else:
                statistics_Euclidean_Distance_temp = stactics_process(Euclidean_Distance, valence, arousal, difference_Euclidean_Distance, difference_valence,difference_arousal)
                statistics_Euclidean_Distance_all.extend(statistics_Euclidean_Distance_temp)

                #print statistics_Euclidean_Distance
                statistics.append(statistics_Euclidean_Distance_all)
            #print statistics

            video_number = video_number + 1


        statistics = pd.DataFrame(statistics)
        statistics.to_csv(save_path_statistics[count_part], header=None, index=None)






def Get_null():
    for null_number in range(3):
        process_file = pd.read_csv(save_path_statistics[null_number], header =None)
        for i in range(len(process_file)):
            for j in range(1,51):
                if np.isnan(process_file.ix[i,j]):
                    process_file.ix[i, j] = 0
                else:
                    pass
        # process_file.fillna(0)
        process_file.to_csv(save_path_statistics[null_number],header=None, index=None )


batch_process(VA_path)
Get_null()






