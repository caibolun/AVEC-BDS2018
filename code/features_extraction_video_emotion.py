#-----------------------------------------------------------------------------------------------------
#-----data: 2018.06.02
#-----feature:  emotion
#-----part:  part1, part2, part3
#-----function: process the emotion feature on 3 topics
#-----input: emotion values in each frame with timestamp
#-----output: emotion statistics feature and histogram feature
#-----11 features in total in each sample
#-----------------------------------------------------------------------------------------------------

import numpy as np
import sys,os
import time
import xlwt
import xlrd
from PIL import Image
import pandas as pd
import string
from _tkinter import _flatten
import math


np.set_printoptions(suppress=True)

path_code = ''
path_csv = path_code + '../data/emotion/'
save_csv = [path_code + '../features/emotion_part1.csv',
            path_code + '../features/emotion_part2.csv',
            path_code + '../features/emotion_part3.csv']
time_csv_path = path_code + '../data/times_3topics_video.csv'


feature_emotion_hog = []

#get emotion features, such as the frequecy histogram of emtoion and four presense of four key emtoion: sadness, anger, happiness, surprise
def HOG(file_path,time_csv_path, video_number,count_part):
    netural_count, sadness_count, disgust_count, anger_count, surprise_count, fear_count, happiness_count = 0, 0, 0, 0, 0, 0, 0
    sadness_flag, anger_flag, happiness_flag, surprise_flag = 0, 0, 0,0
    feature = []

    ExcelFile = xlrd.open_workbook(file_path)
    Excel = ExcelFile.sheet_by_index(0)
    rows = Excel.nrows

    time_csv = pd.read_csv(time_csv_path, header=None)
    time_start = time_csv.at[video_number * 3 + count_part, 0]
    time_end = time_csv.at[video_number * 3 + count_part, 1]
    # print video_number
    # print time_start
    # print time_end

    if time_start == time_end:
        feature.append(Excel.cell(1, 0).value)
        feature.extend([0,0,0,0,0,0,0,0,0,0,0])
    else:

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

            time_last = round((Excel.cell(rows-1, 1).value)/float(1000),1)
            if time_end >= time_last:
                time_end = time_last
            else:
                pass

            time_first = 0.7
            if time_start <= time_first:
                time_start = time_first
            else:
                pass


            temp = round((Excel.cell(i, 1).value)/float(1000),1)


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
                temp1 = round((Excel.cell(n, 1).value) / float(1000), 1)
                if temp1 == time_start:
                    timestamp_start = n
                    # print str(timestamp_start) + 's'
                else:
                    pass

        while timestamp_end == 0:
            time_end = time_end + 1
            for j in range(1, rows):
                temp2 = round((Excel.cell(j, 1).value) / float(1000), 1)
                if temp2 == time_end:
                    timestamp_end = j
                    # print str(timestamp_end) + 's'
                else:
                    pass



        for i in range(timestamp_start,timestamp_end):
            if Excel.cell(i, 9).value == 0:
                netural_count = netural_count + 1
            elif Excel.cell(i, 9).value == 1:
                sadness_count = sadness_count + 1
                sadness_flag = 1
            elif Excel.cell(i, 9).value == 2:
                disgust_count = disgust_count + 1
            elif Excel.cell(i, 9).value == 3:
                anger_count = anger_count + 1
                anger_flag = 1
            elif Excel.cell(i, 9).value == 4:
                surprise_count = surprise_count + 1
                surprise_flag = 1
            elif Excel.cell(i, 9).value == 5:
                fear_count = fear_count + 1
            elif Excel.cell(i, 9).value == 6:
                happiness_count = happiness_count + 1
                happiness_flag = 1
            else:
                pass


        all_count = rows-1
        feature.append(Excel.cell(i, 0).value)
        feature.append(netural_count/float(timestamp_end - timestamp_start))
        feature.append(sadness_count/float(timestamp_end - timestamp_start))
        feature.append(disgust_count/float(timestamp_end - timestamp_start))
        feature.append(anger_count/float(timestamp_end - timestamp_start))
        feature.append(surprise_count/float(timestamp_end - timestamp_start))
        feature.append(fear_count/float(timestamp_end - timestamp_start))
        feature.append(happiness_count/float(timestamp_end - timestamp_start))
        feature.append(sadness_flag)
        feature.append(anger_flag)
        feature.append(happiness_flag)
        feature.append(surprise_flag)
    return feature

# feature_ = HOG(excel_file_path)
# print feature_

#go through all emotion file including the dev/test/train emotion file, and get emotion features for each subject
def batch_process(file_path_):
    global files
    files = []
    for count_part in range(3):
        feature_emotion_hog = []
        for file_path, sub_dirs, files_ in os.walk(file_path_):
            files = sorted(files_)
            print (file_path, sub_dirs, files)

        video_number = 0
        for file_name in files:
            print(file_name)
            feature_emotion = HOG(file_path +file_name, time_csv_path, video_number, count_part)
            #feature_emotion = HOG(file_path + 'dev_004.mp4.xls', time_csv_path, 3)
            feature_emotion_hog.append(feature_emotion)
            video_number = video_number + 1


        csv_result = pd.DataFrame(feature_emotion_hog)
        csv_result.to_csv(save_csv[count_part], header=None, index=None)


batch_process(path_csv)


